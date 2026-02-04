#!/usr/bin/env python3
"""Веб-приложение для разметки аккордов."""
from flask import Flask, render_template, request, jsonify, send_file, session
from annotate_chords import (
    extract_chords,
    select_tonality,
    annotate_text,
    TONALITIES,
)
import os
import uuid
from datetime import datetime

app = Flask(__name__)
# Секретный ключ для сессий (в продакшене лучше использовать переменную окружения)
app.secret_key = os.environ.get('SECRET_KEY', 'playnashville-chords-secret-key-2024')


def get_user_id():
    """Получить или создать уникальный ID пользователя."""
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
        session.permanent = True  # Сессия будет сохраняться дольше
    return session['user_id']


def get_user_results_dir():
    """Получить путь к папке результатов текущего пользователя."""
    user_id = get_user_id()
    user_dir = os.path.join("results", user_id)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Страница не найдена"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Внутренняя ошибка сервера"}), 500


@app.route("/")
def index():
    """Главная страница с формой."""
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    """Обработка текста с аккордами."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Неверный формат данных. Ожидается JSON."}), 400
        
        text = data.get("text", "").strip()
        # Тональность и лад определяются автоматически
        key = None
        mode = None

        if not text:
            return jsonify({"error": "Текст не может быть пустым"}), 400
    except Exception as e:
        return jsonify({"error": f"Ошибка при обработке запроса: {str(e)}"}), 400

    try:
        chords = extract_chords(text)
        if not chords:
            return jsonify({"error": "Не найдено аккордов в тексте"}), 400

        tonality = select_tonality(key, mode, chords)
        annotated = annotate_text(text, tonality.chord_map, TONALITIES)

        # Сохраняем результат в папку пользователя
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"result_{timestamp}.txt"
        user_dir = get_user_results_dir()
        filepath = os.path.join(user_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(annotated)

        return jsonify(
            {
                "success": True,
                "annotated_text": annotated,
                "tonality": f"{tonality.label} ({tonality.mode})",
                "filename": filename,
            }
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Ошибка обработки: {str(e)}"}), 500


@app.route("/download/<filename>")
def download(filename):
    """Скачать файл с результатом пользователя."""
    user_dir = get_user_results_dir()
    filepath = os.path.join(user_dir, filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True, download_name=filename)
    return jsonify({"error": "Файл не найден"}), 404


@app.route("/tonalities")
def get_tonalities():
    """Получить список доступных тональностей."""
    major_keys = sorted([t.label for t in TONALITIES if t.mode == "major"])
    minor_keys = sorted([t.label for t in TONALITIES if t.mode == "minor"])
    return jsonify({"major": major_keys, "minor": minor_keys})


@app.route("/history")
def get_history():
    """Получить список файлов истории текущего пользователя."""
    user_dir = get_user_results_dir()
    
    files = []
    for filename in os.listdir(user_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(user_dir, filename)
            stat = os.stat(filepath)
            # Читаем первую строку файла для отображения в истории
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    # Ограничиваем длину для отображения
                    if len(first_line) > 50:
                        first_line = first_line[:50] + "..."
            except:
                first_line = filename
            
            files.append({
                "filename": filename,
                "created": stat.st_mtime,
                "size": stat.st_size,
                "title": first_line if first_line else filename
            })
    
    # Сортируем по дате создания (новые первые)
    files.sort(key=lambda x: x["created"], reverse=True)
    return jsonify({"files": files})


@app.route("/history/clear", methods=["DELETE"])
def clear_all_history():
    """Очистить всю историю текущего пользователя."""
    user_dir = get_user_results_dir()
    try:
        deleted_count = 0
        for filename in os.listdir(user_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(user_dir, filename)
                os.remove(filepath)
                deleted_count += 1
        return jsonify({"success": True, "message": f"Удалено файлов: {deleted_count}"})
    except Exception as e:
        return jsonify({"error": f"Ошибка очистки истории: {str(e)}"}), 500


@app.route("/history/<filename>")
def get_history_file(filename):
    """Получить содержимое файла истории текущего пользователя."""
    user_dir = get_user_results_dir()
    filepath = os.path.join(user_dir, filename)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return jsonify({"content": content, "filename": filename})
    return jsonify({"error": "Файл не найден"}), 404


@app.route("/history/<filename>", methods=["DELETE"])
def delete_history_file(filename):
    """Удалить файл из истории текущего пользователя."""
    user_dir = get_user_results_dir()
    filepath = os.path.join(user_dir, filename)
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            return jsonify({"success": True, "message": "Файл удалён"})
        except Exception as e:
            return jsonify({"error": f"Ошибка удаления: {str(e)}"}), 500
    return jsonify({"error": "Файл не найден"}), 404


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    # host="0.0.0.0" делает приложение доступным извне (локальная сеть или интернет)
    # debug=True только для разработки, для продакшена установите debug=False
    # На Beget приложение запускается через WSGI, поэтому этот блок используется только локально
    app.run(debug=False, host="0.0.0.0", port=5000)

