#!/usr/bin/env python3
"""Добавляет номера ступеней к аккордам в тексте."""
import argparse
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


ENHARMONIC_EQUIVALENTS = {
    "AB": "G#",
    "BB": "A#",
    "CB": "B",
    "DB": "C#",
    "EB": "D#",
    "FB": "E",
    "GB": "F#",
    "E#": "F",
    "B#": "C",
}

MAJOR_SCALES = {
    "C": ["C", "Dm", "Em", "F", "G", "Am", "Bm"],
    "C#": ["C#", "D#m", "Fm", "F#", "G#", "A#m", "Cm"],
    "D": ["D", "Em", "F#m", "G", "A", "Bm", "C#m"],
    "D#": ["D#", "Fm", "Gm", "G#", "A#", "Cm", "Dm"],
    "E": ["E", "F#m", "G#m", "A", "B", "C#m", "D#m"],
    "F": ["F", "Gm", "Am", "A#", "C", "Dm", "Em"],
    "F#": ["F#", "G#m", "A#m", "B", "C#", "D#m", "Fm"],
    "G": ["G", "Am", "Bm", "C", "D", "Em", "F#m"],
    "G#": ["G#", "A#m", "Cm", "C#", "D#", "Fm", "Gm"],
    "A": ["A", "Bm", "C#m", "D", "E", "F#m", "G#m"],
    "A#": ["A#", "Cm", "Dm", "D#", "F", "Gm", "Am"],
    "B": ["B", "C#m", "D#m", "E", "F#", "G#m", "A#m"],
}

MINOR_SCALES = {
    "F#m": ["F#m", "G#", "Am", "Bm", "C#", "D", "E"],
    "Gm": ["Gm", "A", "A#m", "Cm", "D", "D#m", "F"],
    "G#m": ["G#m", "A#", "Bm", "C#m", "D#", "Em", "F#"],
    "Am": ["Am", "B", "Cm", "Dm", "E", "Fm", "G"],
    "A#m": ["A#m", "C", "C#m", "D#m", "F", "F#m", "G#"],
    "Bm": ["Bm", "C#", "Dm", "Em", "F#", "Gm", "A"],
}

# Регулярное выражение для аккордов: нота + опционально #/b + только музыкальные обозначения
# Разрешаем: простую ноту (C, D, E и т.д.) или ноту с музыкальными обозначениями
# Музыкальные обозначения: m, maj, min, sus, dim, aug, цифры, /, +, -, add
# НЕ разрешаем обычные буквы после ноты (кроме музыкальных обозначений)
# Используем негативный lookahead, чтобы исключить слова, которые не являются аккордами
# Проверяем, что после аккорда не идет строчная буква (кроме случаев, когда это часть музыкального обозначения)
# Для слеша (басовые ноты): /[A-Ga-gHh][#b]? позволяет E/G#, C/Bb и т.д.
# Убран \b в конце, так как # и b не являются word characters и \b срабатывает перед ними
CHORD_REGEX = re.compile(r"\b([A-Ga-gHh][#b]?(?:(?:m(?:aj|in)?|sus|dim|aug|add|[0-9]|/[A-Ga-gHh][#b]?|[\+\-])*)?)(?![a-z])")
CHORD_TOKEN = re.compile(r"^([A-Ga-gHh])([#b]?)(.*)$")


@dataclass(frozen=True)
class Tonality:
    key: str
    mode: str
    label: str
    chord_map: Dict[str, int]


def normalize_chord_symbol(symbol: str) -> Optional[str]:
    cleaned = symbol.strip()
    if not cleaned:
        return None
    cleaned = re.sub(r"[.,:;!?]+$", "", cleaned)
    base = cleaned.split("/")[0]
    match = CHORD_TOKEN.match(base)
    if not match:
        return None
    note = (match.group(1) + match.group(2)).upper()
    # Преобразование русской нотации: H -> B (си)
    if note == "H":
        note = "B"
    tail = match.group(3)
    tail_lower = tail.lower()
    if tail_lower.startswith("maj"):
        quality = ""
    elif tail_lower.startswith("min"):
        quality = "m"
    elif tail_lower.startswith("m") and not tail_lower.startswith("maj"):
        quality = "m"
    else:
        quality = ""
    canonical_note = ENHARMONIC_EQUIVALENTS.get(note, note)
    return f"{canonical_note}{quality}"


def normalize_key_name(name: str, mode: str) -> Optional[str]:
    symbol = normalize_chord_symbol(name)
    if not symbol:
        return None
    if mode == "major" and symbol.endswith("m"):
        return symbol[:-1] or None
    if mode == "minor" and not symbol.endswith("m"):
        return f"{symbol}m"
    return symbol


def build_tonalities() -> Tuple[List[Tonality], Dict[Tuple[str, str], Tonality]]:
    tonalities: List[Tonality] = []
    by_key: Dict[Tuple[str, str], Tonality] = {}
    for mode, scales in (("major", MAJOR_SCALES), ("minor", MINOR_SCALES)):
        for label, chords in scales.items():
            normalized_key = normalize_key_name(label, mode)
            if not normalized_key:
                continue
            chord_map: Dict[str, int] = {}
            for idx, chord in enumerate(chords, start=1):
                normalized_chord = normalize_chord_symbol(chord)
                if normalized_chord:
                    chord_map.setdefault(normalized_chord, idx)
            tonality = Tonality(
                key=normalized_key,
                mode=mode,
                label=label,
                chord_map=chord_map,
            )
            tonalities.append(tonality)
            by_key[(normalized_key, mode)] = tonality
    return tonalities, by_key


TONALITIES, TONALITY_BY_KEY = build_tonalities()


def extract_chords(text: str) -> List[str]:
    result: List[str] = []
    for match in CHORD_REGEX.finditer(text):
        normalized = normalize_chord_symbol(match.group(0))
        if normalized:
            result.append(normalized)
    return result


def _already_has_degree(text: str, idx: int) -> bool:
    length = len(text)
    while idx < length and text[idx].isspace():
        idx += 1
    if idx >= length or text[idx] != "(":
        return False
    idx += 1
    start = idx
    while idx < length and text[idx].isdigit():
        idx += 1
    return idx > start and idx < length and text[idx] == ")"


def annotate_text(text: str, chord_map: Dict[str, int], all_tonalities: Optional[List[Tonality]] = None) -> str:
    pieces: List[str] = []
    last = 0
    for match in CHORD_REGEX.finditer(text):
        start, end = match.span()
        pieces.append(text[last:start])
        chord = match.group(0)
        # Проверяем, что после аккорда не идет строчная буква (это часть слова, а не аккорд)
        if end < len(text) and text[end].islower() and text[end].isalpha():
            # Это часть слова, не аккорд - пропускаем
            pieces.append(chord)
            last = end
            continue
        # Проверяем случай, когда одиночная буква (A, B, C, D, E, F, G, H) стоит перед пробелом
        # Это может быть артикль или часть текста, а не аккорд
        # Определяем по следующему слову: если оно похоже на аккорд — текущая буква тоже аккорд
        if len(chord) == 1 and chord.upper() in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
            if end < len(text) and text[end] == ' ':
                next_char_idx = end + 1
                while next_char_idx < len(text) and text[next_char_idx] == ' ':
                    next_char_idx += 1
                if next_char_idx < len(text):
                    # Находим следующее слово
                    next_word_end = next_char_idx
                    while next_word_end < len(text) and not text[next_word_end].isspace():
                        next_word_end += 1
                    next_word = text[next_char_idx:next_word_end]
                    # Проверяем, похоже ли следующее слово на аккорд:
                    # - короткое (до 6 символов, как C#m7/G)
                    # - начинается с ноты
                    # - содержит только допустимые символы аккорда
                    is_next_word_chord = (
                        0 < len(next_word) <= 6 and
                        next_word[0].upper() in 'ABCDEFGH' and
                        all(c in 'ABCDEFGHabcdefgh#bmsujdiaog0123456789/+-' for c in next_word)
                    )
                    if not is_next_word_chord:
                        # Следующее слово — обычный текст, значит текущая буква — артикль
                        pieces.append(chord)
                        last = end
                        continue
        if _already_has_degree(text, end):
            pieces.append(chord)
        else:
            normalized = normalize_chord_symbol(chord)
            # Дополнительная проверка: если нормализованный аккорд пустой или None, это не аккорд
            if not normalized:
                pieces.append(chord)
            else:
                degree = chord_map.get(normalized or "")
                # Показываем ступень только если аккорд найден в основной тональности
                if degree:
                    pieces.append(f"{chord} ({degree})")
                else:
                    pieces.append(chord)
        last = end
    pieces.append(text[last:])
    return "".join(pieces)


def select_tonality(
    key_arg: Optional[str],
    mode_arg: Optional[str],
    chords: Sequence[str],
) -> Tonality:
    if key_arg:
        mode = (mode_arg or "major").lower()
        normalized_key = normalize_key_name(key_arg, mode)
        if not normalized_key:
            raise ValueError("Некорректное название тональности.")
        tonality = TONALITY_BY_KEY.get((normalized_key, mode))
        if not tonality:
            raise ValueError("Для указанной тональности нет данных из таблицы.")
        return tonality
    if not chords:
        raise ValueError("Не удалось найти аккорды во входном тексте.")
    allowed_modes = {mode_arg.lower()} if mode_arg else {"major", "minor"}
    best: Optional[Tuple[Tuple[int, int, int], Tonality]] = None
    for tonality in TONALITIES:
        if tonality.mode not in allowed_modes:
            continue
        hits = sum(1 for chord in chords if chord in tonality.chord_map)
        if hits == 0:
            continue
        unique_hits = len({ch for ch in chords if ch in tonality.chord_map})
        tonic_hits = sum(
            1 for chord in chords if tonality.chord_map.get(chord) == 1
        )
        score = (hits, unique_hits, tonic_hits)
        if not best or score > best[0]:
            best = (score, tonality)
    if not best:
        raise ValueError("Автоматически определить тональность не удалось.")
    return best[1]


def read_text(path: Optional[str], encoding: str) -> str:
    if path:
        with open(path, "r", encoding=encoding) as source:
            return source.read()
    return sys.stdin.read()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Добавляет номера ступеней из таблицы тональностей.",
    )
    parser.add_argument("--key", help="Тональность, например F, Bb, C#.")
    parser.add_argument(
        "--mode",
        choices=["major", "minor"],
        help="Тип лада для --key или фильтра автоопределения.",
    )
    parser.add_argument(
        "--input",
        help="Путь к файлу с аккордами; без параметра читается STDIN.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Кодировка входного файла (по умолчанию UTF-8).",
    )
    parser.add_argument(
        "--show-tonality",
        action="store_true",
        help="Печатать выбранную тональность в stderr.",
    )
    args = parser.parse_args()

    text = read_text(args.input, args.encoding)
    chords = extract_chords(text)
    try:
        tonality = select_tonality(args.key, args.mode, chords)
    except ValueError as error:
        parser.error(str(error))
    annotated = annotate_text(text, tonality.chord_map, TONALITIES)
    if args.show_tonality:
        print(
            f"Использована тональность: {tonality.label} ({tonality.mode})",
            file=sys.stderr,
        )
    sys.stdout.write(annotated)


if __name__ == "__main__":
    main()

