#!/usr/bin/env python3
"""WSGI файл для запуска приложения на Beget."""
import sys
import os

# Путь к виртуальному окружению
project_dir = os.path.dirname(os.path.abspath(__file__))
venv_path = os.path.join(project_dir, 'venv')

# Если виртуальное окружение существует, используем его
if os.path.exists(venv_path):
    venv_site_packages = os.path.join(venv_path, 'lib', 'python3.12', 'site-packages')
    # Пробуем найти правильную версию Python
    if not os.path.exists(venv_site_packages):
        # Пробуем другие версии Python
        for py_version in ['python3.11', 'python3.10', 'python3.9', 'python3.8']:
            alt_path = os.path.join(venv_path, 'lib', py_version, 'site-packages')
            if os.path.exists(alt_path):
                venv_site_packages = alt_path
                break
    
    if os.path.exists(venv_site_packages):
        if venv_site_packages not in sys.path:
            sys.path.insert(0, venv_site_packages)

# Добавляем путь к проекту в sys.path
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

# Импортируем приложение
from app import app

# Это необходимо для Passenger на Beget
application = app