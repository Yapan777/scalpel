"""
conftest.py — корневой конфиг pytest.
Добавляет папку проекта в sys.path, чтобы 'from scalpel.xxx import'
работало без предварительного pip install -e .
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
