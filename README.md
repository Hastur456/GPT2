# GPT-2 Implementation from Scratch

Простая и чистая реализация архитектуры GPT-2 на PyTorch с возможностью генерации текста и загрузкой предобученных весов.

## 🚀 Особенности

- ✅ Чистая реализация архитектуры GPT-2 на PyTorch
- ✅ Загрузка предобученных весов от Hugging Face (gpt2, gpt2-medium, gpt2-large, gpt2-xl)
- ✅ Генерация текста с настраиваемыми параметрами (temperature, top-k)
- ✅ Модульная и хорошо организованная структура кода
- ✅ Поддержка CPU и GPU (CUDA)
- ✅ Простота использования и понимания

## 📁 Структура проекта

```
gpt2-project/
├── src/                    # Исходный код проекта
│   ├── model/             # Компоненты модели GPT-2
│   │   ├── __init__.py
│   │   ├── config.py      # Конфигурация модели
│   │   ├── modules.py     # Базовые модули (Attention, MLP, Block)
│   │   └── gpt_model.py   # Основная модель GPT
│   ├── data/              # Обработка данных
│   │   ├── __init__.py
│   │   ├── dataset.py     # Dataset класс
│   │   └── datamodule.py  # DataModule для управления данными
│   ├── inference/         # Генерация текста
│   │   ├── __init__.py
│   │   └── generator.py   # Класс для инференса
│   └── __init__.py
├── scripts/               # Исполняемые скрипты
│   ├── inference.py       # Скрипт для генерации текста
│   └── train.py          # Скрипт для обучения (заготовка)
├── weights/               # Директория для весов модели
├── requirements.txt       # Зависимости проекта
└── README.md             # Этот файл
```

## ⚙️ Установка

1. Клонируйте репозиторий:
```bash
git clone <repository-url>
cd gpt2-project
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

Основные зависимости:
- torch == 2.8.0
- transformers == 4.55.4
- numpy
- tqdm
- scikit-learn

## 🎯 Быстрый старт

### Генерация текста с предобученной моделью

```bash
python scripts/inference.py \
    --prompt "The future of artificial intelligence" \
    --model_type gpt2 \
    --max_tokens 50 \
    --temperature 0.8 \
    --top_k 50
```

### Пример использования в коде

```python
import torch
from transformers import GPT2Tokenizer
from src.model.gpt_model import load_model
from src.inference.generator import GPT2Inference

# Автоматический выбор устройства (CUDA/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка токенизатора и модели
tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
model = load_model("", "gpt2")  # Пустая строка = загрузка из Hugging Face

# Создание инференсера
inferencer = GPT2Inference(model, tokenizer, device)

# Генерация текста
result = inferencer.generate_sequence(
    prompt="The future of AI is",
    max_new_tokens=100,
    temperature=0.7,
    top_k=50
)

print("Сгенерированный текст:")
print(result)
```

## 🔧 Использование

### Скрипт inference.py

```bash
python scripts/inference.py \
    --prompt "Ваш текст здесь" \
    --model_type [gpt2|gpt2-medium|gpt2-large|gpt2-xl] \
    --model_path "путь/к/весам" \
    --max_tokens 100 \
    --temperature 0.8 \
    --top_k 50
```

**Параметры:**
- `--prompt`: Начальный текст для генерации (обязательный)
- `--model_type`: Тип модели (по умолчанию: gpt2)
- `--model_path`: Путь к собственным весам модели (опционально)
- `--max_tokens`: Максимальное количество токенов для генерации
- `--temperature`: Температура для управления случайностью (0.1-2.0)
- `--top_k`: Top-k sampling для ограничения выбора токенов

### Поддерживаемые модели

- `gpt2` - 124M параметров
- `gpt2-medium` - 350M параметров  
- `gpt2-large` - 774M параметров
- `gpt2-xl` - 1558M параметров

## 🧠 Архитектура модели

Проект реализует полную архитектуру GPT-2:

- **Self-Attention**: Многоголовое самовнимание с маскированием
- **MLP**: Двухслойная сеть с активацией GELU
- **Transformer Blocks**: Комбинация attention и MLP с residual connections
- **Positional Embeddings**: Учет позиции токенов
- **Layer Normalization**: Нормализация перед каждым блоком

## 💾 Загрузка весов

Модель автоматически загружает веса из Hugging Face:

```python
# Загрузка предобученной модели
model = load_model("", "gpt2-medium")

# Или загрузка собственных весов
model = load_model("path/to/checkpoint.pth", "gpt2")
```

## Благодарности

- OpenAI за оригинальную архитектуру GPT-2
- Hugging Face за предоставление предобученных весов
- Сообщество PyTorch за отличную документацию

---

**Примечание**: Этот проект предназначен для образовательных целей и демонстрации работы архитектуры GPT-2. Для production использования рекомендуется использовать готовые реализации от Hugging Face или других проверенных поставщиков.