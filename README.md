# SmartTask FAQ Service

🤖 Умный FAQ сервис для SmartTask с использованием RAG (Retrieval-Augmented Generation).

## 📋 Описание

Мини-сервис, который обрабатывает вопросы пользователей и отвечает на них, используя базу знаний компании SmartTask. Сервис использует векторный поиск для нахождения релевантных фрагментов документации и генерирует ответы с помощью LLM (Anthropic Claude или OpenAI GPT).

## 🏗️ Архитектура

```
┌─────────────────────────────────────────────────────────────────┐
│                        Web Browser                               │
│                     (static/index.html)                          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Application                         │
│                         (app/main.py)                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  /api/ask   │  │/api/documents│ │     /api/health         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
         │                  │                    │
         ▼                  ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│    Redis     │    │   ChromaDB   │    │  PostgreSQL  │
│   (cache)    │    │  (vectors)   │    │  (history)   │
└──────────────┘    └──────────────┘    └──────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│        LLM API (Anthropic/OpenAI)    │
└──────────────────────────────────────┘
```

### Компоненты

| Компонент | Технология | Назначение |
|-----------|------------|------------|
| API | FastAPI | REST API сервис |
| База данных | PostgreSQL | История запросов |
| Кэш | Redis | Кэширование ответов (TTL 1 час) |
| Векторная БД | ChromaDB | Хранение эмбеддингов документов |
| LLM | Anthropic Claude / OpenAI | Генерация ответов |

## 🚀 Быстрый старт

### Предварительные требования

- Docker и Docker Compose
- API ключ Anthropic или OpenAI

### Установка и запуск

1. **Клонируйте репозиторий:**
```bash
git clone https://github.com/your-username/smarttask-faq.git
cd smarttask-faq
```

2. **Создайте файл `.env`:**
```bash
cp .env.example .env
```

3. **Добавьте API ключ в `.env`:**
```env
ANTHROPIC_API_KEY=your_api_key_here
# или
OPENAI_API_KEY=your_api_key_here
LLM_PROVIDER=openai
```

4. **Запустите сервисы:**
```bash
docker-compose up --build
```

5. **Откройте в браузере:**
- Веб-интерфейс: http://localhost:8000
- API документация: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 📚 API Endpoints

### `GET /api/health`
Проверка состояния всех сервисов.

**Ответ:**
```json
{
  "status": "healthy",
  "postgres": "healthy",
  "redis": "healthy",
  "chromadb": "healthy"
}
```

### `POST /api/ask`
Задать вопрос FAQ системе.

**Запрос:**
```json
{
  "question": "Какие тарифные планы есть у SmartTask?"
}
```

**Ответ:**
```json
{
  "answer": "SmartTask предлагает три тарифных плана...",
  "sources": [
    {
      "document": "smarttask_overview.txt",
      "chunk": "Тарифные планы: 1. Free — до 5 пользователей..."
    }
  ],
  "tokens_used": 245,
  "response_time_ms": 1523,
  "cached": false
}
```

### `POST /api/documents`
Загрузить документ в базу знаний.

**Запрос:** `multipart/form-data` с файлом (`.txt` или `.md`)

**Ответ:**
```json
{
  "message": "Document uploaded successfully",
  "filename": "new_doc.txt",
  "chunks_created": 5
}
```

### `GET /api/history`
Получить историю запросов.

### `GET /api/stats`
Получить статистику сервиса.

## 🧪 Тестирование

### Запуск тестов

```bash
# Установка зависимостей для тестов
pip install -r requirements.txt

# Запуск всех тестов
pytest

# Запуск с покрытием
pytest --cov=app --cov-report=html

# Запуск конкретного теста
pytest tests/test_api.py -v
```

### Структура тестов

- `tests/test_api.py` - Unit тесты API endpoints
- `tests/test_rag.py` - Unit тесты RAG и кэширования
- `tests/test_integration.py` - Интеграционные тесты

## 📁 Структура проекта

```
smarttask-faq/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI приложение
│   ├── config.py            # Конфигурация
│   ├── api/
│   │   └── routes.py        # API endpoints
│   ├── db/
│   │   ├── database.py      # PostgreSQL подключение
│   │   └── models.py        # SQLAlchemy модели
│   ├── models/
│   │   └── schemas.py       # Pydantic модели
│   ├── services/
│   │   ├── cache_service.py # Redis кэширование
│   │   ├── rag_service.py   # RAG pipeline
│   │   └── llm_service.py   # LLM интеграция
│   └── utils/
│       └── logging.py       # Логирование
├── documents/               # Документы базы знаний
├── static/
│   └── index.html          # Веб-интерфейс
├── tests/
│   ├── conftest.py         # Pytest fixtures
│   ├── test_api.py
│   ├── test_rag.py
│   └── test_integration.py
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── requirements.txt
└── README.md
```

## ⚙️ Конфигурация

### Переменные окружения

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| `ANTHROPIC_API_KEY` | API ключ Anthropic | - |
| `OPENAI_API_KEY` | API ключ OpenAI | - |
| `LLM_PROVIDER` | Провайдер LLM (`anthropic` или `openai`) | `anthropic` |
| `POSTGRES_*` | Настройки PostgreSQL | см. `.env.example` |
| `REDIS_*` | Настройки Redis | см. `.env.example` |
| `CHROMA_*` | Настройки ChromaDB | см. `.env.example` |
| `CHUNK_SIZE` | Размер чанка для RAG | `500` |
| `CHUNK_OVERLAP` | Перекрытие чанков | `50` |
| `TOP_K_RESULTS` | Количество результатов поиска | `3` |
| `REDIS_CACHE_TTL` | TTL кэша в секундах | `3600` |

## 📊 Метрики и логирование

Сервис собирает следующие метрики:
- **Токены**: количество использованных токенов на запрос
- **Время ответа**: время генерации ответа в миллисекундах
- **Cache hit/miss**: попадания и промахи кэша

Логи выводятся в структурированном JSON формате (или в человекочитаемом формате при `DEBUG=true`).

## 🔒 Безопасность

- API ключи хранятся в переменных окружения
- Не hardcode чувствительных данных в коде
- Валидация входных данных через Pydantic
- Защита от XSS в веб-интерфейсе

## 📝 Лицензия

MIT License

## 🤝 Contributing

1. Fork репозитория
2. Создайте feature branch (`git checkout -b feature/amazing-feature`)
3. Commit изменения (`git commit -m 'Add amazing feature'`)
4. Push в branch (`git push origin feature/amazing-feature`)
5. Откройте Pull Request
