# Recipe Recommender

## Описание проекта
Приложение для поиска рецептов по введённым ингредиентам. Оно использует предобученную модель BERT для выделения ингредиентов из текста и находит наиболее подходящие рецепты из датасета.

## Функционал
- Обучение модели Named Entity Recognition (NER) на датасете рецептов.
- Выделение ингредиентов из пользовательского запроса.
- Поиск рецептов с совпадающими ингредиентами.

## Установка
Перед запуском необходимо установить все зависимости:
```bash
pip install torch transformers pandas nltk numpy sklearn gensim datasets
```

## Подготовка датасета
1. Используется датасет рецептов, содержащий заголовки, описания и ингредиенты.
2. Предобработка данных: лемматизация, токенизация.
3. Разметка ингредиентов в текстах для обучения модели.

## Обучение модели
1. Загружается предобученный `bert-base-multilingual-cased`.
2. Добавляется слой для классификации (O, B-ING, I-ING).
3. Модель обучается на размеченном датасете.
4. Итоговая модель сохраняется в `ner_model.pth`.

Ссылка на блокнот с обучением: https://colab.research.google.com/drive/1z6HwpkPBjPL3qoswdMpfF2_I9sVWCyDV?usp=sharing
## Запуск приложения
```bash
python recipe_app.py
```
При запуске приложение запрашивает список ингредиентов и выводит наиболее подходящие рецепты.

## Файлы проекта
- `recipe_app.py` — основной скрипт приложения.
- `ner_model.pth` — обученная модель.
- `tokenizer` — предобученный токенизатор BERT.
- `word2vec_recipes.model` — модель Word2Vec.

## Пример работы
![Демонстрация к проекту](https://github.com/sxannyy/project_recipes/blob/main/demo.png)

## Возможные улучшения и развитие проекта
- Поддержка нескольких языков.
- Улучшение модели NER.
- Повышение точности и доработка алгоритма для высоких результатов.
