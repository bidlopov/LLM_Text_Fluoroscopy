import pandas as pd

# Загружаем датасет
df = pd.read_csv("dataset/Training_Essay_Data.csv")
print("Колонки:", df.columns)

# Проверяем уникальные значения метки
print("Значения label:", df["label"].unique())

# Отбираем по 100 GPT и 100 Human
gpt_texts = df[df["label"] == 1].sample(n=100, random_state=42)
human_texts = df[df["label"] == 0].sample(n=100, random_state=42)

# Объединяем и сохраняем
balanced = pd.concat([gpt_texts, human_texts]).reset_index(drop=True)
balanced[["text", "label"]].to_csv("dataset.csv", index=False)

print("Сохранено dataset.csv (100 GPT + 100 Human)")
