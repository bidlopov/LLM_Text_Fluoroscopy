import pandas as pd

# загрузка датасета
df = pd.read_csv("../dataset/Training_Essay_Data.csv")

# взять из датасета 500 человеческого и 500 LLM-ного
gpt_texts = df[df["label"] == 1].sample(n=500, random_state=42)
human_texts = df[df["label"] == 0].sample(n=500, random_state=42)

# объединить взятые тексты и положить в отдельный csv
balanced = pd.concat([gpt_texts, human_texts]).reset_index(drop=True)
balanced[["text", "label"]].to_csv("../dataset/dataset.csv", index=False)
print("Уменьшенный датасет (500 LLM + 500 Human) сохранен: ./dataset/dataset.csv")
