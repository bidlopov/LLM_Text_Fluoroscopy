import pandas as pd
import joblib
import matplotlib.pyplot as plt
from extract_features import get_middle_layer_features

# Загружаем классификатор
clf = joblib.load("fluoroscopy_mlp.joblib")

# Загружаем датасет
df = pd.read_csv("C:/Users/maxby/LLM_Detect_Project/Text-Fluoroscopy/dataset/Training_Essay_Data.csv")

# Берём первые 5 строк (label == 0) и последние 5 строк (label == 1)
human_texts = df[df["label"] == 0].head(25)["text"].tolist()
gpt_texts = df[df["label"] == 1].tail(25)["text"].tolist()

texts = human_texts + gpt_texts
labels = ["Human"] * len(human_texts) + ["GPT"] * len(gpt_texts)
probs = []

# Предсказания
for i, text in enumerate(texts):
    print(f"\n[{labels[i]} #{i+1}] {text[:80]}...")
    vec = get_middle_layer_features(text).reshape(1, -1)
    prob = clf.predict_proba(vec)[0][1]
    probs.append(prob)
    print(f"GPT-probability: {prob:.4f}")

# График
x_labels = [f"{labels[i]} #{i+1}" for i in range(len(texts))]
colors = ["green"] * len(human_texts) + ["red"] * len(gpt_texts)

plt.figure(figsize=(12, 6))
plt.bar(x_labels, probs, color=colors)
plt.axhline(0.5, color="gray", linestyle="--", label="Threshold 0.5")
plt.ylabel("GPT Probability")
plt.title("GPT Detection (Real Dataset Examples)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

# Сохраняем и открываем
plt.savefig("result.png")
print("График сохранён как result.png")