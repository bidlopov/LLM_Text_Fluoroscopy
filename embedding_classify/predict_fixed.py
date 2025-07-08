import pandas as pd
import joblib
import matplotlib.pyplot as plt
from extract_features_fixed import get_middle_layer_features

# загрузка обученного классификатора
clf = joblib.load("fluoroscopy_mlp_fixed.joblib")

# загрузка датасета
df = pd.read_csv("../dataset/Training_Essay_Data.csv")

# взять из всего датасета 25 первых и 25 последних
human_texts = df[df["label"] == 0].head(25)["text"].tolist()
gpt_texts = df[df["label"] == 1].tail(25)["text"].tolist()

texts = human_texts + gpt_texts
labels = ["Human"] * len(human_texts) + ["LLM"] * len(gpt_texts)
probs = []

# предсказание
for i, text in enumerate(texts):
    print(f"\n[{labels[i]} #{i+1}] {text[:80]}...")
    vec = get_middle_layer_features(text).reshape(1, -1)
    prob = clf.predict_proba(vec)[0][1]
    probs.append(prob)
    print(f"LLM-probability: {prob:.4f}")

# график result
x_labels = [f"{labels[i]} #{i+1}" for i in range(len(texts))]
colors = ["green"] * len(human_texts) + ["red"] * len(gpt_texts)
plt.figure(figsize=(12, 6))
plt.bar(x_labels, probs, color=colors)
plt.axhline(0.5, color="gray", linestyle="--", label="Threshold 0.5")
plt.ylabel("LLM Probability")
plt.title("LLM Detection (Real Dataset Examples)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("../assets/result_fixed.png")
print("График сохранён: ./assest/result_fixed.png")
