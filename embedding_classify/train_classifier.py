import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import joblib

from extract_features import get_middle_layer_features

# Загружаем датасет
df = pd.read_csv("dataset/dataset.csv")
texts = df["text"].tolist()
labels = df["label"].tolist()

# Извлекаем признаки
features = []
for text in texts:
    vec = get_middle_layer_features(text)
    features.append(vec.numpy())

X = np.stack(features)
y = np.array(labels)

# ✂Разделяем на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучаем классификатор
clf = MLPClassifier(hidden_layer_sizes=(512, 128), activation="tanh", max_iter=200)
clf.fit(X_train, y_train)

# Оцениваем
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Сохраняем модель
joblib.dump(clf, "fluoroscopy_mlp.joblib")