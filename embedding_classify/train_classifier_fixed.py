import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import joblib

from extract_features_fixed import get_middle_layer_features

# Загружаем датасет
df = pd.read_csv("../dataset/dataset.csv")
texts = df["text"].tolist()
labels = df["label"].tolist()

# Извлекаем признаки
features = []
for text in texts:
    vec = get_middle_layer_features(text)
    features.append(vec.numpy())

X = np.stack(features)
y = np.array(labels)

# разделить на train/test-выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение MLP-классификатора
clf = MLPClassifier(hidden_layer_sizes=(512, 128), activation="tanh", max_iter=200)
clf.fit(X_train, y_train)

# результат на обучающая выборка (проверка на переобучение)
y_train_pred = clf.predict(X_train)
print("Train Performance:")
print(classification_report(y_train, y_train_pred))

# оценка на тестовой выборке (проверка на переобучение)
y_pred = clf.predict(X_test)
print("Test Performance:")
print(classification_report(y_test, y_pred))

# сохранить готовую модель чтобы заново не обучать
joblib.dump(clf, "fluoroscopy_mlp_fixed.joblib")
