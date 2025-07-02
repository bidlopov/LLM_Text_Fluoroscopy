import joblib

# тестовый скрипт чтобы посмотреть количество слоев и функцию активации
clf = joblib.load("../embedding_classify/fluoroscopy_mlp.joblib")
print(clf)
