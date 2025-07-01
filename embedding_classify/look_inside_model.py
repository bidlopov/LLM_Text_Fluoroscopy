import joblib

clf = joblib.load("fluoroscopy_mlp.joblib")
print(clf)