import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.calibration import calibration_curve
import joblib
import os


df = pd.read_csv("/Users/ecemceylan/Desktop/heart3/heart.csv")


df["Sex"] = df["Sex"].map({"M": 1, "F": 0})
df["ExerciseAngina"] = df["ExerciseAngina"].map({"Y": 1, "N": 0})
label_maps = {}

for col in ["ChestPainType", "RestingECG", "ST_Slope"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_maps[col] = dict(zip(le.classes_, le.transform(le.classes_)))

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


base_model = XGBClassifier(n_estimators=8, use_label_encoder=False, verbosity=0, random_state=42)
base_model.fit(X_train, y_train)


model = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_proba)
loss = log_loss(y_test, y_proba)

print("📊 Model Performance Metrics")
print("---------------------------")
print(f"✅ Accuracy       : {acc:.3f}")
print(f"📈 ROC-AUC        : {roc:.3f}")
print(f"📉 Log Loss       : {loss:.3f}")

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/calibrated_model.pkl")
joblib.dump(label_maps, "model/label_maps.pkl")

import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay


plt.figure()
RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.title("ROC Curve")
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure()
prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)

plt.plot(prob_pred, prob_true, marker='o', label='My Model')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Reference Line')
plt.xlabel("Predicted Probability")
plt.ylabel("True Probability")
plt.title("Calibration Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



