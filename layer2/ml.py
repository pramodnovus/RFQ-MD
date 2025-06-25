import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib


#step 1: load data
X=np.load(r"C:\Users\Contract.DESKTOP-AF4IHN4\Desktop\final_model\synthetic_data\X_enhanced.npy")
y=np.load(r"C:\Users\Contract.DESKTOP-AF4IHN4\Desktop\final_model\synthetic_data\y_enhanced.npy")
print("Data loaded. Shape:", X.shape)

print("Checking for NaNs in y:", np.isnan(y).sum())
if np.isnan(y).any():
    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]

# Convert to integers
y = y.astype(int)
print("Unique classes in y:", np.unique(y))

unique_classes = np.unique(y)
if len(unique_classes) < 2:
    raise ValueError(f"Not enough classes in target y: found {unique_classes}")


#step 2: train test data split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Data split: {X_train.shape[0]} train / {X_test.shape[0]} test")

#step 3: intialize and train model
model=xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    max_depth=5,
    learning_rate=0.1,
    n_estimators=100
)

print("training model...")
model.fit(X_train, y_train)
print("Model training complete")

#step 4: evaluate
y_pred=model.predict(X_test)

report = classification_report(y_test, y_pred)
with open("classification_report.txt", "w") as f:
    f.write(report)

print("\n confusion matrix")
print(confusion_matrix(y_test, y_pred))

print("\n accuracy score:")
print("Accuracy:", accuracy_score(y_test, y_pred))


#step 5: save the trained model
joblib.dump(model, "xgboost_model.joblib")
print("model saved as 'xgboost_model.joblib'")
