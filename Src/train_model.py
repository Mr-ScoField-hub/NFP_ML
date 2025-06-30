# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# 1️⃣ Load dataset
df = pd.read_csv('../data/prepared_nfp_dataset.csv')

# 2️⃣ Extract date features
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df = df.drop(columns=['Date'])

# 3️⃣ Features and target
X = df.drop('Target', axis=1)
y = df['Target']

# 4️⃣ Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5️⃣ Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 6️⃣ Balance training data with SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# 7️⃣ Setup XGBoost classifier and parameter grid for tuning
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 1.0],
    'colsample_bytree': [0.7, 1.0],
}

grid_search = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring='accuracy'
)

# 8️⃣ Train with grid search on balanced data
grid_search.fit(X_train_res, y_train_res)

best_xgb = grid_search.best_estimator_

print("🔍 Best hyperparameters:", grid_search.best_params_)

# 9️⃣ Evaluate on test data
y_pred = best_xgb.predict(X_test)

print("✅ Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\n✅ Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n✅ Classification Report:\n", classification_report(y_test, y_pred))

# 🔟 Cross-validation score
cv_scores = cross_val_score(best_xgb, X_scaled, y, cv=5)
print("\n✅ 5-Fold CV Mean Accuracy: {:.2f}%".format(cv_scores.mean() * 100))
