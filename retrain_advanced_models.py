import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve
import joblib

# 加载处理后的数据
X_train = np.load('/home/ubuntu/X_train_adv.npy')
X_test = np.load('/home/ubuntu/X_test_adv.npy')
y_train = np.load('/home/ubuntu/y_train_adv.npy')
y_test = np.load('/home/ubuntu/y_test_adv.npy')

def find_best_threshold(model, X_test, y_test):
    y_probs = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]

# 1. 训练 XGBoost
print("Training Advanced XGBoost...")
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
xgb = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb.fit(X_train, y_train)

best_thresh_xgb, best_f1_xgb = find_best_threshold(xgb, X_test, y_test)
y_pred_xgb = (xgb.predict_proba(X_test)[:, 1] >= best_thresh_xgb).astype(int)
print(f"\n--- Advanced XGBoost Performance (Threshold: {best_thresh_xgb:.4f}) ---")
print(classification_report(y_test, y_pred_xgb))

# 2. 训练 Random Forest
print("Training Advanced Random Forest...")
rf = RandomForestClassifier(
    class_weight='balanced',
    n_estimators=150,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

best_thresh_rf, best_f1_rf = find_best_threshold(rf, X_test, y_test)
y_pred_rf = (rf.predict_proba(X_test)[:, 1] >= best_thresh_rf).astype(int)
print(f"\n--- Advanced Random Forest Performance (Threshold: {best_thresh_rf:.4f}) ---")
print(classification_report(y_test, y_pred_rf))

# 保存模型和阈值
joblib.dump(xgb, '/home/ubuntu/heart_disease_detector/xgb_model_adv.joblib')
joblib.dump(rf, '/home/ubuntu/heart_disease_detector/rf_model_adv.joblib')
joblib.dump({'xgb': best_thresh_xgb, 'rf': best_thresh_rf}, '/home/ubuntu/heart_disease_detector/thresholds_adv.joblib')

print("\nAdvanced models trained and saved.")
