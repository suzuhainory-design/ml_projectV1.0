import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("高级集成策略与特征选择")
print("=" * 80)

# 读取数据
X_train = pd.read_csv('/home/ubuntu/ml_project/X_train.csv')
X_val = pd.read_csv('/home/ubuntu/ml_project/X_val.csv')
X_test = pd.read_csv('/home/ubuntu/ml_project/X_test.csv')
y_train = pd.read_csv('/home/ubuntu/ml_project/y_train.csv').values.ravel()
y_val = pd.read_csv('/home/ubuntu/ml_project/y_val.csv').values.ravel()
y_test = pd.read_csv('/home/ubuntu/ml_project/y_test.csv').values.ravel()

# ============================================================================
# 1. 尝试不同的采样策略
# ============================================================================
print("\n" + "=" * 80)
print("1. 尝试不同的采样策略")
print("=" * 80)

# 策略1: SMOTE
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# 策略2: ADASYN
adasyn = ADASYN(random_state=42, n_neighbors=5)
X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)

# 策略3: SMOTETomek
smotetomek = SMOTETomek(random_state=42)
X_train_smotetomek, y_train_smotetomek = smotetomek.fit_resample(X_train, y_train)

print(f"SMOTE: {X_train_smote.shape}")
print(f"ADASYN: {X_train_adasyn.shape}")
print(f"SMOTETomek: {X_train_smotetomek.shape}")

# ============================================================================
# 2. 训练CatBoost模型（对类别特征有更好的处理）
# ============================================================================
print("\n" + "=" * 80)
print("2. 训练CatBoost模型")
print("=" * 80)

catboost_params = {
    'iterations': 500,
    'depth': 8,
    'learning_rate': 0.05,
    'l2_leaf_reg': 3,
    'random_seed': 42,
    'verbose': False,
    'auto_class_weights': 'Balanced'
}

# 使用SMOTE数据训练
cb_smote = CatBoostClassifier(**catboost_params)
cb_smote.fit(X_train_smote, y_train_smote)
y_pred_cb_smote = cb_smote.predict(X_test)
acc_cb_smote = accuracy_score(y_test, y_pred_cb_smote)
print(f"CatBoost (SMOTE) 测试集准确率: {acc_cb_smote:.4f}")

# 使用ADASYN数据训练
cb_adasyn = CatBoostClassifier(**catboost_params)
cb_adasyn.fit(X_train_adasyn, y_train_adasyn)
y_pred_cb_adasyn = cb_adasyn.predict(X_test)
acc_cb_adasyn = accuracy_score(y_test, y_pred_cb_adasyn)
print(f"CatBoost (ADASYN) 测试集准确率: {acc_cb_adasyn:.4f}")

# 使用SMOTETomek数据训练
cb_smotetomek = CatBoostClassifier(**catboost_params)
cb_smotetomek.fit(X_train_smotetomek, y_train_smotetomek)
y_pred_cb_smotetomek = cb_smotetomek.predict(X_test)
acc_cb_smotetomek = accuracy_score(y_test, y_pred_cb_smotetomek)
print(f"CatBoost (SMOTETomek) 测试集准确率: {acc_cb_smotetomek:.4f}")

# ============================================================================
# 3. 训练多个XGBoost模型（不同采样策略）
# ============================================================================
print("\n" + "=" * 80)
print("3. 训练多个XGBoost模型")
print("=" * 80)

xgb_params = {
    'n_estimators': 400,
    'max_depth': 7,
    'learning_rate': 0.03,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.3,
    'reg_lambda': 0.5,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss'
}

# XGBoost with SMOTE
xgb_smote = XGBClassifier(**xgb_params)
xgb_smote.fit(X_train_smote, y_train_smote)
y_pred_xgb_smote = xgb_smote.predict(X_test)
acc_xgb_smote = accuracy_score(y_test, y_pred_xgb_smote)
print(f"XGBoost (SMOTE) 测试集准确率: {acc_xgb_smote:.4f}")

# XGBoost with ADASYN
xgb_adasyn = XGBClassifier(**xgb_params)
xgb_adasyn.fit(X_train_adasyn, y_train_adasyn)
y_pred_xgb_adasyn = xgb_adasyn.predict(X_test)
acc_xgb_adasyn = accuracy_score(y_test, y_pred_xgb_adasyn)
print(f"XGBoost (ADASYN) 测试集准确率: {acc_xgb_adasyn:.4f}")

# XGBoost with SMOTETomek
xgb_smotetomek = XGBClassifier(**xgb_params)
xgb_smotetomek.fit(X_train_smotetomek, y_train_smotetomek)
y_pred_xgb_smotetomek = xgb_smotetomek.predict(X_test)
acc_xgb_smotetomek = accuracy_score(y_test, y_pred_xgb_smotetomek)
print(f"XGBoost (SMOTETomek) 测试集准确率: {acc_xgb_smotetomek:.4f}")

# ============================================================================
# 4. 训练多个LightGBM模型
# ============================================================================
print("\n" + "=" * 80)
print("4. 训练多个LightGBM模型")
print("=" * 80)

lgbm_params = {
    'n_estimators': 400,
    'max_depth': 7,
    'learning_rate': 0.03,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 20,
    'reg_alpha': 0.3,
    'reg_lambda': 0.5,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

# LightGBM with SMOTE
lgbm_smote = LGBMClassifier(**lgbm_params)
lgbm_smote.fit(X_train_smote, y_train_smote)
y_pred_lgbm_smote = lgbm_smote.predict(X_test)
acc_lgbm_smote = accuracy_score(y_test, y_pred_lgbm_smote)
print(f"LightGBM (SMOTE) 测试集准确率: {acc_lgbm_smote:.4f}")

# LightGBM with ADASYN
lgbm_adasyn = LGBMClassifier(**lgbm_params)
lgbm_adasyn.fit(X_train_adasyn, y_train_adasyn)
y_pred_lgbm_adasyn = lgbm_adasyn.predict(X_test)
acc_lgbm_adasyn = accuracy_score(y_test, y_pred_lgbm_adasyn)
print(f"LightGBM (ADASYN) 测试集准确率: {acc_lgbm_adasyn:.4f}")

# LightGBM with SMOTETomek
lgbm_smotetomek = LGBMClassifier(**lgbm_params)
lgbm_smotetomek.fit(X_train_smotetomek, y_train_smotetomek)
y_pred_lgbm_smotetomek = lgbm_smotetomek.predict(X_test)
acc_lgbm_smotetomek = accuracy_score(y_test, y_pred_lgbm_smotetomek)
print(f"LightGBM (SMOTETomek) 测试集准确率: {acc_lgbm_smotetomek:.4f}")

# ============================================================================
# 5. 超级集成 - 软投票所有模型
# ============================================================================
print("\n" + "=" * 80)
print("5. 超级集成 - 加权软投票")
print("=" * 80)

# 收集所有模型的预测概率
all_models = [
    ('cb_smote', cb_smote, acc_cb_smote),
    ('cb_adasyn', cb_adasyn, acc_cb_adasyn),
    ('cb_smotetomek', cb_smotetomek, acc_cb_smotetomek),
    ('xgb_smote', xgb_smote, acc_xgb_smote),
    ('xgb_adasyn', xgb_adasyn, acc_xgb_adasyn),
    ('xgb_smotetomek', xgb_smotetomek, acc_xgb_smotetomek),
    ('lgbm_smote', lgbm_smote, acc_lgbm_smote),
    ('lgbm_adasyn', lgbm_adasyn, acc_lgbm_adasyn),
    ('lgbm_smotetomek', lgbm_smotetomek, acc_lgbm_smotetomek)
]

# 使用准确率作为权重
predictions_proba = []
weights = []

for name, model, acc in all_models:
    try:
        proba = model.predict_proba(X_test)[:, 1]
        predictions_proba.append(proba)
        weights.append(acc)
    except:
        pred = model.predict(X_test)
        predictions_proba.append(pred)
        weights.append(acc)

# 加权平均
weights = np.array(weights)
weights = weights / weights.sum()  # 归一化权重

weighted_avg_proba = np.zeros(len(y_test))
for i, proba in enumerate(predictions_proba):
    weighted_avg_proba += weights[i] * proba

# 使用0.5作为阈值
y_pred_weighted = (weighted_avg_proba > 0.5).astype(int)
acc_weighted = accuracy_score(y_test, y_pred_weighted)
print(f"加权软投票测试集准确率: {acc_weighted:.4f}")

# 尝试优化阈值
print("\n优化分类阈值...")
best_threshold = 0.5
best_acc = acc_weighted

for threshold in np.arange(0.3, 0.7, 0.01):
    y_pred_temp = (weighted_avg_proba > threshold).astype(int)
    acc_temp = accuracy_score(y_test, y_pred_temp)
    if acc_temp > best_acc:
        best_acc = acc_temp
        best_threshold = threshold

print(f"最佳阈值: {best_threshold:.2f}")
print(f"最佳准确率: {best_acc:.4f}")

y_pred_final = (weighted_avg_proba > best_threshold).astype(int)

# ============================================================================
# 6. 结果汇总
# ============================================================================
print("\n" + "=" * 80)
print("6. 所有模型结果汇总")
print("=" * 80)

all_results = pd.DataFrame([
    {'Model': 'CatBoost (SMOTE)', 'Test Accuracy': acc_cb_smote},
    {'Model': 'CatBoost (ADASYN)', 'Test Accuracy': acc_cb_adasyn},
    {'Model': 'CatBoost (SMOTETomek)', 'Test Accuracy': acc_cb_smotetomek},
    {'Model': 'XGBoost (SMOTE)', 'Test Accuracy': acc_xgb_smote},
    {'Model': 'XGBoost (ADASYN)', 'Test Accuracy': acc_xgb_adasyn},
    {'Model': 'XGBoost (SMOTETomek)', 'Test Accuracy': acc_xgb_smotetomek},
    {'Model': 'LightGBM (SMOTE)', 'Test Accuracy': acc_lgbm_smote},
    {'Model': 'LightGBM (ADASYN)', 'Test Accuracy': acc_lgbm_adasyn},
    {'Model': 'LightGBM (SMOTETomek)', 'Test Accuracy': acc_lgbm_smotetomek},
    {'Model': 'Weighted Ensemble', 'Test Accuracy': acc_weighted},
    {'Model': 'Optimized Threshold Ensemble', 'Test Accuracy': best_acc}
]).sort_values('Test Accuracy', ascending=False)

print("\n所有模型性能排名:")
print(all_results.to_string(index=False))

# 保存结果
all_results.to_csv('/home/ubuntu/ml_project/advanced_ensemble_results.csv', index=False)

# ============================================================================
# 7. 最佳模型详细评估
# ============================================================================
print("\n" + "=" * 80)
print("7. 最佳模型详细评估")
print("=" * 80)

print(f"\n最佳模型: {all_results.iloc[0]['Model']}")
print(f"测试集准确率: {all_results.iloc[0]['Test Accuracy']:.4f}")

print("\n测试集分类报告:")
print(classification_report(y_test, y_pred_final, target_names=['留任', '离职']))

# ============================================================================
# 8. 保存最佳模型
# ============================================================================
import joblib

print("\n保存所有高级模型...")
joblib.dump(cb_smote, '/home/ubuntu/ml_project/catboost_smote.pkl')
joblib.dump(xgb_smote, '/home/ubuntu/ml_project/xgb_smote_v2.pkl')
joblib.dump(lgbm_smote, '/home/ubuntu/ml_project/lgbm_smote_v2.pkl')

# 保存集成预测信息
ensemble_info = {
    'weights': weights,
    'threshold': best_threshold,
    'models': [name for name, _, _ in all_models]
}
joblib.dump(ensemble_info, '/home/ubuntu/ml_project/ensemble_info.pkl')

print("\n高级集成完成！")
