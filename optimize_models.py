import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import optuna
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("超参数优化与高级集成")
print("=" * 80)

# 读取数据
X_train = pd.read_csv('/home/ubuntu/ml_project/X_train.csv')
X_val = pd.read_csv('/home/ubuntu/ml_project/X_val.csv')
X_test = pd.read_csv('/home/ubuntu/ml_project/X_test.csv')
y_train = pd.read_csv('/home/ubuntu/ml_project/y_train.csv').values.ravel()
y_val = pd.read_csv('/home/ubuntu/ml_project/y_val.csv').values.ravel()
y_test = pd.read_csv('/home/ubuntu/ml_project/y_test.csv').values.ravel()

# 使用SMOTE处理类别不平衡
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"训练集大小: {X_train_balanced.shape}")
print(f"验证集大小: {X_val.shape}")
print(f"测试集大小: {X_test.shape}")

# ============================================================================
# 1. XGBoost超参数优化
# ============================================================================
print("\n" + "=" * 80)
print("1. XGBoost超参数优化")
print("=" * 80)

def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 500),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'logloss'
    }
    
    model = XGBClassifier(**params)
    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    return accuracy

print("开始XGBoost超参数搜索（30次试验）...")
study_xgb = optuna.create_study(direction='maximize', study_name='xgb_optimization')
study_xgb.optimize(objective_xgb, n_trials=30, show_progress_bar=True)

print(f"\n最佳XGBoost参数:")
print(study_xgb.best_params)
print(f"最佳验证集准确率: {study_xgb.best_value:.4f}")

# 使用最佳参数训练XGBoost
best_xgb = XGBClassifier(**study_xgb.best_params, random_state=42, n_jobs=-1, eval_metric='logloss')
best_xgb.fit(X_train_balanced, y_train_balanced)
y_pred_xgb_test = best_xgb.predict(X_test)
acc_xgb = accuracy_score(y_test, y_pred_xgb_test)
print(f"优化后XGBoost测试集准确率: {acc_xgb:.4f}")

# ============================================================================
# 2. LightGBM超参数优化
# ============================================================================
print("\n" + "=" * 80)
print("2. LightGBM超参数优化")
print("=" * 80)

def objective_lgbm(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 500),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    model = LGBMClassifier(**params)
    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    return accuracy

print("开始LightGBM超参数搜索（30次试验）...")
study_lgbm = optuna.create_study(direction='maximize', study_name='lgbm_optimization')
study_lgbm.optimize(objective_lgbm, n_trials=30, show_progress_bar=True)

print(f"\n最佳LightGBM参数:")
print(study_lgbm.best_params)
print(f"最佳验证集准确率: {study_lgbm.best_value:.4f}")

# 使用最佳参数训练LightGBM
best_lgbm = LGBMClassifier(**study_lgbm.best_params, random_state=42, n_jobs=-1, verbose=-1)
best_lgbm.fit(X_train_balanced, y_train_balanced)
y_pred_lgbm_test = best_lgbm.predict(X_test)
acc_lgbm = accuracy_score(y_test, y_pred_lgbm_test)
print(f"优化后LightGBM测试集准确率: {acc_lgbm:.4f}")

# ============================================================================
# 3. Random Forest超参数优化
# ============================================================================
print("\n" + "=" * 80)
print("3. Random Forest超参数优化")
print("=" * 80)

def objective_rf(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 500),
        'max_depth': trial.suggest_int('max_depth', 10, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = RandomForestClassifier(**params)
    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    return accuracy

print("开始Random Forest超参数搜索（30次试验）...")
study_rf = optuna.create_study(direction='maximize', study_name='rf_optimization')
study_rf.optimize(objective_rf, n_trials=30, show_progress_bar=True)

print(f"\n最佳Random Forest参数:")
print(study_rf.best_params)
print(f"最佳验证集准确率: {study_rf.best_value:.4f}")

# 使用最佳参数训练Random Forest
best_rf = RandomForestClassifier(**study_rf.best_params, random_state=42, n_jobs=-1)
best_rf.fit(X_train_balanced, y_train_balanced)
y_pred_rf_test = best_rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf_test)
print(f"优化后Random Forest测试集准确率: {acc_rf:.4f}")

# ============================================================================
# 4. Stacking集成
# ============================================================================
print("\n" + "=" * 80)
print("4. Stacking集成")
print("=" * 80)

# 使用优化后的模型作为基学习器
base_learners = [
    ('xgb', best_xgb),
    ('lgbm', best_lgbm),
    ('rf', best_rf)
]

# 使用Logistic Regression作为元学习器
meta_learner = LogisticRegression(max_iter=1000, random_state=42)

stacking_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5,
    n_jobs=-1
)

print("训练Stacking集成模型...")
stacking_clf.fit(X_train_balanced, y_train_balanced)
y_pred_stacking_val = stacking_clf.predict(X_val)
y_pred_stacking_test = stacking_clf.predict(X_test)
acc_stacking_val = accuracy_score(y_val, y_pred_stacking_val)
acc_stacking_test = accuracy_score(y_test, y_pred_stacking_test)

print(f"Stacking验证集准确率: {acc_stacking_val:.4f}")
print(f"Stacking测试集准确率: {acc_stacking_test:.4f}")

# ============================================================================
# 5. 结果汇总
# ============================================================================
print("\n" + "=" * 80)
print("5. 优化后模型结果汇总")
print("=" * 80)

optimized_results = pd.DataFrame([
    {'Model': 'Optimized XGBoost', 'Test Accuracy': acc_xgb},
    {'Model': 'Optimized LightGBM', 'Test Accuracy': acc_lgbm},
    {'Model': 'Optimized Random Forest', 'Test Accuracy': acc_rf},
    {'Model': 'Stacking Ensemble', 'Test Accuracy': acc_stacking_test}
]).sort_values('Test Accuracy', ascending=False)

print("\n优化后模型性能对比:")
print(optimized_results.to_string(index=False))

# 保存结果
optimized_results.to_csv('/home/ubuntu/ml_project/optimized_results.csv', index=False)

# ============================================================================
# 6. 最佳模型详细评估
# ============================================================================
print("\n" + "=" * 80)
print("6. 最佳模型详细评估")
print("=" * 80)

best_model_name = optimized_results.iloc[0]['Model']
best_accuracy = optimized_results.iloc[0]['Test Accuracy']

if 'XGBoost' in best_model_name:
    y_pred_best = y_pred_xgb_test
    best_model = best_xgb
elif 'LightGBM' in best_model_name:
    y_pred_best = y_pred_lgbm_test
    best_model = best_lgbm
elif 'Random Forest' in best_model_name:
    y_pred_best = y_pred_rf_test
    best_model = best_rf
else:
    y_pred_best = y_pred_stacking_test
    best_model = stacking_clf

print(f"\n最佳模型: {best_model_name}")
print(f"测试集准确率: {best_accuracy:.4f}")

print("\n测试集分类报告:")
print(classification_report(y_test, y_pred_best, target_names=['留任', '离职']))

# ============================================================================
# 7. 保存最佳模型
# ============================================================================
print("\n" + "=" * 80)
print("7. 保存最佳模型")
print("=" * 80)

import joblib

joblib.dump(best_xgb, '/home/ubuntu/ml_project/best_xgb_optimized.pkl')
joblib.dump(best_lgbm, '/home/ubuntu/ml_project/best_lgbm_optimized.pkl')
joblib.dump(best_rf, '/home/ubuntu/ml_project/best_rf_optimized.pkl')
joblib.dump(stacking_clf, '/home/ubuntu/ml_project/stacking_ensemble.pkl')

print("已保存所有优化后的模型")

print("\n" + "=" * 80)
print("超参数优化完成！")
print("=" * 80)
