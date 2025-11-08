import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("模型训练与优化")
print("=" * 80)

# 读取处理后的数据
X_train = pd.read_csv('/home/ubuntu/ml_project/X_train.csv')
X_val = pd.read_csv('/home/ubuntu/ml_project/X_val.csv')
X_test = pd.read_csv('/home/ubuntu/ml_project/X_test.csv')
y_train = pd.read_csv('/home/ubuntu/ml_project/y_train.csv').values.ravel()
y_val = pd.read_csv('/home/ubuntu/ml_project/y_val.csv').values.ravel()
y_test = pd.read_csv('/home/ubuntu/ml_project/y_test.csv').values.ravel()

# 读取缩放后的数据
X_train_scaled = pd.read_csv('/home/ubuntu/ml_project/X_train_scaled.csv')
X_val_scaled = pd.read_csv('/home/ubuntu/ml_project/X_val_scaled.csv')
X_test_scaled = pd.read_csv('/home/ubuntu/ml_project/X_test_scaled.csv')

print(f"训练集大小: {X_train.shape}")
print(f"验证集大小: {X_val.shape}")
print(f"测试集大小: {X_test.shape}")

# ============================================================================
# 1. 处理类别不平衡 - 使用SMOTE
# ============================================================================
print("\n" + "=" * 80)
print("1. 处理类别不平衡 - 使用SMOTE")
print("=" * 80)

print(f"原始训练集类别分布:")
print(f"类别0: {np.sum(y_train == 0)} ({np.sum(y_train == 0) / len(y_train) * 100:.2f}%)")
print(f"类别1: {np.sum(y_train == 1)} ({np.sum(y_train == 1) / len(y_train) * 100:.2f}%)")

# 使用SMOTE进行过采样
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"\nSMOTE后训练集类别分布:")
print(f"类别0: {np.sum(y_train_balanced == 0)} ({np.sum(y_train_balanced == 0) / len(y_train_balanced) * 100:.2f}%)")
print(f"类别1: {np.sum(y_train_balanced == 1)} ({np.sum(y_train_balanced == 1) / len(y_train_balanced) * 100:.2f}%)")

# ============================================================================
# 2. 基准模型训练
# ============================================================================
print("\n" + "=" * 80)
print("2. 基准模型训练")
print("=" * 80)

models = {}
results = []

# 2.1 Logistic Regression
print("\n训练 Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr.fit(X_train_scaled, y_train)
y_pred_lr_val = lr.predict(X_val_scaled)
y_pred_lr_test = lr.predict(X_test_scaled)
acc_lr_val = accuracy_score(y_val, y_pred_lr_val)
acc_lr_test = accuracy_score(y_test, y_pred_lr_test)
models['LogisticRegression'] = lr
results.append({
    'Model': 'Logistic Regression',
    'Validation Accuracy': acc_lr_val,
    'Test Accuracy': acc_lr_test
})
print(f"验证集准确率: {acc_lr_val:.4f}")
print(f"测试集准确率: {acc_lr_test:.4f}")

# 2.2 Random Forest (with SMOTE)
print("\n训练 Random Forest (with SMOTE)...")
rf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=10,
                            min_samples_leaf=4, random_state=42, n_jobs=-1)
rf.fit(X_train_balanced, y_train_balanced)
y_pred_rf_val = rf.predict(X_val)
y_pred_rf_test = rf.predict(X_test)
acc_rf_val = accuracy_score(y_val, y_pred_rf_val)
acc_rf_test = accuracy_score(y_test, y_pred_rf_test)
models['RandomForest'] = rf
results.append({
    'Model': 'Random Forest (SMOTE)',
    'Validation Accuracy': acc_rf_val,
    'Test Accuracy': acc_rf_test
})
print(f"验证集准确率: {acc_rf_val:.4f}")
print(f"测试集准确率: {acc_rf_test:.4f}")

# 2.3 XGBoost (with SMOTE)
print("\n训练 XGBoost (with SMOTE)...")
xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8,
                   random_state=42, n_jobs=-1, eval_metric='logloss')
xgb.fit(X_train_balanced, y_train_balanced)
y_pred_xgb_val = xgb.predict(X_val)
y_pred_xgb_test = xgb.predict(X_test)
acc_xgb_val = accuracy_score(y_val, y_pred_xgb_val)
acc_xgb_test = accuracy_score(y_test, y_pred_xgb_test)
models['XGBoost'] = xgb
results.append({
    'Model': 'XGBoost (SMOTE)',
    'Validation Accuracy': acc_xgb_val,
    'Test Accuracy': acc_xgb_test
})
print(f"验证集准确率: {acc_xgb_val:.4f}")
print(f"测试集准确率: {acc_xgb_test:.4f}")

# 2.4 LightGBM (with SMOTE)
print("\n训练 LightGBM (with SMOTE)...")
lgbm = LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8,
                     random_state=42, n_jobs=-1, verbose=-1)
lgbm.fit(X_train_balanced, y_train_balanced)
y_pred_lgbm_val = lgbm.predict(X_val)
y_pred_lgbm_test = lgbm.predict(X_test)
acc_lgbm_val = accuracy_score(y_val, y_pred_lgbm_val)
acc_lgbm_test = accuracy_score(y_test, y_pred_lgbm_test)
models['LightGBM'] = lgbm
results.append({
    'Model': 'LightGBM (SMOTE)',
    'Validation Accuracy': acc_lgbm_val,
    'Test Accuracy': acc_lgbm_test
})
print(f"验证集准确率: {acc_lgbm_val:.4f}")
print(f"测试集准确率: {acc_lgbm_test:.4f}")

# 2.5 Gradient Boosting (with SMOTE)
print("\n训练 Gradient Boosting (with SMOTE)...")
gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                               subsample=0.8, random_state=42)
gb.fit(X_train_balanced, y_train_balanced)
y_pred_gb_val = gb.predict(X_val)
y_pred_gb_test = gb.predict(X_test)
acc_gb_val = accuracy_score(y_val, y_pred_gb_val)
acc_gb_test = accuracy_score(y_test, y_pred_gb_test)
models['GradientBoosting'] = gb
results.append({
    'Model': 'Gradient Boosting (SMOTE)',
    'Validation Accuracy': acc_gb_val,
    'Test Accuracy': acc_gb_test
})
print(f"验证集准确率: {acc_gb_val:.4f}")
print(f"测试集准确率: {acc_gb_test:.4f}")

# ============================================================================
# 3. 集成学习 - Voting Classifier
# ============================================================================
print("\n" + "=" * 80)
print("3. 集成学习 - Voting Classifier")
print("=" * 80)

# 使用表现最好的几个模型进行软投票
voting_clf = VotingClassifier(
    estimators=[
        ('rf', rf),
        ('xgb', xgb),
        ('lgbm', lgbm),
        ('gb', gb)
    ],
    voting='soft',
    n_jobs=-1
)

print("\n训练 Voting Classifier...")
voting_clf.fit(X_train_balanced, y_train_balanced)
y_pred_voting_val = voting_clf.predict(X_val)
y_pred_voting_test = voting_clf.predict(X_test)
acc_voting_val = accuracy_score(y_val, y_pred_voting_val)
acc_voting_test = accuracy_score(y_test, y_pred_voting_test)
models['VotingClassifier'] = voting_clf
results.append({
    'Model': 'Voting Classifier',
    'Validation Accuracy': acc_voting_val,
    'Test Accuracy': acc_voting_test
})
print(f"验证集准确率: {acc_voting_val:.4f}")
print(f"测试集准确率: {acc_voting_test:.4f}")

# ============================================================================
# 4. 模型结果汇总
# ============================================================================
print("\n" + "=" * 80)
print("4. 模型结果汇总")
print("=" * 80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Test Accuracy', ascending=False)
print("\n所有模型性能对比:")
print(results_df.to_string(index=False))

# 保存结果
results_df.to_csv('/home/ubuntu/ml_project/model_results.csv', index=False)

# ============================================================================
# 5. 最佳模型详细评估
# ============================================================================
print("\n" + "=" * 80)
print("5. 最佳模型详细评估")
print("=" * 80)

best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name.replace(' (SMOTE)', '').replace(' ', '')]

print(f"\n最佳模型: {best_model_name}")

# 在测试集上的详细评估
if 'Voting' in best_model_name:
    y_pred_best = y_pred_voting_test
elif 'XGBoost' in best_model_name:
    y_pred_best = y_pred_xgb_test
elif 'LightGBM' in best_model_name:
    y_pred_best = y_pred_lgbm_test
elif 'Random Forest' in best_model_name:
    y_pred_best = y_pred_rf_test
elif 'Gradient' in best_model_name:
    y_pred_best = y_pred_gb_test
else:
    y_pred_best = y_pred_lr_test

print("\n测试集分类报告:")
print(classification_report(y_test, y_pred_best, target_names=['留任', '离职']))

print("\n测试集混淆矩阵:")
cm = confusion_matrix(y_test, y_pred_best)
print(cm)
print(f"\n真负例 (TN): {cm[0,0]}")
print(f"假正例 (FP): {cm[0,1]}")
print(f"假负例 (FN): {cm[1,0]}")
print(f"真正例 (TP): {cm[1,1]}")

# ============================================================================
# 6. 保存最佳模型
# ============================================================================
print("\n" + "=" * 80)
print("6. 保存模型")
print("=" * 80)

import joblib

# 保存所有模型
for name, model in models.items():
    joblib.dump(model, f'/home/ubuntu/ml_project/model_{name}.pkl')
    print(f"已保存: model_{name}.pkl")

print("\n" + "=" * 80)
print("模型训练完成！")
print("=" * 80)
print(f"\n最佳模型: {best_model_name}")
print(f"测试集准确率: {results_df.iloc[0]['Test Accuracy']:.4f}")
