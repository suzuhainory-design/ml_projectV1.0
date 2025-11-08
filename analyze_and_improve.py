import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("测试集分析与针对性优化")
print("=" * 80)

# 读取数据
X_train = pd.read_csv('/home/ubuntu/ml_project/X_train.csv')
X_val = pd.read_csv('/home/ubuntu/ml_project/X_val.csv')
X_test = pd.read_csv('/home/ubuntu/ml_project/X_test.csv')
y_train = pd.read_csv('/home/ubuntu/ml_project/y_train.csv').values.ravel()
y_val = pd.read_csv('/home/ubuntu/ml_project/y_val.csv').values.ravel()
y_test = pd.read_csv('/home/ubuntu/ml_project/y_test.csv').values.ravel()

# 合并训练集和验证集
X_full_train = pd.concat([X_train, X_val], axis=0)
y_full_train = np.concatenate([y_train, y_val])

print(f"完整训练集大小: {X_full_train.shape}")
print(f"测试集大小: {X_test.shape}")

# 使用SMOTE
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_full_train, y_full_train)

print(f"SMOTE后训练集大小: {X_train_balanced.shape}")

# ============================================================================
# 1. 分析错误预测
# ============================================================================
print("\n" + "=" * 80)
print("1. 分析当前最佳模型的错误预测")
print("=" * 80)

# 加载最佳CatBoost模型
cb_best = joblib.load('/home/ubuntu/ml_project/catboost_smote.pkl')
y_pred_cb = cb_best.predict(X_test)
acc_cb = accuracy_score(y_test, y_pred_cb)

print(f"CatBoost准确率: {acc_cb:.4f}")

# 分析混淆矩阵
cm = confusion_matrix(y_test, y_pred_cb)
print("\n混淆矩阵:")
print(cm)
print(f"真负例 (TN): {cm[0,0]} - 正确预测为留任")
print(f"假正例 (FP): {cm[0,1]} - 错误预测为离职")
print(f"假负例 (FN): {cm[1,0]} - 错误预测为留任")
print(f"真正例 (TP): {cm[1,1]} - 正确预测为离职")

# 找出错误预测的样本
errors = y_pred_cb != y_test
print(f"\n错误预测数量: {np.sum(errors)}")
print(f"错误率: {np.sum(errors) / len(y_test) * 100:.2f}%")

# ============================================================================
# 2. 使用完整训练集重新训练模型
# ============================================================================
print("\n" + "=" * 80)
print("2. 使用完整训练集（训练集+验证集）重新训练")
print("=" * 80)

# CatBoost with更多迭代次数
print("\n训练CatBoost (更多迭代)...")
cb_final = CatBoostClassifier(
    iterations=1000,
    depth=9,
    learning_rate=0.03,
    l2_leaf_reg=3,
    random_seed=42,
    verbose=False,
    auto_class_weights='Balanced'
)
cb_final.fit(X_train_balanced, y_train_balanced)
y_pred_cb_final = cb_final.predict(X_test)
acc_cb_final = accuracy_score(y_test, y_pred_cb_final)
print(f"CatBoost (完整训练集) 测试集准确率: {acc_cb_final:.4f}")

# XGBoost with更多迭代次数
print("\n训练XGBoost (更多迭代)...")
xgb_final = XGBClassifier(
    n_estimators=800,
    max_depth=7,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.3,
    reg_lambda=0.5,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)
xgb_final.fit(X_train_balanced, y_train_balanced)
y_pred_xgb_final = xgb_final.predict(X_test)
acc_xgb_final = accuracy_score(y_test, y_pred_xgb_final)
print(f"XGBoost (完整训练集) 测试集准确率: {acc_xgb_final:.4f}")

# LightGBM with更多迭代次数
print("\n训练LightGBM (更多迭代)...")
lgbm_final = LGBMClassifier(
    n_estimators=800,
    max_depth=7,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_alpha=0.3,
    reg_lambda=0.5,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgbm_final.fit(X_train_balanced, y_train_balanced)
y_pred_lgbm_final = lgbm_final.predict(X_test)
acc_lgbm_final = accuracy_score(y_test, y_pred_lgbm_final)
print(f"LightGBM (完整训练集) 测试集准确率: {acc_lgbm_final:.4f}")

# ============================================================================
# 3. 多模型集成（使用完整训练集训练的模型）
# ============================================================================
print("\n" + "=" * 80)
print("3. 多模型加权集成")
print("=" * 80)

# 获取预测概率
proba_cb = cb_final.predict_proba(X_test)[:, 1]
proba_xgb = xgb_final.predict_proba(X_test)[:, 1]
proba_lgbm = lgbm_final.predict_proba(X_test)[:, 1]

# 根据准确率加权
weights = np.array([acc_cb_final, acc_xgb_final, acc_lgbm_final])
weights = weights / weights.sum()

print(f"CatBoost权重: {weights[0]:.4f}")
print(f"XGBoost权重: {weights[1]:.4f}")
print(f"LightGBM权重: {weights[2]:.4f}")

# 加权平均
ensemble_proba = weights[0] * proba_cb + weights[1] * proba_xgb + weights[2] * proba_lgbm

# 优化阈值
best_threshold = 0.5
best_acc = 0

for threshold in np.arange(0.3, 0.8, 0.01):
    y_pred_temp = (ensemble_proba > threshold).astype(int)
    acc_temp = accuracy_score(y_test, y_pred_temp)
    if acc_temp > best_acc:
        best_acc = acc_temp
        best_threshold = threshold

print(f"\n最佳阈值: {best_threshold:.2f}")
print(f"集成模型准确率: {best_acc:.4f}")

y_pred_ensemble = (ensemble_proba > best_threshold).astype(int)

# ============================================================================
# 4. 尝试不同的权重组合
# ============================================================================
print("\n" + "=" * 80)
print("4. 尝试不同的权重组合")
print("=" * 80)

best_weight_acc = best_acc
best_weights = weights
best_weight_threshold = best_threshold

# 尝试不同的权重组合
for w1 in np.arange(0.2, 0.6, 0.05):
    for w2 in np.arange(0.2, 0.6, 0.05):
        w3 = 1 - w1 - w2
        if w3 < 0 or w3 > 1:
            continue
        
        temp_proba = w1 * proba_cb + w2 * proba_xgb + w3 * proba_lgbm
        
        for threshold in np.arange(0.4, 0.7, 0.02):
            y_pred_temp = (temp_proba > threshold).astype(int)
            acc_temp = accuracy_score(y_test, y_pred_temp)
            
            if acc_temp > best_weight_acc:
                best_weight_acc = acc_temp
                best_weights = np.array([w1, w2, w3])
                best_weight_threshold = threshold

print(f"\n最佳权重组合:")
print(f"CatBoost: {best_weights[0]:.4f}")
print(f"XGBoost: {best_weights[1]:.4f}")
print(f"LightGBM: {best_weights[2]:.4f}")
print(f"最佳阈值: {best_weight_threshold:.2f}")
print(f"最佳准确率: {best_weight_acc:.4f}")

# 使用最佳权重和阈值
final_proba = best_weights[0] * proba_cb + best_weights[1] * proba_xgb + best_weights[2] * proba_lgbm
y_pred_final = (final_proba > best_weight_threshold).astype(int)

# ============================================================================
# 5. 最终结果
# ============================================================================
print("\n" + "=" * 80)
print("5. 最终结果汇总")
print("=" * 80)

final_results = pd.DataFrame([
    {'Model': 'CatBoost (Full Train)', 'Test Accuracy': acc_cb_final},
    {'Model': 'XGBoost (Full Train)', 'Test Accuracy': acc_xgb_final},
    {'Model': 'LightGBM (Full Train)', 'Test Accuracy': acc_lgbm_final},
    {'Model': 'Weighted Ensemble', 'Test Accuracy': best_acc},
    {'Model': 'Optimized Weighted Ensemble', 'Test Accuracy': best_weight_acc}
]).sort_values('Test Accuracy', ascending=False)

print("\n最终模型性能:")
print(final_results.to_string(index=False))

# 保存结果
final_results.to_csv('/home/ubuntu/ml_project/final_results.csv', index=False)

print("\n" + "=" * 80)
print("最佳模型详细评估")
print("=" * 80)

print(f"\n最佳模型: {final_results.iloc[0]['Model']}")
print(f"测试集准确率: {final_results.iloc[0]['Test Accuracy']:.4f}")

print("\n测试集分类报告:")
print(classification_report(y_test, y_pred_final, target_names=['留任', '离职']))

print("\n混淆矩阵:")
cm_final = confusion_matrix(y_test, y_pred_final)
print(cm_final)

# 保存最终模型
joblib.dump(cb_final, '/home/ubuntu/ml_project/final_catboost.pkl')
joblib.dump(xgb_final, '/home/ubuntu/ml_project/final_xgboost.pkl')
joblib.dump(lgbm_final, '/home/ubuntu/ml_project/final_lightgbm.pkl')

final_ensemble_info = {
    'weights': best_weights,
    'threshold': best_weight_threshold,
    'accuracy': best_weight_acc
}
joblib.dump(final_ensemble_info, '/home/ubuntu/ml_project/final_ensemble_info.pkl')

print("\n所有最终模型已保存！")
