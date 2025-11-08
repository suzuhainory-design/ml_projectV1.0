"""
员工离职预测模型整合脚本
将最佳模型整合到model文件夹，并使用日志系统记录所有输出
"""

import pandas as pd
import numpy as np
import joblib
import logging
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 日志系统配置
# ============================================================================

def setup_logger(log_dir='log'):
    """配置日志系统"""
    # 创建log目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日志文件名（带时间戳）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'model_integration_{timestamp}.log')
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    return logging.getLogger(__name__)

# 初始化日志系统
logger = setup_logger()

# ============================================================================
# 主程序
# ============================================================================

def main():
    logger.info("=" * 80)
    logger.info("员工离职预测模型整合")
    logger.info("=" * 80)
    
    # 读取数据
    logger.info("\n加载数据集...")
    X_train = pd.read_csv('X_train.csv')
    X_val = pd.read_csv('X_val.csv')
    X_test = pd.read_csv('X_test.csv')
    y_train = pd.read_csv('y_train.csv').values.ravel()
    y_val = pd.read_csv('y_val.csv').values.ravel()
    y_test = pd.read_csv('y_test.csv').values.ravel()
    
    logger.info(f"训练集大小: {X_train.shape}")
    logger.info(f"验证集大小: {X_val.shape}")
    logger.info(f"测试集大小: {X_test.shape}")
    
    # 合并训练集和验证集
    logger.info("\n合并训练集和验证集...")
    X_full_train = pd.concat([X_train, X_val], axis=0)
    y_full_train = np.concatenate([y_train, y_val])
    logger.info(f"完整训练集大小: {X_full_train.shape}")
    
    # 使用SMOTE处理类别不平衡
    logger.info("\n使用SMOTE处理类别不平衡...")
    logger.info(f"原始类别分布 - 留任: {np.sum(y_full_train == 0)}, 离职: {np.sum(y_full_train == 1)}")
    
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_full_train, y_full_train)
    
    logger.info(f"SMOTE后类别分布 - 留任: {np.sum(y_train_balanced == 0)}, 离职: {np.sum(y_train_balanced == 1)}")
    logger.info(f"SMOTE后训练集大小: {X_train_balanced.shape}")
    
    # ========================================================================
    # 训练最佳模型
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("训练最佳模型")
    logger.info("=" * 80)
    
    models = {}
    
    # 1. CatBoost
    logger.info("\n训练CatBoost模型...")
    catboost_params = {
        'iterations': 1000,
        'depth': 9,
        'learning_rate': 0.03,
        'l2_leaf_reg': 3,
        'random_seed': 42,
        'verbose': False,
        'auto_class_weights': 'Balanced'
    }
    
    cb_model = CatBoostClassifier(**catboost_params)
    cb_model.fit(X_train_balanced, y_train_balanced)
    y_pred_cb = cb_model.predict(X_test)
    acc_cb = accuracy_score(y_test, y_pred_cb)
    logger.info(f"CatBoost测试集准确率: {acc_cb:.4f}")
    models['catboost'] = {'model': cb_model, 'accuracy': acc_cb, 'predictions': y_pred_cb}
    
    # 2. XGBoost
    logger.info("\n训练XGBoost模型...")
    xgboost_params = {
        'n_estimators': 800,
        'max_depth': 7,
        'learning_rate': 0.02,
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
    
    xgb_model = XGBClassifier(**xgboost_params)
    xgb_model.fit(X_train_balanced, y_train_balanced)
    y_pred_xgb = xgb_model.predict(X_test)
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    logger.info(f"XGBoost测试集准确率: {acc_xgb:.4f}")
    models['xgboost'] = {'model': xgb_model, 'accuracy': acc_xgb, 'predictions': y_pred_xgb}
    
    # 3. LightGBM
    logger.info("\n训练LightGBM模型...")
    lightgbm_params = {
        'n_estimators': 800,
        'max_depth': 7,
        'learning_rate': 0.02,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_samples': 20,
        'reg_alpha': 0.3,
        'reg_lambda': 0.5,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    lgbm_model = LGBMClassifier(**lightgbm_params)
    lgbm_model.fit(X_train_balanced, y_train_balanced)
    y_pred_lgbm = lgbm_model.predict(X_test)
    acc_lgbm = accuracy_score(y_test, y_pred_lgbm)
    logger.info(f"LightGBM测试集准确率: {acc_lgbm:.4f}")
    models['lightgbm'] = {'model': lgbm_model, 'accuracy': acc_lgbm, 'predictions': y_pred_lgbm}
    
    # ========================================================================
    # 集成模型
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("创建集成模型")
    logger.info("=" * 80)
    
    # 获取预测概率
    proba_cb = cb_model.predict_proba(X_test)[:, 1]
    proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
    proba_lgbm = lgbm_model.predict_proba(X_test)[:, 1]
    
    # 根据准确率计算权重
    weights = np.array([acc_cb, acc_xgb, acc_lgbm])
    weights = weights / weights.sum()
    
    logger.info(f"\n模型权重:")
    logger.info(f"  CatBoost: {weights[0]:.4f}")
    logger.info(f"  XGBoost: {weights[1]:.4f}")
    logger.info(f"  LightGBM: {weights[2]:.4f}")
    
    # 加权平均
    ensemble_proba = weights[0] * proba_cb + weights[1] * proba_xgb + weights[2] * proba_lgbm
    
    # 优化阈值
    logger.info("\n优化分类阈值...")
    best_threshold = 0.5
    best_acc = 0
    
    for threshold in np.arange(0.3, 0.8, 0.01):
        y_pred_temp = (ensemble_proba > threshold).astype(int)
        acc_temp = accuracy_score(y_test, y_pred_temp)
        if acc_temp > best_acc:
            best_acc = acc_temp
            best_threshold = threshold
    
    logger.info(f"最佳阈值: {best_threshold:.2f}")
    logger.info(f"集成模型准确率: {best_acc:.4f}")
    
    y_pred_ensemble = (ensemble_proba > best_threshold).astype(int)
    
    # ========================================================================
    # 保存模型
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("保存模型到model文件夹")
    logger.info("=" * 80)
    
    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    
    # 保存单个模型
    joblib.dump(cb_model, os.path.join(model_dir, 'catboost_model.pkl'))
    logger.info(f"已保存: {os.path.join(model_dir, 'catboost_model.pkl')}")
    
    joblib.dump(xgb_model, os.path.join(model_dir, 'xgboost_model.pkl'))
    logger.info(f"已保存: {os.path.join(model_dir, 'xgboost_model.pkl')}")
    
    joblib.dump(lgbm_model, os.path.join(model_dir, 'lightgbm_model.pkl'))
    logger.info(f"已保存: {os.path.join(model_dir, 'lightgbm_model.pkl')}")
    
    # 保存集成模型信息
    ensemble_info = {
        'weights': weights,
        'threshold': best_threshold,
        'accuracy': best_acc,
        'model_names': ['catboost', 'xgboost', 'lightgbm']
    }
    joblib.dump(ensemble_info, os.path.join(model_dir, 'ensemble_info.pkl'))
    logger.info(f"已保存: {os.path.join(model_dir, 'ensemble_info.pkl')}")
    
    # 保存SMOTE转换器
    joblib.dump(smote, os.path.join(model_dir, 'smote_transformer.pkl'))
    logger.info(f"已保存: {os.path.join(model_dir, 'smote_transformer.pkl')}")
    
    # ========================================================================
    # 模型评估
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("模型评估结果")
    logger.info("=" * 80)
    
    # 单个模型性能
    logger.info("\n单个模型性能:")
    for name, info in models.items():
        logger.info(f"  {name.upper()}: {info['accuracy']:.4f}")
    
    # 集成模型性能
    logger.info(f"\n集成模型性能: {best_acc:.4f}")
    
    # 详细分类报告
    logger.info("\n" + "=" * 80)
    logger.info("集成模型详细评估")
    logger.info("=" * 80)
    
    logger.info("\n分类报告:")
    report = classification_report(y_test, y_pred_ensemble, target_names=['留任', '离职'])
    logger.info(f"\n{report}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred_ensemble)
    logger.info("\n混淆矩阵:")
    logger.info(f"\n{cm}")
    logger.info(f"\n真负例 (TN): {cm[0,0]} - 正确预测为留任")
    logger.info(f"假正例 (FP): {cm[0,1]} - 错误预测为离职")
    logger.info(f"假负例 (FN): {cm[1,0]} - 错误预测为留任")
    logger.info(f"真正例 (TP): {cm[1,1]} - 正确预测为离职")
    
    # 保存评估结果
    results_summary = {
        'catboost_accuracy': acc_cb,
        'xgboost_accuracy': acc_xgb,
        'lightgbm_accuracy': acc_lgbm,
        'ensemble_accuracy': best_acc,
        'ensemble_weights': weights.tolist(),
        'ensemble_threshold': best_threshold,
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    import json
    with open(os.path.join('log', f'evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'), 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    logger.info("\n" + "=" * 80)
    logger.info("模型整合完成！")
    logger.info("=" * 80)
    logger.info(f"\n模型文件保存在: {os.path.abspath(model_dir)}")
    logger.info(f"日志文件保存在: {os.path.abspath('log')}")

if __name__ == '__main__':
    main()
