"""
员工离职预测脚本
使用保存的集成模型进行预测
"""

import pandas as pd
import numpy as np
import joblib
import logging
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 日志系统配置
# ============================================================================

def setup_logger(log_dir='log'):
    """配置日志系统"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'prediction_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

logger = setup_logger()

# ============================================================================
# 模型加载
# ============================================================================

class EnsemblePredictor:
    """集成模型预测器"""
    
    def __init__(self, model_dir='model'):
        """初始化预测器，加载所有模型"""
        self.model_dir = model_dir
        self.models = {}
        self.ensemble_info = None
        
        logger.info("=" * 80)
        logger.info("加载模型")
        logger.info("=" * 80)
        
        # 加载单个模型
        model_files = {
            'catboost': 'catboost_model.pkl',
            'xgboost': 'xgboost_model.pkl',
            'lightgbm': 'lightgbm_model.pkl'
        }
        
        for name, filename in model_files.items():
            model_path = os.path.join(model_dir, filename)
            if os.path.exists(model_path):
                self.models[name] = joblib.load(model_path)
                logger.info(f"已加载: {name} 模型")
            else:
                logger.warning(f"未找到模型文件: {model_path}")
        
        # 加载集成信息
        ensemble_path = os.path.join(model_dir, 'ensemble_info.pkl')
        if os.path.exists(ensemble_path):
            self.ensemble_info = joblib.load(ensemble_path)
            logger.info("已加载: 集成模型信息")
            logger.info(f"  权重: {self.ensemble_info['weights']}")
            logger.info(f"  阈值: {self.ensemble_info['threshold']}")
        else:
            logger.warning(f"未找到集成信息文件: {ensemble_path}")
    
    def predict(self, X):
        """使用集成模型进行预测"""
        if not self.models or not self.ensemble_info:
            raise ValueError("模型未正确加载")
        
        logger.info("\n" + "=" * 80)
        logger.info("开始预测")
        logger.info("=" * 80)
        logger.info(f"输入数据形状: {X.shape}")
        
        # 获取每个模型的预测概率
        probas = []
        for name in ['catboost', 'xgboost', 'lightgbm']:
            if name in self.models:
                proba = self.models[name].predict_proba(X)[:, 1]
                probas.append(proba)
                logger.info(f"{name.upper()} 预测完成")
        
        # 加权平均
        weights = self.ensemble_info['weights']
        ensemble_proba = sum(w * p for w, p in zip(weights, probas))
        
        # 使用阈值进行分类
        threshold = self.ensemble_info['threshold']
        predictions = (ensemble_proba > threshold).astype(int)
        
        logger.info(f"\n预测完成，共 {len(predictions)} 条记录")
        logger.info(f"预测为留任: {np.sum(predictions == 0)} ({np.sum(predictions == 0) / len(predictions) * 100:.2f}%)")
        logger.info(f"预测为离职: {np.sum(predictions == 1)} ({np.sum(predictions == 1) / len(predictions) * 100:.2f}%)")
        
        return predictions, ensemble_proba
    
    def predict_single(self, X):
        """预测单条记录"""
        predictions, probas = self.predict(X)
        return predictions[0], probas[0]

# ============================================================================
# 主程序
# ============================================================================

def main():
    """主程序：加载测试数据并进行预测"""
    logger.info("=" * 80)
    logger.info("员工离职预测系统")
    logger.info("=" * 80)
    
    # 初始化预测器
    predictor = EnsemblePredictor(model_dir='model')
    
    # 加载测试数据
    logger.info("\n加载测试数据...")
    X_test = pd.read_csv('X_test.csv')
    y_test = pd.read_csv('y_test.csv').values.ravel()
    
    logger.info(f"测试集大小: {X_test.shape}")
    
    # 进行预测
    predictions, probabilities = predictor.predict(X_test)
    
    # 计算准确率
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    
    accuracy = accuracy_score(y_test, predictions)
    logger.info("\n" + "=" * 80)
    logger.info("预测结果评估")
    logger.info("=" * 80)
    logger.info(f"\n准确率: {accuracy:.4f}")
    
    # 分类报告
    logger.info("\n分类报告:")
    report = classification_report(y_test, predictions, target_names=['留任', '离职'])
    logger.info(f"\n{report}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, predictions)
    logger.info("\n混淆矩阵:")
    logger.info(f"\n{cm}")
    
    # 保存预测结果
    results_df = pd.DataFrame({
        'actual': y_test,
        'predicted': predictions,
        'probability': probabilities
    })
    
    output_file = f'log/predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    results_df.to_csv(output_file, index=False)
    logger.info(f"\n预测结果已保存到: {output_file}")
    
    logger.info("\n" + "=" * 80)
    logger.info("预测完成！")
    logger.info("=" * 80)

if __name__ == '__main__':
    main()
