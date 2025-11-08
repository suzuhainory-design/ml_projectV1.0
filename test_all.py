"""
自动化测试脚本
测试所有核心功能
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# 测试结果记录
test_results = []

def log_test(test_name, status, message=""):
    """记录测试结果"""
    result = {
        'test': test_name,
        'status': '✅ PASS' if status else '❌ FAIL',
        'message': message
    }
    test_results.append(result)
    print(f"{result['status']} - {test_name}")
    if message:
        print(f"    {message}")

print("=" * 80)
print("自动化测试开始")
print("=" * 80)

# ============================================================================
# 测试1: 检查文件夹结构
# ============================================================================
print("\n[测试1] 检查文件夹结构...")

required_dirs = ['model', 'log']
for dir_name in required_dirs:
    exists = os.path.exists(dir_name)
    log_test(f"文件夹存在: {dir_name}/", exists)

# ============================================================================
# 测试2: 检查模型文件
# ============================================================================
print("\n[测试2] 检查模型文件...")

model_files = [
    'model/catboost_model.pkl',
    'model/xgboost_model.pkl',
    'model/lightgbm_model.pkl',
    'model/ensemble_info.pkl',
    'model/smote_transformer.pkl'
]

for file_path in model_files:
    exists = os.path.exists(file_path)
    if exists:
        size = os.path.getsize(file_path)
        log_test(f"模型文件: {file_path}", True, f"大小: {size/1024:.1f} KB")
    else:
        log_test(f"模型文件: {file_path}", False, "文件不存在")

# ============================================================================
# 测试3: 检查数据文件
# ============================================================================
print("\n[测试3] 检查数据文件...")

data_files = [
    'X_train.csv',
    'X_val.csv',
    'X_test.csv',
    'y_train.csv',
    'y_val.csv',
    'y_test.csv'
]

for file_path in data_files:
    exists = os.path.exists(file_path)
    if exists:
        df = pd.read_csv(file_path)
        log_test(f"数据文件: {file_path}", True, f"形状: {df.shape}")
    else:
        log_test(f"数据文件: {file_path}", False, "文件不存在")

# ============================================================================
# 测试4: 测试模型加载
# ============================================================================
print("\n[测试4] 测试模型加载...")

try:
    from predict import EnsemblePredictor
    predictor = EnsemblePredictor(model_dir='model')
    log_test("EnsemblePredictor初始化", True, "成功加载所有模型")
except Exception as e:
    log_test("EnsemblePredictor初始化", False, str(e))
    sys.exit(1)

# ============================================================================
# 测试5: 测试批量预测
# ============================================================================
print("\n[测试5] 测试批量预测...")

try:
    X_test = pd.read_csv('X_test.csv')
    y_test = pd.read_csv('y_test.csv').values.ravel()
    
    predictions, probabilities = predictor.predict(X_test)
    
    # 检查预测结果
    assert len(predictions) == len(X_test), "预测数量不匹配"
    assert len(probabilities) == len(X_test), "概率数量不匹配"
    assert all(p in [0, 1] for p in predictions), "预测值应为0或1"
    assert all(0 <= p <= 1 for p in probabilities), "概率应在0-1之间"
    
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, predictions)
    
    log_test("批量预测功能", True, f"准确率: {accuracy:.4f}")
    
except Exception as e:
    log_test("批量预测功能", False, str(e))

# ============================================================================
# 测试6: 测试单条预测
# ============================================================================
print("\n[测试6] 测试单条预测...")

try:
    X_single = X_test.iloc[[0]]
    pred, prob = predictor.predict_single(X_single)
    
    assert pred in [0, 1], "预测值应为0或1"
    assert 0 <= prob <= 1, "概率应在0-1之间"
    
    log_test("单条预测功能", True, f"预测: {pred}, 概率: {prob:.4f}")
    
except Exception as e:
    log_test("单条预测功能", False, str(e))

# ============================================================================
# 测试7: 测试日志系统
# ============================================================================
print("\n[测试7] 测试日志系统...")

log_files = [f for f in os.listdir('log') if f.endswith('.log')]
if log_files:
    log_test("日志文件生成", True, f"找到 {len(log_files)} 个日志文件")
else:
    log_test("日志文件生成", False, "未找到日志文件")

# ============================================================================
# 测试8: 测试模型性能
# ============================================================================
print("\n[测试8] 测试模型性能...")

try:
    from sklearn.metrics import classification_report, confusion_matrix
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    # 准确率检查
    if accuracy >= 0.85:
        log_test("模型准确率", True, f"{accuracy:.4f} >= 0.85")
    else:
        log_test("模型准确率", False, f"{accuracy:.4f} < 0.85")
    
    # 留任预测检查
    retention_recall = tn / (tn + fp)
    if retention_recall >= 0.95:
        log_test("留任预测召回率", True, f"{retention_recall:.4f} >= 0.95")
    else:
        log_test("留任预测召回率", False, f"{retention_recall:.4f} < 0.95")
    
except Exception as e:
    log_test("模型性能评估", False, str(e))

# ============================================================================
# 测试9: 测试集成信息
# ============================================================================
print("\n[测试9] 测试集成信息...")

try:
    ensemble_info = joblib.load('model/ensemble_info.pkl')
    
    assert 'weights' in ensemble_info, "缺少权重信息"
    assert 'threshold' in ensemble_info, "缺少阈值信息"
    assert 'accuracy' in ensemble_info, "缺少准确率信息"
    
    weights_sum = sum(ensemble_info['weights'])
    assert abs(weights_sum - 1.0) < 0.01, "权重之和应为1"
    
    log_test("集成信息验证", True, 
             f"权重: {ensemble_info['weights']}, 阈值: {ensemble_info['threshold']:.2f}")
    
except Exception as e:
    log_test("集成信息验证", False, str(e))

# ============================================================================
# 测试10: 测试文档完整性
# ============================================================================
print("\n[测试10] 测试文档完整性...")

doc_files = [
    'README.md',
    'QUICKSTART.md',
    'DELIVERY.md',
    'final_report.md'
]

for doc in doc_files:
    exists = os.path.exists(doc)
    if exists:
        size = os.path.getsize(doc)
        log_test(f"文档: {doc}", True, f"大小: {size/1024:.1f} KB")
    else:
        log_test(f"文档: {doc}", False, "文件不存在")

# ============================================================================
# 测试总结
# ============================================================================
print("\n" + "=" * 80)
print("测试总结")
print("=" * 80)

total_tests = len(test_results)
passed_tests = sum(1 for r in test_results if '✅' in r['status'])
failed_tests = total_tests - passed_tests

print(f"\n总测试数: {total_tests}")
print(f"通过: {passed_tests} ✅")
print(f"失败: {failed_tests} ❌")
print(f"通过率: {passed_tests/total_tests*100:.1f}%")

# 保存测试报告
report_file = f'log/test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
with open(report_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("自动化测试报告\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    for result in test_results:
        f.write(f"{result['status']} - {result['test']}\n")
        if result['message']:
            f.write(f"    {result['message']}\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write(f"总测试数: {total_tests}\n")
    f.write(f"通过: {passed_tests} ✅\n")
    f.write(f"失败: {failed_tests} ❌\n")
    f.write(f"通过率: {passed_tests/total_tests*100:.1f}%\n")

print(f"\n测试报告已保存: {report_file}")

# 如果有失败的测试，退出码为1
if failed_tests > 0:
    print("\n⚠️ 存在失败的测试，请检查！")
    sys.exit(1)
else:
    print("\n✅ 所有测试通过！")
    sys.exit(0)
