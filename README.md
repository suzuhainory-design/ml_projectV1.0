# 员工离职预测系统

基于机器学习的员工离职预测系统，使用集成学习方法达到89.71%的测试集准确率。

## 项目结构

```
ml_project/
├── model/                          # 模型文件夹
│   ├── catboost_model.pkl         # CatBoost模型
│   ├── xgboost_model.pkl          # XGBoost模型
│   ├── lightgbm_model.pkl         # LightGBM模型
│   ├── ensemble_info.pkl          # 集成模型信息（权重、阈值）
│   └── smote_transformer.pkl      # SMOTE转换器
│
├── log/                            # 日志文件夹
│   ├── model_integration_*.log    # 模型整合日志
│   ├── prediction_*.log           # 预测日志
│   ├── evaluation_results_*.json  # 评估结果JSON
│   └── predictions_*.csv          # 预测结果CSV
│
├── integrate_best_models.py       # 模型整合脚本
├── predict.py                     # 预测脚本
├── README.md                      # 项目说明文档
│
├── X_train.csv                    # 训练集特征
├── X_val.csv                      # 验证集特征
├── X_test.csv                     # 测试集特征
├── y_train.csv                    # 训练集标签
├── y_val.csv                      # 验证集标签
├── y_test.csv                     # 测试集标签
│
└── final_report.md                # 完整项目报告
```

## 模型说明

### 集成模型组成

本系统使用3个优秀的梯度提升模型进行集成：

1. **CatBoost** (权重: 33.37%)
   - 迭代次数: 1000
   - 深度: 9
   - 学习率: 0.03
   - 自动类别权重平衡

2. **XGBoost** (权重: 33.26%)
   - 迭代次数: 800
   - 最大深度: 7
   - 学习率: 0.02
   - 正则化参数优化

3. **LightGBM** (权重: 33.37%)
   - 迭代次数: 800
   - 最大深度: 7
   - 学习率: 0.02
   - 叶子节点优化

### 模型性能

| 模型 | 测试集准确率 |
|------|-------------|
| CatBoost | 88.86% |
| XGBoost | 88.57% |
| LightGBM | 88.86% |
| **集成模型** | **89.71%** |

### 详细性能指标

**混淆矩阵**:
```
                预测留任    预测离职
实际留任          291         6
实际离职           30        23
```

**分类报告**:
- 留任预测准确率: 91% (Precision), 98% (Recall)
- 离职预测准确率: 79% (Precision), 43% (Recall)
- 总体准确率: 89.71%

## 使用方法

### 1. 模型训练与整合

运行模型整合脚本，训练并保存最佳模型：

```bash
python3.11 integrate_best_models.py
```

**输出**:
- 模型文件保存在 `model/` 文件夹
- 训练日志保存在 `log/` 文件夹
- 评估结果保存为JSON格式

### 2. 使用模型进行预测

运行预测脚本，对测试数据进行预测：

```bash
python3.11 predict.py
```

**输出**:
- 预测日志保存在 `log/` 文件夹
- 预测结果保存为CSV文件，包含实际值、预测值和概率

### 3. 自定义预测

在Python代码中使用模型：

```python
from predict import EnsemblePredictor
import pandas as pd

# 初始化预测器
predictor = EnsemblePredictor(model_dir='model')

# 加载数据
X = pd.read_csv('your_data.csv')

# 进行预测
predictions, probabilities = predictor.predict(X)

# 预测单条记录
single_prediction, single_probability = predictor.predict_single(X.iloc[[0]])

print(f"预测结果: {'离职' if single_prediction == 1 else '留任'}")
print(f"离职概率: {single_probability:.2%}")
```

## 特征说明

模型使用51个特征，包括：

### 原始特征（30个）
- **人口统计**: Age, Gender, MaritalStatus
- **工作信息**: Department, JobRole, JobLevel, BusinessTravel
- **薪酬**: MonthlyIncome, PercentSalaryHike, StockOptionLevel
- **工作年限**: TotalWorkingYears, YearsAtCompany, YearsInCurrentRole等
- **满意度**: JobSatisfaction, EnvironmentSatisfaction, RelationshipSatisfaction
- **其他**: OverTime, DistanceFromHome, Education等

### 外部数据特征（6个）
- **DepartmentTurnoverBenchmark**: 部门离职率基准
- **JobRoleTurnoverBenchmark**: 职位角色离职率基准
- **OvertimeImpact**: 加班影响系数
- **TravelImpact**: 出差频率影响系数
- **MaritalImpact**: 婚姻状况影响系数
- **ExternalRiskScore**: 综合离职风险评分

### 衍生特征（18个）
- **收入相关**: IncomePerYear, IncomeToAgeRatio, IncomeLevelMatch
- **工作年限比率**: YearsAtCompanyRatio, YearsInRoleRatio等
- **满意度综合**: TotalSatisfaction, SatisfactionScore
- **职业发展**: CareerProgressionRate, PromotionFrequency
- **稳定性**: JobStability
- **二值特征**: IsNewEmployee, IsSeniorEmployee, LongTimeNoPromotion

## 数据预处理

### 类别不平衡处理

使用SMOTE（合成少数类过采样技术）处理类别不平衡问题：
- 原始训练集: 留任 83.8%, 离职 16.2%
- SMOTE后: 留任 50%, 离职 50%

### 特征编码

- **类别特征**: 使用Label Encoding
- **数值特征**: 保持原始值（树模型对特征缩放不敏感）

## 日志系统

所有操作都会记录详细日志，包括：

1. **模型整合日志** (`model_integration_*.log`)
   - 数据加载信息
   - SMOTE处理过程
   - 模型训练进度
   - 模型保存路径
   - 评估结果

2. **预测日志** (`prediction_*.log`)
   - 模型加载状态
   - 预测数据信息
   - 预测结果统计
   - 性能评估指标

3. **评估结果** (`evaluation_results_*.json`)
   - 各模型准确率
   - 集成权重和阈值
   - 混淆矩阵
   - 分类报告

## 技术栈

- **Python**: 3.11
- **机器学习框架**:
  - scikit-learn: 基础机器学习工具
  - XGBoost: 梯度提升框架
  - LightGBM: 微软梯度提升框架
  - CatBoost: Yandex梯度提升框架
- **数据处理**:
  - pandas: 数据处理
  - numpy: 数值计算
  - imbalanced-learn: 类别不平衡处理
- **其他**:
  - joblib: 模型序列化
  - logging: 日志系统

## 性能优化策略

1. **数据增强**: 引入行业离职率基准等外部数据
2. **特征工程**: 创建18个衍生特征
3. **类别平衡**: 使用SMOTE处理不平衡数据
4. **超参数优化**: 使用Optuna进行贝叶斯优化
5. **集成学习**: 加权软投票集成3个模型
6. **阈值优化**: 优化分类阈值提升性能

## 模型局限性

1. **离职召回率较低** (43%): 由于类别不平衡，模型对离职员工的识别能力有限
2. **数据规模**: 训练集仅1,100条，限制了模型的泛化能力
3. **特征信息**: 员工离职受多种因素影响，现有特征可能无法完全捕捉

## 改进建议

1. 收集更多训练数据，特别是离职样本
2. 引入更多特征（绩效评估、团队氛围等）
3. 使用成本敏感学习优化离职预测召回率
4. 尝试深度学习方法
5. 加入时序分析（如果有历史数据）

## 许可证

本项目仅供学习和研究使用。

## 联系方式

如有问题或建议，请查看 `final_report.md` 获取更多详细信息。

---

**最后更新**: 2025年11月7日  
**版本**: 1.0  
**测试集准确率**: 89.71%
