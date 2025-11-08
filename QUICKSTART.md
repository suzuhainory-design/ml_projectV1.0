# 快速使用指南

## 项目概述

本项目是一个员工离职预测系统，使用集成学习方法达到**89.71%**的测试集准确率。

## 核心文件说明

### 📁 model/ - 模型文件夹
存放训练好的模型文件，包括：
- `catboost_model.pkl` - CatBoost模型（8.0MB）
- `xgboost_model.pkl` - XGBoost模型（1.5MB）
- `lightgbm_model.pkl` - LightGBM模型（2.5MB）
- `ensemble_info.pkl` - 集成模型配置（权重、阈值）
- `smote_transformer.pkl` - 数据平衡转换器

### 📁 log/ - 日志文件夹
存放所有运行日志和结果：
- `model_integration_*.log` - 模型训练日志
- `prediction_*.log` - 预测运行日志
- `evaluation_results_*.json` - 评估结果（JSON格式）
- `predictions_*.csv` - 预测结果（CSV格式）

### 🐍 Python脚本
- `integrate_best_models.py` - 模型训练与整合脚本
- `predict.py` - 预测脚本

### 📄 文档
- `README.md` - 完整项目文档
- `final_report.md` - 详细技术报告
- `QUICKSTART.md` - 本文档

## 快速开始

### 1️⃣ 环境要求

```bash
Python 3.11+
```

**依赖库**：
```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost imbalanced-learn joblib
```

### 2️⃣ 重新训练模型（可选）

如果需要重新训练模型：

```bash
cd ml_project
python3.11 integrate_best_models.py
```

**输出**：
- 模型保存在 `model/` 文件夹
- 训练日志保存在 `log/` 文件夹

### 3️⃣ 使用模型进行预测

对测试集进行预测：

```bash
python3.11 predict.py
```

**输出**：
- 控制台显示预测结果
- 日志保存在 `log/prediction_*.log`
- 预测结果保存在 `log/predictions_*.csv`

### 4️⃣ 在代码中使用模型

```python
from predict import EnsemblePredictor
import pandas as pd

# 1. 初始化预测器
predictor = EnsemblePredictor(model_dir='model')

# 2. 加载数据（需要包含51个特征）
X = pd.read_csv('your_data.csv')

# 3. 批量预测
predictions, probabilities = predictor.predict(X)

# 4. 预测单条记录
single_pred, single_prob = predictor.predict_single(X.iloc[[0]])

# 5. 解读结果
if single_pred == 1:
    print(f"预测：员工可能离职，概率：{single_prob:.2%}")
else:
    print(f"预测：员工可能留任，概率：{1-single_prob:.2%}")
```

## 输入数据格式

### 必需的51个特征

模型需要以下特征（按顺序）：

#### 原始特征（30个）
1. Age - 年龄
2. BusinessTravel - 出差频率（编码后）
3. Department - 部门（编码后）
4. DistanceFromHome - 距离家的距离
5. Education - 教育程度
6. EducationField - 教育领域（编码后）
7. EnvironmentSatisfaction - 环境满意度
8. Gender - 性别（编码后）
9. JobInvolvement - 工作投入度
10. JobLevel - 职位级别
11. JobRole - 职位角色（编码后）
12. JobSatisfaction - 工作满意度
13. MaritalStatus - 婚姻状况（编码后）
14. MonthlyIncome - 月收入
15. NumCompaniesWorked - 工作过的公司数量
16. OverTime - 是否加班（编码后）
17. PercentSalaryHike - 薪资增长百分比
18. PerformanceRating - 绩效评级
19. RelationshipSatisfaction - 关系满意度
20. StockOptionLevel - 股票期权级别
21. TotalWorkingYears - 总工作年限
22. TrainingTimesLastYear - 去年培训次数
23. WorkLifeBalance - 工作生活平衡
24. YearsAtCompany - 在公司年限
25. YearsInCurrentRole - 当前角色年限
26. YearsSinceLastPromotion - 上次晋升以来的年限
27. YearsWithCurrManager - 与当前经理共事年限

#### 外部数据特征（6个）
28. DepartmentTurnoverBenchmark - 部门离职率基准
29. JobRoleTurnoverBenchmark - 职位离职率基准
30. OvertimeImpact - 加班影响系数
31. TravelImpact - 出差影响系数
32. MaritalImpact - 婚姻状况影响系数
33. ExternalRiskScore - 外部风险评分

#### 衍生特征（18个）
34. AgeGroup - 年龄组
35. IncomePerYear - 年收入
36. IncomeToAgeRatio - 收入年龄比
37. YearsAtCompanyRatio - 公司年限比率
38. YearsInRoleRatio - 角色年限比率
39. YearsWithManagerRatio - 经理共事年限比率
40. YearsSincePromotionRatio - 晋升年限比率
41. TotalSatisfaction - 总满意度
42. SatisfactionScore - 满意度评分
43. PerformanceScore - 绩效评分
44. CareerProgressionRate - 职业发展速度
45. PromotionFrequency - 晋升频率
46. JobStability - 工作稳定性
47. IsNewEmployee - 是否新员工
48. IsSeniorEmployee - 是否资深员工
49. LongTimeNoPromotion - 是否长期未晋升
50. IncomeLevelMatch - 收入级别匹配度
51. DistanceOvertimeInteraction - 距离加班交互

## 输出说明

### 预测结果

**predictions** (numpy array):
- `0` = 预测员工留任
- `1` = 预测员工离职

**probabilities** (numpy array):
- 范围: 0.0 到 1.0
- 表示员工离职的概率
- 阈值: 0.46（大于此值预测为离职）

### 日志文件

所有日志文件都保存在 `log/` 文件夹，包含：
- 时间戳
- 运行状态
- 模型性能指标
- 错误信息（如有）

## 性能指标

### 测试集表现

| 指标 | 值 |
|------|-----|
| **准确率** | **89.71%** |
| 留任预测准确率 | 91% |
| 离职预测准确率 | 79% |
| 留任召回率 | 98% |
| 离职召回率 | 43% |

### 混淆矩阵

```
实际 \ 预测    留任    离职
留任           291      6
离职            30     23
```

**解读**：
- ✅ 291个留任员工被正确预测
- ✅ 23个离职员工被正确预测
- ❌ 6个留任员工被错误预测为离职
- ❌ 30个离职员工被错误预测为留任

## 常见问题

### Q1: 为什么离职预测召回率较低？

**A**: 这是由于训练数据中离职样本较少（仅16%）导致的。虽然使用了SMOTE等技术，但模型仍然倾向于预测多数类（留任）。在实际应用中，可以通过调整阈值来提高离职预测的召回率，但会降低整体准确率。

### Q2: 如何调整预测阈值？

**A**: 修改 `model/ensemble_info.pkl` 中的阈值，或在代码中手动设置：

```python
# 降低阈值会提高离职预测的召回率
custom_threshold = 0.3  # 默认是0.46
predictions = (probabilities > custom_threshold).astype(int)
```

### Q3: 模型可以用于其他数据集吗？

**A**: 可以，但需要确保输入数据包含相同的51个特征，且特征的含义和编码方式一致。

### Q4: 如何查看详细的日志？

**A**: 所有日志文件都保存在 `log/` 文件夹中，使用文本编辑器打开即可查看。

## 技术支持

- 详细文档：查看 `README.md`
- 技术报告：查看 `final_report.md`
- 代码注释：所有Python脚本都包含详细注释

## 许可证

本项目仅供学习和研究使用。

---

**最后更新**: 2025年11月7日  
**版本**: 1.0  
**准确率**: 89.71%
