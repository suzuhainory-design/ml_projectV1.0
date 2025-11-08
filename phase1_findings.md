# 阶段1：数据探索与分析 - 关键发现

## 数据集概况

### 基本信息
- **训练集规模**: 1,100条记录，31个特征（包括目标变量）
- **测试集规模**: 350条记录，31个特征（包括目标变量）
- **任务类型**: 员工离职预测（Employee Attrition Prediction）
- **目标变量**: Attrition（0=留任，1=离职）

### 目标变量分布
- **训练集**: 留任(0): 922条 (83.8%), 离职(1): 178条 (16.2%)
- **测试集**: 留任(0): 297条 (84.9%), 离职(1): 53条 (15.1%)
- **类别不平衡**: 存在明显的类别不平衡问题，离职样本占比约16%

## 特征分析

### 数值型特征 (22个)
1. Age - 年龄
2. DistanceFromHome - 离家距离
3. Education - 教育水平
4. EmployeeNumber - 员工编号
5. EnvironmentSatisfaction - 环境满意度
6. JobInvolvement - 工作投入度
7. JobLevel - 职位级别
8. JobSatisfaction - 工作满意度
9. MonthlyIncome - 月收入
10. NumCompaniesWorked - 曾工作公司数
11. PercentSalaryHike - 薪资涨幅百分比
12. PerformanceRating - 绩效评级
13. RelationshipSatisfaction - 关系满意度
14. StandardHours - 标准工时（常量，所有值为80）
15. StockOptionLevel - 股票期权级别
16. TotalWorkingYears - 总工作年限
17. TrainingTimesLastYear - 去年培训次数
18. WorkLifeBalance - 工作生活平衡
19. YearsAtCompany - 在公司年限
20. YearsInCurrentRole - 当前角色年限
21. YearsSinceLastPromotion - 距上次晋升年限
22. YearsWithCurrManager - 与当前经理共事年限

### 类别型特征 (8个)
1. BusinessTravel - 出差频率（3类: Travel_Rarely, Travel_Frequently, Non-Travel）
2. Department - 部门（3类: Research & Development, Sales, Human Resources）
3. EducationField - 教育领域（6类）
4. Gender - 性别（2类: Male, Female）
5. JobRole - 职位角色（9类）
6. MaritalStatus - 婚姻状况（3类: Married, Single, Divorced）
7. Over18 - 是否成年（1类: Y，常量特征）
8. OverTime - 是否加班（2类: Yes, No）

### 常量特征识别
- **StandardHours**: 所有值均为80，无区分度
- **Over18**: 所有值均为Y，无区分度
- 这两个特征在建模时应该移除

## 特征与离职的相关性分析

### 负相关（工作年限越长/收入越高，离职率越低）
1. **TotalWorkingYears** (-0.188): 总工作年限越长，离职率越低
2. **Age** (-0.175): 年龄越大，离职率越低
3. **JobLevel** (-0.169): 职位级别越高，离职率越低
4. **YearsInCurrentRole** (-0.163): 当前角色年限越长，离职率越低
5. **YearsWithCurrManager** (-0.159): 与当前经理共事越久，离职率越低
6. **MonthlyIncome** (-0.156): 月收入越高，离职率越低
7. **YearsAtCompany** (-0.144): 在公司年限越长，离职率越低
8. **StockOptionLevel** (-0.138): 股票期权级别越高，离职率越低

### 负相关（满意度越高，离职率越低）
9. **JobSatisfaction** (-0.126): 工作满意度越高，离职率越低
10. **JobInvolvement** (-0.123): 工作投入度越高，离职率越低
11. **EnvironmentSatisfaction** (-0.097): 环境满意度越高，离职率越低

### 正相关（但相关性较弱）
1. **DistanceFromHome** (0.089): 离家距离越远，离职率略高
2. **PerformanceRating** (0.047): 绩效评级与离职率正相关（较弱）
3. **NumCompaniesWorked** (0.026): 曾工作公司数与离职率正相关（很弱）

## 类别特征与离职的关系

### 加班情况（OverTime）
- **加班员工离职率明显更高**（约30%），不加班员工离职率较低（约10%）
- 这是一个非常重要的预测特征

### 部门（Department）
- 不同部门的离职率有差异
- Sales和Research & Development部门离职率相对较高

### 婚姻状况（MaritalStatus）
- Single（单身）员工的离职率可能较高

## 数据质量

### 优点
- **无缺失值**: 训练集和测试集均无缺失值
- **数据完整**: 所有特征都有完整的数据

### 需要注意的问题
1. **类别不平衡**: 离职样本仅占16%，需要使用适当的采样或加权策略
2. **常量特征**: StandardHours和Over18需要移除
3. **EmployeeNumber**: 员工编号可能是唯一标识符，不应作为预测特征

## 下一步计划

### 外部数据增强策略
1. **行业离职率数据**: 获取不同行业、职位的平均离职率数据
2. **薪资基准数据**: 获取行业薪资标准，计算相对薪资水平
3. **地理位置数据**: 如果有城市信息，可以引入城市生活成本数据
4. **经济指标数据**: 引入宏观经济指标（失业率、GDP增长率等）

### 特征工程策略
1. 创建交互特征（如：收入与年龄比、工作年限与职位级别比）
2. 创建聚合特征（如：各类满意度的平均值）
3. 对类别特征进行编码（One-Hot、Label Encoding、Target Encoding）
4. 特征缩放和标准化

### 模型策略
1. 处理类别不平衡（SMOTE、权重调整）
2. 尝试多种模型（随机森林、XGBoost、LightGBM、神经网络）
3. 集成学习（Stacking、Voting）
4. 超参数优化（GridSearch、RandomSearch、Optuna）
