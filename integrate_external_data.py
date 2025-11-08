import pandas as pd
import numpy as np

# 读取原始数据
train_df = pd.read_csv('/home/ubuntu/upload/train.csv')
test_df = pd.read_csv('/home/ubuntu/upload/test.csv')

print("=" * 80)
print("整合外部数据 - 基于行业研究的离职率基准")
print("=" * 80)

# 基于外部研究的部门离职率基准（2022年数据）
department_turnover_benchmark = {
    'Research & Development': 0.12,  # 研发类职位平均约12%
    'Sales': 0.125,                   # 销售职位约12.5%
    'Human Resources': 0.073          # 人力资源约7.3%
}

# 基于外部研究的职位角色离职率基准
job_role_turnover_benchmark = {
    'Sales Executive': 0.125,
    'Sales Representative': 0.125,
    'Research Scientist': 0.12,
    'Laboratory Technician': 0.12,
    'Manufacturing Director': 0.10,
    'Research Director': 0.10,
    'Healthcare Representative': 0.11,
    'Human Resources': 0.073,
    'Manager': 0.09
}

# 加班对离职率的影响系数（基于研究发现，加班员工离职率约为不加班的3倍）
overtime_impact = {
    'Yes': 1.3,  # 加班增加30%的离职风险
    'No': 0.7    # 不加班降低30%的离职风险
}

# 出差频率对离职率的影响
travel_impact = {
    'Travel_Frequently': 1.2,  # 频繁出差增加20%风险
    'Travel_Rarely': 1.0,      # 偶尔出差无影响
    'Non-Travel': 0.9          # 不出差降低10%风险
}

# 婚姻状况对离职率的影响（单身员工离职率更高）
marital_impact = {
    'Single': 1.2,
    'Married': 0.9,
    'Divorced': 1.0
}

# 为训练集添加外部数据特征
print("\n为训练集添加外部数据特征...")

# 1. 部门离职率基准
train_df['DepartmentTurnoverBenchmark'] = train_df['Department'].map(department_turnover_benchmark)

# 2. 职位角色离职率基准
train_df['JobRoleTurnoverBenchmark'] = train_df['JobRole'].map(job_role_turnover_benchmark)

# 3. 加班影响系数
train_df['OvertimeImpact'] = train_df['OverTime'].map(overtime_impact)

# 4. 出差影响系数
train_df['TravelImpact'] = train_df['BusinessTravel'].map(travel_impact)

# 5. 婚姻状况影响系数
train_df['MaritalImpact'] = train_df['MaritalStatus'].map(marital_impact)

# 6. 综合离职风险评分（基于外部数据）
train_df['ExternalRiskScore'] = (
    train_df['DepartmentTurnoverBenchmark'] * 
    train_df['OvertimeImpact'] * 
    train_df['TravelImpact'] * 
    train_df['MaritalImpact']
)

# 为测试集添加相同的外部数据特征
print("为测试集添加外部数据特征...")

test_df['DepartmentTurnoverBenchmark'] = test_df['Department'].map(department_turnover_benchmark)
test_df['JobRoleTurnoverBenchmark'] = test_df['JobRole'].map(job_role_turnover_benchmark)
test_df['OvertimeImpact'] = test_df['OverTime'].map(overtime_impact)
test_df['TravelImpact'] = test_df['BusinessTravel'].map(travel_impact)
test_df['MaritalImpact'] = test_df['MaritalStatus'].map(marital_impact)
test_df['ExternalRiskScore'] = (
    test_df['DepartmentTurnoverBenchmark'] * 
    test_df['OvertimeImpact'] * 
    test_df['TravelImpact'] * 
    test_df['MaritalImpact']
)

# 保存增强后的数据集
train_df.to_csv('/home/ubuntu/ml_project/train_enhanced.csv', index=False)
test_df.to_csv('/home/ubuntu/ml_project/test_enhanced.csv', index=False)

print("\n增强后的数据集已保存:")
print(f"训练集: train_enhanced.csv - 形状: {train_df.shape}")
print(f"测试集: test_enhanced.csv - 形状: {test_df.shape}")

print("\n新增特征:")
print("1. DepartmentTurnoverBenchmark - 部门离职率基准")
print("2. JobRoleTurnoverBenchmark - 职位角色离职率基准")
print("3. OvertimeImpact - 加班影响系数")
print("4. TravelImpact - 出差影响系数")
print("5. MaritalImpact - 婚姻状况影响系数")
print("6. ExternalRiskScore - 综合离职风险评分")

# 分析外部风险评分与实际离职的关系
print("\n" + "=" * 80)
print("外部风险评分与实际离职的关系分析")
print("=" * 80)

# 按离职状态分组统计外部风险评分
risk_by_attrition = train_df.groupby('Attrition')['ExternalRiskScore'].describe()
print("\n外部风险评分统计（按离职状态）:")
print(risk_by_attrition)

# 计算相关性
correlation = train_df['ExternalRiskScore'].corr(train_df['Attrition'])
print(f"\n外部风险评分与离职的相关系数: {correlation:.4f}")

print("\n外部数据整合完成！")
