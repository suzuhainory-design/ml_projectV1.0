import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("数据预处理与特征工程")
print("=" * 80)

# 读取增强后的数据
train_df = pd.read_csv('/home/ubuntu/ml_project/train_enhanced.csv')
test_df = pd.read_csv('/home/ubuntu/ml_project/test_enhanced.csv')

print(f"\n原始数据形状:")
print(f"训练集: {train_df.shape}")
print(f"测试集: {test_df.shape}")

# 分离特征和目标变量
X_full = train_df.drop('Attrition', axis=1)
y_full = train_df['Attrition']
X_test_final = test_df.drop('Attrition', axis=1)
y_test_final = test_df['Attrition']

print(f"\n目标变量分布:")
print(y_full.value_counts())
print(y_full.value_counts(normalize=True))

# ============================================================================
# 1. 移除无用特征
# ============================================================================
print("\n" + "=" * 80)
print("1. 移除无用特征")
print("=" * 80)

# 移除常量特征和员工编号
useless_features = ['StandardHours', 'Over18', 'EmployeeNumber']
X_full = X_full.drop(useless_features, axis=1)
X_test_final = X_test_final.drop(useless_features, axis=1)

print(f"移除的特征: {useless_features}")
print(f"移除后特征数: {X_full.shape[1]}")

# ============================================================================
# 2. 特征工程 - 创建新特征
# ============================================================================
print("\n" + "=" * 80)
print("2. 特征工程 - 创建新特征")
print("=" * 80)

def create_features(df):
    """创建新特征"""
    df = df.copy()
    
    # 2.1 年龄相关特征
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 100], 
                            labels=[0, 1, 2, 3])
    
    # 2.2 收入相关特征
    df['IncomePerYear'] = df['MonthlyIncome'] * 12
    df['IncomeToAgeRatio'] = df['MonthlyIncome'] / (df['Age'] + 1)
    
    # 2.3 工作年限相关特征
    df['YearsAtCompanyRatio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)
    df['YearsInRoleRatio'] = df['YearsInCurrentRole'] / (df['YearsAtCompany'] + 1)
    df['YearsWithManagerRatio'] = df['YearsWithCurrManager'] / (df['YearsAtCompany'] + 1)
    df['YearsSincePromotionRatio'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)
    
    # 2.4 满意度相关特征
    df['TotalSatisfaction'] = (df['EnvironmentSatisfaction'] + 
                               df['JobSatisfaction'] + 
                               df['RelationshipSatisfaction']) / 3
    
    df['SatisfactionScore'] = (df['EnvironmentSatisfaction'] + 
                               df['JobSatisfaction'] + 
                               df['RelationshipSatisfaction'] +
                               df['WorkLifeBalance'])
    
    # 2.5 工作投入度和绩效
    df['PerformanceScore'] = df['JobInvolvement'] * df['PerformanceRating']
    
    # 2.6 职业发展指标
    df['CareerProgressionRate'] = df['JobLevel'] / (df['TotalWorkingYears'] + 1)
    df['PromotionFrequency'] = df['YearsAtCompany'] / (df['YearsSinceLastPromotion'] + 1)
    
    # 2.7 稳定性指标
    df['JobStability'] = df['YearsAtCompany'] / (df['NumCompaniesWorked'] + 1)
    
    # 2.8 是否新员工
    df['IsNewEmployee'] = (df['YearsAtCompany'] <= 2).astype(int)
    
    # 2.9 是否资深员工
    df['IsSeniorEmployee'] = (df['YearsAtCompany'] >= 10).astype(int)
    
    # 2.10 是否长期未晋升
    df['LongTimeNoPromotion'] = (df['YearsSinceLastPromotion'] >= 5).astype(int)
    
    # 2.11 收入与职位级别的匹配度
    df['IncomeLevelMatch'] = df['MonthlyIncome'] / (df['JobLevel'] * 1000 + 1)
    
    # 2.12 距离与加班的交互
    df['DistanceOvertimeInteraction'] = df['DistanceFromHome'] * (df['OverTime'] == 'Yes').astype(int)
    
    return df

print("创建新特征...")
X_full = create_features(X_full)
X_test_final = create_features(X_test_final)

print(f"特征工程后特征数: {X_full.shape[1]}")

# ============================================================================
# 3. 类别特征编码
# ============================================================================
print("\n" + "=" * 80)
print("3. 类别特征编码")
print("=" * 80)

# 识别类别特征
categorical_features = X_full.select_dtypes(include=['object']).columns.tolist()
print(f"类别特征: {categorical_features}")

# 对类别特征进行编码
label_encoders = {}

for col in categorical_features:
    le = LabelEncoder()
    # 合并训练集和测试集的类别，确保编码一致
    all_categories = pd.concat([X_full[col], X_test_final[col]]).unique()
    le.fit(all_categories)
    
    X_full[col] = le.transform(X_full[col])
    X_test_final[col] = le.transform(X_test_final[col])
    
    label_encoders[col] = le

print(f"已编码 {len(categorical_features)} 个类别特征")

# ============================================================================
# 4. 划分训练集和验证集
# ============================================================================
print("\n" + "=" * 80)
print("4. 划分训练集和验证集")
print("=" * 80)

# 使用stratify确保训练集和验证集的类别分布一致
X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_full
)

print(f"训练集大小: {X_train.shape}")
print(f"验证集大小: {X_val.shape}")
print(f"测试集大小: {X_test_final.shape}")

print(f"\n训练集目标分布:")
print(y_train.value_counts())
print(y_train.value_counts(normalize=True))

print(f"\n验证集目标分布:")
print(y_val.value_counts())
print(y_val.value_counts(normalize=True))

# ============================================================================
# 5. 特征缩放
# ============================================================================
print("\n" + "=" * 80)
print("5. 特征缩放")
print("=" * 80)

# 识别数值特征（排除已经是0-1范围的二值特征）
numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()

# 对数值特征进行标准化
scaler = StandardScaler()
scaler.fit(X_train[numeric_features])

X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_test_scaled = X_test_final.copy()

X_train_scaled[numeric_features] = scaler.transform(X_train[numeric_features])
X_val_scaled[numeric_features] = scaler.transform(X_val[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test_final[numeric_features])

print(f"已缩放 {len(numeric_features)} 个数值特征")

# ============================================================================
# 6. 保存处理后的数据
# ============================================================================
print("\n" + "=" * 80)
print("6. 保存处理后的数据")
print("=" * 80)

# 保存未缩放的数据（用于树模型）
X_train.to_csv('/home/ubuntu/ml_project/X_train.csv', index=False)
X_val.to_csv('/home/ubuntu/ml_project/X_val.csv', index=False)
X_test_final.to_csv('/home/ubuntu/ml_project/X_test.csv', index=False)
y_train.to_csv('/home/ubuntu/ml_project/y_train.csv', index=False)
y_val.to_csv('/home/ubuntu/ml_project/y_val.csv', index=False)
y_test_final.to_csv('/home/ubuntu/ml_project/y_test.csv', index=False)

# 保存缩放后的数据（用于线性模型和神经网络）
X_train_scaled.to_csv('/home/ubuntu/ml_project/X_train_scaled.csv', index=False)
X_val_scaled.to_csv('/home/ubuntu/ml_project/X_val_scaled.csv', index=False)
X_test_scaled.to_csv('/home/ubuntu/ml_project/X_test_scaled.csv', index=False)

print("已保存以下文件:")
print("- X_train.csv, X_val.csv, X_test.csv (未缩放)")
print("- X_train_scaled.csv, X_val_scaled.csv, X_test_scaled.csv (已缩放)")
print("- y_train.csv, y_val.csv, y_test.csv")

# ============================================================================
# 7. 特征重要性分析（使用随机森林）
# ============================================================================
print("\n" + "=" * 80)
print("7. 特征重要性初步分析")
print("=" * 80)

from sklearn.ensemble import RandomForestClassifier

# 训练一个简单的随机森林来查看特征重要性
rf_temp = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_temp.fit(X_train, y_train)

# 获取特征重要性
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_temp.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 最重要特征:")
print(feature_importance.head(20))

# 保存特征重要性
feature_importance.to_csv('/home/ubuntu/ml_project/feature_importance.csv', index=False)

print("\n" + "=" * 80)
print("数据预处理与特征工程完成！")
print("=" * 80)
print(f"\n最终特征数: {X_train.shape[1]}")
print(f"训练样本数: {X_train.shape[0]}")
print(f"验证样本数: {X_val.shape[0]}")
print(f"测试样本数: {X_test_final.shape[0]}")
