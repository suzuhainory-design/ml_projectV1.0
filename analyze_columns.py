import pandas as pd
import numpy as np

# 读取数据
train_df = pd.read_csv('/home/ubuntu/upload/train.csv', header=None)
test_df = pd.read_csv('/home/ubuntu/upload/test.csv', header=None)

print("=" * 80)
print("识别列名")
print("=" * 80)

# 训练集第一行是列名
train_header = train_df.iloc[0].tolist()
print("训练集列名（第一行）:")
print(train_header)
print()

# 测试集第一行也是列名
test_header = test_df.iloc[0].tolist()
print("测试集列名（第一行）:")
print(test_header)
print()

# 重新读取数据，使用第一行作为列名
train_df = pd.read_csv('/home/ubuntu/upload/train.csv', header=0)
test_df = pd.read_csv('/home/ubuntu/upload/test.csv', header=0)

print("=" * 80)
print("重新加载后的数据信息")
print("=" * 80)
print(f"训练集形状: {train_df.shape}")
print(f"测试集形状: {test_df.shape}")
print()

print("训练集列名:")
print(train_df.columns.tolist())
print()

print("测试集列名:")
print(test_df.columns.tolist())
print()

# 检查目标变量
print("=" * 80)
print("目标变量分析")
print("=" * 80)

# 训练集的目标变量应该是Attrition
if 'Attrition' in train_df.columns:
    print("训练集目标变量(Attrition)分布:")
    print(train_df['Attrition'].value_counts())
    print()
    print("训练集目标变量比例:")
    print(train_df['Attrition'].value_counts(normalize=True))
    print()

# 测试集的目标变量
if 'Attrition' in test_df.columns:
    print("测试集目标变量(Attrition)分布:")
    print(test_df['Attrition'].value_counts())
    print()

# 分析特征类型
print("=" * 80)
print("特征类型分析")
print("=" * 80)

# 识别数值型和类别型特征
numeric_features = []
categorical_features = []

for col in train_df.columns:
    if col == 'Attrition':
        continue
    
    try:
        # 尝试转换为数值型
        pd.to_numeric(train_df[col])
        numeric_features.append(col)
    except:
        categorical_features.append(col)

print(f"数值型特征 ({len(numeric_features)}):")
print(numeric_features)
print()

print(f"类别型特征 ({len(categorical_features)}):")
print(categorical_features)
print()

# 数值型特征统计
print("=" * 80)
print("数值型特征统计")
print("=" * 80)
if numeric_features:
    print(train_df[numeric_features].describe())
print()

# 类别型特征统计
print("=" * 80)
print("类别型特征唯一值数量")
print("=" * 80)
for col in categorical_features:
    print(f"{col}: {train_df[col].nunique()} 个唯一值")
    print(f"  值: {train_df[col].unique()[:10]}")  # 显示前10个
    print()

# 保存特征信息
feature_info = {
    'numeric_features': numeric_features,
    'categorical_features': categorical_features,
    'train_shape': train_df.shape,
    'test_shape': test_df.shape
}

import json
with open('/home/ubuntu/ml_project/feature_info.json', 'w') as f:
    json.dump(feature_info, f, indent=2)

print("特征信息已保存到 feature_info.json")
