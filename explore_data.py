import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
train_df = pd.read_csv('/home/ubuntu/upload/train.csv', header=None)
test_df = pd.read_csv('/home/ubuntu/upload/test.csv', header=None)

print("=" * 80)
print("训练集基本信息")
print("=" * 80)
print(f"训练集形状: {train_df.shape}")
print(f"测试集形状: {test_df.shape}")
print()

# 检查第一行是否为列名
print("训练集前5行:")
print(train_df.head())
print()

print("测试集前5行:")
print(test_df.head())
print()

# 数据类型分析
print("=" * 80)
print("数据类型分析")
print("=" * 80)
print(train_df.dtypes)
print()

# 缺失值检查
print("=" * 80)
print("缺失值检查")
print("=" * 80)
print("训练集缺失值:")
print(train_df.isnull().sum())
print()
print("测试集缺失值:")
print(test_df.isnull().sum())
print()

# 目标变量分析（假设第一列是目标变量）
print("=" * 80)
print("目标变量分析")
print("=" * 80)
if 0 in train_df.columns:
    print("训练集目标变量分布:")
    print(train_df[0].value_counts())
    print()
    print("训练集目标变量比例:")
    print(train_df[0].value_counts(normalize=True))
    print()

# 检查测试集是否有目标变量
if 0 in test_df.columns:
    print("测试集最后一列分布:")
    print(test_df.iloc[:, -1].value_counts())
    print()

# 数值型特征统计
print("=" * 80)
print("数值型特征统计")
print("=" * 80)
print(train_df.describe())
print()

# 保存基本信息到文件
with open('/home/ubuntu/ml_project/data_summary.txt', 'w', encoding='utf-8') as f:
    f.write(f"训练集形状: {train_df.shape}\n")
    f.write(f"测试集形状: {test_df.shape}\n\n")
    f.write("训练集列数据类型:\n")
    f.write(str(train_df.dtypes) + "\n\n")
    f.write("训练集缺失值:\n")
    f.write(str(train_df.isnull().sum()) + "\n\n")
    if 0 in train_df.columns:
        f.write("目标变量分布:\n")
        f.write(str(train_df[0].value_counts()) + "\n\n")

print("数据探索完成！")
