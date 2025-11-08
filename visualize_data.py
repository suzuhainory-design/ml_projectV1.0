import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
train_df = pd.read_csv('/home/ubuntu/upload/train.csv')
test_df = pd.read_csv('/home/ubuntu/upload/test.csv')

print(f"训练集形状: {train_df.shape}")
print(f"测试集形状: {test_df.shape}")

# 创建可视化
fig = plt.figure(figsize=(20, 12))

# 1. 目标变量分布
ax1 = plt.subplot(3, 4, 1)
train_df['Attrition'].value_counts().plot(kind='bar', ax=ax1, color=['#2ecc71', '#e74c3c'])
ax1.set_title('训练集目标变量分布 (Attrition)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Attrition (0=留任, 1=离职)')
ax1.set_ylabel('数量')
ax1.set_xticklabels(['留任 (0)', '离职 (1)'], rotation=0)

# 2. 年龄分布
ax2 = plt.subplot(3, 4, 2)
train_df['Age'].hist(bins=30, ax=ax2, color='skyblue', edgecolor='black')
ax2.set_title('年龄分布', fontsize=12, fontweight='bold')
ax2.set_xlabel('年龄')
ax2.set_ylabel('频数')

# 3. 月收入分布
ax3 = plt.subplot(3, 4, 3)
train_df['MonthlyIncome'].hist(bins=30, ax=ax3, color='lightcoral', edgecolor='black')
ax3.set_title('月收入分布', fontsize=12, fontweight='bold')
ax3.set_xlabel('月收入')
ax3.set_ylabel('频数')

# 4. 工作年限分布
ax4 = plt.subplot(3, 4, 4)
train_df['TotalWorkingYears'].hist(bins=30, ax=ax4, color='lightgreen', edgecolor='black')
ax4.set_title('总工作年限分布', fontsize=12, fontweight='bold')
ax4.set_xlabel('总工作年限')
ax4.set_ylabel('频数')

# 5. 部门分布
ax5 = plt.subplot(3, 4, 5)
train_df['Department'].value_counts().plot(kind='bar', ax=ax5, color='orange')
ax5.set_title('部门分布', fontsize=12, fontweight='bold')
ax5.set_xlabel('部门')
ax5.set_ylabel('数量')
ax5.tick_params(axis='x', rotation=45)

# 6. 性别分布
ax6 = plt.subplot(3, 4, 6)
train_df['Gender'].value_counts().plot(kind='bar', ax=ax6, color=['#3498db', '#e91e63'])
ax6.set_title('性别分布', fontsize=12, fontweight='bold')
ax6.set_xlabel('性别')
ax6.set_ylabel('数量')
ax6.set_xticklabels(ax6.get_xticklabels(), rotation=0)

# 7. 加班情况
ax7 = plt.subplot(3, 4, 7)
train_df['OverTime'].value_counts().plot(kind='bar', ax=ax7, color=['#95a5a6', '#e67e22'])
ax7.set_title('加班情况分布', fontsize=12, fontweight='bold')
ax7.set_xlabel('是否加班')
ax7.set_ylabel('数量')
ax7.set_xticklabels(ax7.get_xticklabels(), rotation=0)

# 8. 婚姻状况
ax8 = plt.subplot(3, 4, 8)
train_df['MaritalStatus'].value_counts().plot(kind='bar', ax=ax8, color='purple')
ax8.set_title('婚姻状况分布', fontsize=12, fontweight='bold')
ax8.set_xlabel('婚姻状况')
ax8.set_ylabel('数量')
ax8.tick_params(axis='x', rotation=45)

# 9. 离职率与年龄关系
ax9 = plt.subplot(3, 4, 9)
for attrition in [0, 1]:
    subset = train_df[train_df['Attrition'] == attrition]
    ax9.hist(subset['Age'], bins=20, alpha=0.6, label=f'Attrition={attrition}')
ax9.set_title('离职率与年龄关系', fontsize=12, fontweight='bold')
ax9.set_xlabel('年龄')
ax9.set_ylabel('频数')
ax9.legend(['留任', '离职'])

# 10. 离职率与月收入关系
ax10 = plt.subplot(3, 4, 10)
for attrition in [0, 1]:
    subset = train_df[train_df['Attrition'] == attrition]
    ax10.hist(subset['MonthlyIncome'], bins=20, alpha=0.6, label=f'Attrition={attrition}')
ax10.set_title('离职率与月收入关系', fontsize=12, fontweight='bold')
ax10.set_xlabel('月收入')
ax10.set_ylabel('频数')
ax10.legend(['留任', '离职'])

# 11. 离职率与加班关系
ax11 = plt.subplot(3, 4, 11)
overtime_attrition = pd.crosstab(train_df['OverTime'], train_df['Attrition'], normalize='index')
overtime_attrition.plot(kind='bar', ax=ax11, color=['#2ecc71', '#e74c3c'])
ax11.set_title('加班与离职率关系', fontsize=12, fontweight='bold')
ax11.set_xlabel('是否加班')
ax11.set_ylabel('比例')
ax11.legend(['留任', '离职'])
ax11.set_xticklabels(ax11.get_xticklabels(), rotation=0)

# 12. 离职率与部门关系
ax12 = plt.subplot(3, 4, 12)
dept_attrition = pd.crosstab(train_df['Department'], train_df['Attrition'], normalize='index')
dept_attrition.plot(kind='bar', ax=ax12, color=['#2ecc71', '#e74c3c'])
ax12.set_title('部门与离职率关系', fontsize=12, fontweight='bold')
ax12.set_xlabel('部门')
ax12.set_ylabel('比例')
ax12.legend(['留任', '离职'])
ax12.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('/home/ubuntu/ml_project/data_visualization.png', dpi=150, bbox_inches='tight')
print("可视化图表已保存到 data_visualization.png")

# 创建相关性热图
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
correlation_matrix = train_df[numeric_cols].corr()

fig2, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('数值特征相关性热图', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('/home/ubuntu/ml_project/correlation_heatmap.png', dpi=150, bbox_inches='tight')
print("相关性热图已保存到 correlation_heatmap.png")

# 分析与Attrition相关性最高的特征
attrition_corr = correlation_matrix['Attrition'].sort_values(ascending=False)
print("\n与Attrition相关性最高的特征:")
print(attrition_corr)

with open('/home/ubuntu/ml_project/attrition_correlation.txt', 'w', encoding='utf-8') as f:
    f.write("与Attrition相关性排序:\n")
    f.write(str(attrition_corr))

print("\n数据可视化分析完成！")
