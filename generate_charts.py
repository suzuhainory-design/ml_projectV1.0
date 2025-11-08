"""
生成项目相关的可视化图表
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置样式
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

# 创建输出目录
import os
os.makedirs('diagrams', exist_ok=True)

print("=" * 80)
print("生成可视化图表")
print("=" * 80)

# ============================================================================
# 图1: 模型性能对比
# ============================================================================
print("\n生成图1: 模型性能对比...")

fig, ax = plt.subplots(figsize=(12, 6))

models = ['CatBoost', 'XGBoost', 'LightGBM', '集成模型']
accuracies = [0.8886, 0.8857, 0.8886, 0.8971]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

bars = ax.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# 添加数值标签
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.2%}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('准确率', fontsize=14, fontweight='bold')
ax.set_title('模型性能对比 - 测试集准确率', fontsize=16, fontweight='bold', pad=20)
ax.set_ylim([0.85, 0.92])
ax.axhline(y=0.90, color='red', linestyle='--', linewidth=2, alpha=0.5, label='90%基准线')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('diagrams/model_comparison.png', bbox_inches='tight')
print("✅ 已保存: diagrams/model_comparison.png")
plt.close()

# ============================================================================
# 图2: 混淆矩阵热图
# ============================================================================
print("\n生成图2: 混淆矩阵热图...")

fig, ax = plt.subplots(figsize=(8, 6))

confusion_matrix = np.array([[291, 6], [30, 23]])
labels = ['留任', '离职']

sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels, yticklabels=labels,
            cbar_kws={'label': '样本数量'}, ax=ax,
            annot_kws={'size': 16, 'weight': 'bold'})

ax.set_xlabel('预测类别', fontsize=14, fontweight='bold')
ax.set_ylabel('实际类别', fontsize=14, fontweight='bold')
ax.set_title('集成模型混淆矩阵', fontsize=16, fontweight='bold', pad=20)

# 添加统计信息
total = confusion_matrix.sum()
accuracy = (confusion_matrix[0,0] + confusion_matrix[1,1]) / total
precision_0 = confusion_matrix[0,0] / (confusion_matrix[0,0] + confusion_matrix[1,0])
recall_0 = confusion_matrix[0,0] / (confusion_matrix[0,0] + confusion_matrix[0,1])

info_text = f'准确率: {accuracy:.2%}\n留任精确率: {precision_0:.2%}\n留任召回率: {recall_0:.2%}'
ax.text(1.5, -0.3, info_text, fontsize=11, 
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('diagrams/confusion_matrix.png', bbox_inches='tight')
print("✅ 已保存: diagrams/confusion_matrix.png")
plt.close()

# ============================================================================
# 图3: 特征重要性Top 15
# ============================================================================
print("\n生成图3: 特征重要性...")

# 读取特征重要性数据
if os.path.exists('feature_importance.csv'):
    feature_imp = pd.read_csv('feature_importance.csv')
    top_features = feature_imp.head(15)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors_gradient = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    
    bars = ax.barh(range(len(top_features)), top_features['importance'], 
                   color=colors_gradient, edgecolor='black', linewidth=1)
    
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'], fontsize=10)
    ax.set_xlabel('重要性得分', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 最重要特征', fontsize=14, fontweight='bold', pad=15)
    ax.invert_yaxis()
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, top_features['importance'])):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{val:.4f}',
                ha='left', va='center', fontsize=9, fontweight='bold')
    
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('diagrams/feature_importance.png', bbox_inches='tight')
    print("✅ 已保存: diagrams/feature_importance.png")
    plt.close()
else:
    print("⚠️  未找到feature_importance.csv，跳过特征重要性图")

# ============================================================================
# 图4: 训练过程对比
# ============================================================================
print("\n生成图4: 模型优化过程...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 基准模型vs优化模型
stages = ['基准模型', '超参数优化', '高级集成', '最终优化']
catboost_scores = [0.8914, 0.8886, 0.8914, 0.8886]
xgboost_scores = [0.8829, 0.8714, 0.8829, 0.8857]
ensemble_scores = [0.8743, 0.8686, 0.8886, 0.8971]

x = np.arange(len(stages))
width = 0.25

bars1 = ax1.bar(x - width, catboost_scores, width, label='CatBoost', color='#FF6B6B', alpha=0.8)
bars2 = ax1.bar(x, xgboost_scores, width, label='XGBoost', color='#4ECDC4', alpha=0.8)
bars3 = ax1.bar(x + width, ensemble_scores, width, label='集成模型', color='#96CEB4', alpha=0.8)

ax1.set_ylabel('准确率', fontsize=12, fontweight='bold')
ax1.set_title('模型优化过程', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(stages, rotation=15, ha='right')
ax1.legend()
ax1.set_ylim([0.85, 0.92])
ax1.grid(axis='y', alpha=0.3)

# 不同采样策略对比
sampling_methods = ['SMOTE', 'ADASYN', 'SMOTETomek']
cb_sampling = [0.8914, 0.8857, 0.8743]
xgb_sampling = [0.8829, 0.8829, 0.8800]
lgbm_sampling = [0.8771, 0.8686, 0.8771]

x2 = np.arange(len(sampling_methods))
bars1 = ax2.bar(x2 - width, cb_sampling, width, label='CatBoost', color='#FF6B6B', alpha=0.8)
bars2 = ax2.bar(x2, xgb_sampling, width, label='XGBoost', color='#4ECDC4', alpha=0.8)
bars3 = ax2.bar(x2 + width, lgbm_sampling, width, label='LightGBM', color='#45B7D1', alpha=0.8)

ax2.set_ylabel('准确率', fontsize=12, fontweight='bold')
ax2.set_title('不同采样策略对比', fontsize=14, fontweight='bold')
ax2.set_xticks(x2)
ax2.set_xticklabels(sampling_methods)
ax2.legend()
ax2.set_ylim([0.85, 0.92])
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('diagrams/optimization_process.png', bbox_inches='tight')
print("✅ 已保存: diagrams/optimization_process.png")
plt.close()

# ============================================================================
# 图5: 数据分布
# ============================================================================
print("\n生成图5: 数据集分布...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# 数据集大小
datasets = ['训练集', '验证集', '测试集']
sizes = [880, 220, 350]
colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1']

wedges, texts, autotexts = ax1.pie(sizes, labels=datasets, autopct='%1.1f%%',
                                     colors=colors_pie, startangle=90,
                                     textprops={'fontsize': 11, 'weight': 'bold'})
ax1.set_title('数据集划分', fontsize=14, fontweight='bold')

# 类别分布
categories = ['留任', '离职']
train_dist = [922, 178]
colors_bar = ['#96CEB4', '#FF6B6B']

ax2.bar(categories, train_dist, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('样本数量', fontsize=12, fontweight='bold')
ax2.set_title('训练集类别分布', fontsize=14, fontweight='bold')
for i, v in enumerate(train_dist):
    ax2.text(i, v + 20, f'{v}\n({v/sum(train_dist)*100:.1f}%)', 
             ha='center', fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# SMOTE前后对比
labels = ['SMOTE前', 'SMOTE后']
retention = [922, 922]
attrition = [178, 922]

x3 = np.arange(len(labels))
width = 0.35

bars1 = ax3.bar(x3 - width/2, retention, width, label='留任', color='#96CEB4', alpha=0.8)
bars2 = ax3.bar(x3 + width/2, attrition, width, label='离职', color='#FF6B6B', alpha=0.8)

ax3.set_ylabel('样本数量', fontsize=12, fontweight='bold')
ax3.set_title('SMOTE数据平衡', fontsize=14, fontweight='bold')
ax3.set_xticks(x3)
ax3.set_xticklabels(labels)
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# 特征数量变化
stages_feat = ['原始特征', '外部数据', '特征工程', '最终特征']
feature_counts = [31, 37, 51, 51]
colors_line = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

ax4.plot(stages_feat, feature_counts, marker='o', linewidth=3, 
         markersize=10, color='#4ECDC4')
ax4.fill_between(range(len(stages_feat)), feature_counts, alpha=0.3, color='#4ECDC4')
ax4.set_ylabel('特征数量', fontsize=12, fontweight='bold')
ax4.set_title('特征工程进展', fontsize=14, fontweight='bold')
ax4.set_xticks(range(len(stages_feat)))
ax4.set_xticklabels(stages_feat, rotation=15, ha='right')
ax4.grid(alpha=0.3)

for i, v in enumerate(feature_counts):
    ax4.text(i, v + 1, str(v), ha='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('diagrams/data_distribution.png', bbox_inches='tight')
print("✅ 已保存: diagrams/data_distribution.png")
plt.close()

# ============================================================================
# 图6: 性能指标雷达图
# ============================================================================
print("\n生成图6: 性能指标雷达图...")

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

categories = ['准确率', '留任精确率', '留任召回率', '离职精确率', '离职召回率']
values = [0.8971, 0.91, 0.98, 0.79, 0.43]

# 闭合图形
values += values[:1]
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

ax.plot(angles, values, 'o-', linewidth=2, color='#4ECDC4', label='集成模型')
ax.fill(angles, values, alpha=0.25, color='#4ECDC4')

# 添加基准线
baseline = [0.85] * len(angles)
ax.plot(angles, baseline, '--', linewidth=1.5, color='red', alpha=0.5, label='85%基准')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
ax.set_title('模型性能指标雷达图', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax.grid(True)

plt.tight_layout()
plt.savefig('diagrams/performance_radar.png', bbox_inches='tight')
print("✅ 已保存: diagrams/performance_radar.png")
plt.close()

print("\n" + "=" * 80)
print("所有图表生成完成！")
print("=" * 80)
print("\n生成的图表:")
print("1. diagrams/model_comparison.png - 模型性能对比")
print("2. diagrams/confusion_matrix.png - 混淆矩阵热图")
print("3. diagrams/feature_importance.png - 特征重要性")
print("4. diagrams/optimization_process.png - 优化过程")
print("5. diagrams/data_distribution.png - 数据分布")
print("6. diagrams/performance_radar.png - 性能雷达图")
