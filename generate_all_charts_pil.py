"""
使用PIL完整重新生成所有图表
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import os

# 字体路径
FONT_PATH = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'

os.makedirs('diagrams', exist_ok=True)

print("=" * 80)
print("使用PIL重新生成所有图表")
print("=" * 80)

# ============================================================================
# 图1: 模型性能对比
# ============================================================================
print("\n生成图1: 模型性能对比...")

fig, ax = plt.subplots(figsize=(12, 6))

models = ['CatBoost', 'XGBoost', 'LightGBM', 'Ensemble']
accuracies = [0.8886, 0.8857, 0.8886, 0.8971]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

bars = ax.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.2%}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
ax.set_ylim([0.85, 0.92])
ax.axhline(y=0.90, color='red', linestyle='--', linewidth=2, alpha=0.5, label='90% baseline')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
temp_path = 'diagrams/model_comparison_temp.png'
plt.savefig(temp_path, bbox_inches='tight', dpi=300)
plt.close()

# 用PIL添加中文
img = Image.open(temp_path)
draw = ImageDraw.Draw(img)
font_title = ImageFont.truetype(FONT_PATH, 48)
font_label = ImageFont.truetype(FONT_PATH, 42)
font_legend = ImageFont.truetype(FONT_PATH, 33)

# 添加中文标题
draw.text((img.width//2, 50), '模型性能对比 - 测试集准确率', font=font_title, fill='black', anchor='mm')

# 添加中文y轴标签
draw.text((120, img.height//2), '准确率', font=font_label, fill='black', anchor='mm', angle=90)

# 添加中文x轴最后一个标签
draw.text((img.width-350, img.height-180), '集成模型', font=font_label, fill='black', anchor='mm')

# 添加中文图例
draw.text((img.width-250, 150), '90%基准线', font=font_legend, fill='red', anchor='lm')

img.save('diagrams/model_comparison.png', dpi=(300, 300))
os.remove(temp_path)
print("✅ 已保存: diagrams/model_comparison.png")

# ============================================================================
# 图4: 优化过程
# ============================================================================
print("\n生成图4: 优化过程...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

stages = ['Baseline', 'Tuned', 'Advanced', 'Final']
catboost_scores = [0.8914, 0.8886, 0.8914, 0.8886]
xgboost_scores = [0.8829, 0.8714, 0.8829, 0.8857]
ensemble_scores = [0.8743, 0.8686, 0.8886, 0.8971]

x = np.arange(len(stages))
width = 0.25

ax1.bar(x - width, catboost_scores, width, label='CatBoost', color='#FF6B6B', alpha=0.8)
ax1.bar(x, xgboost_scores, width, label='XGBoost', color='#4ECDC4', alpha=0.8)
ax1.bar(x + width, ensemble_scores, width, label='Ensemble', color='#96CEB4', alpha=0.8)

ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Model Optimization', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(stages, rotation=15, ha='right')
ax1.legend()
ax1.set_ylim([0.85, 0.92])
ax1.grid(axis='y', alpha=0.3)

sampling_methods = ['SMOTE', 'ADASYN', 'SMOTETomek']
cb_sampling = [0.8914, 0.8857, 0.8743]
xgb_sampling = [0.8829, 0.8829, 0.8800]
lgbm_sampling = [0.8771, 0.8686, 0.8771]

x2 = np.arange(len(sampling_methods))
ax2.bar(x2 - width, cb_sampling, width, label='CatBoost', color='#FF6B6B', alpha=0.8)
ax2.bar(x2, xgb_sampling, width, label='XGBoost', color='#4ECDC4', alpha=0.8)
ax2.bar(x2 + width, lgbm_sampling, width, label='LightGBM', color='#45B7D1', alpha=0.8)

ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax2.set_title('Sampling Strategy', fontsize=14, fontweight='bold')
ax2.set_xticks(x2)
ax2.set_xticklabels(sampling_methods)
ax2.legend()
ax2.set_ylim([0.85, 0.92])
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
temp_path = 'diagrams/optimization_process_temp.png'
plt.savefig(temp_path, bbox_inches='tight', dpi=300)
plt.close()

# 用PIL添加中文
img = Image.open(temp_path)
draw = ImageDraw.Draw(img)
font_title = ImageFont.truetype(FONT_PATH, 42)
font_label = ImageFont.truetype(FONT_PATH, 36)
font_xtick = ImageFont.truetype(FONT_PATH, 30)
font_legend = ImageFont.truetype(FONT_PATH, 28)

# 左图中文
draw.text((img.width//4, 50), '模型优化过程', font=font_title, fill='black', anchor='mm')
draw.text((100, img.height//2), '准确率', font=font_label, fill='black', anchor='mm', angle=90)
# x轴标签
draw.text((img.width//4-400, img.height-120), '基准模型', font=font_xtick, fill='black', anchor='mm')
draw.text((img.width//4-130, img.height-100), '超参数优化', font=font_xtick, fill='black', anchor='mm', angle=15)
draw.text((img.width//4+140, img.height-100), '高级集成', font=font_xtick, fill='black', anchor='mm', angle=15)
draw.text((img.width//4+410, img.height-100), '最终优化', font=font_xtick, fill='black', anchor='mm', angle=15)
# 图例
draw.text((img.width//4+200, img.height-400), '集成模型', font=font_legend, fill='#96CEB4', anchor='lm')

# 右图中文
draw.text((img.width*3//4, 50), '不同采样策略对比', font=font_title, fill='black', anchor='mm')
draw.text((img.width//2+100, img.height//2), '准确率', font=font_label, fill='black', anchor='mm', angle=90)

img.save('diagrams/optimization_process.png', dpi=(300, 300))
os.remove(temp_path)
print("✅ 已保存: diagrams/optimization_process.png")

# ============================================================================
# 图5: 数据分布
# ============================================================================
print("\n生成图5: 数据分布...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# 饼图
datasets = ['Train', 'Val', 'Test']
sizes = [880, 220, 350]
colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1']

ax1.pie(sizes, labels=datasets, autopct='%1.1f%%',
        colors=colors_pie, startangle=90,
        textprops={'fontsize': 11, 'weight': 'bold'})
ax1.set_title('Dataset Split', fontsize=14, fontweight='bold')

# 类别分布
categories = ['Retention', 'Attrition']
train_dist = [922, 178]
colors_bar = ['#96CEB4', '#FF6B6B']

ax2.bar(categories, train_dist, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
ax2.set_title('Class Distribution', fontsize=14, fontweight='bold')
for i, v in enumerate(train_dist):
    ax2.text(i, v + 20, f'{v}\n({v/sum(train_dist)*100:.1f}%)', 
             ha='center', fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# SMOTE对比
labels = ['Before', 'After']
retention = [922, 922]
attrition = [178, 922]

x3 = np.arange(len(labels))
width = 0.35

ax3.bar(x3 - width/2, retention, width, label='Retention', color='#96CEB4', alpha=0.8)
ax3.bar(x3 + width/2, attrition, width, label='Attrition', color='#FF6B6B', alpha=0.8)

ax3.set_ylabel('Count', fontsize=12, fontweight='bold')
ax3.set_title('SMOTE Balancing', fontsize=14, fontweight='bold')
ax3.set_xticks(x3)
ax3.set_xticklabels(labels)
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# 特征进展
stages_feat = ['Original', 'External', 'Engineered', 'Final']
feature_counts = [31, 37, 51, 51]

ax4.plot(stages_feat, feature_counts, marker='o', linewidth=3, 
         markersize=10, color='#4ECDC4')
ax4.fill_between(range(len(stages_feat)), feature_counts, alpha=0.3, color='#4ECDC4')
ax4.set_ylabel('Count', fontsize=12, fontweight='bold')
ax4.set_title('Feature Engineering', fontsize=14, fontweight='bold')
ax4.set_xticks(range(len(stages_feat)))
ax4.set_xticklabels(stages_feat, rotation=15, ha='right')
ax4.grid(alpha=0.3)

for i, v in enumerate(feature_counts):
    ax4.text(i, v + 1, str(v), ha='center', fontweight='bold', fontsize=11)

plt.tight_layout()
temp_path = 'diagrams/data_distribution_temp.png'
plt.savefig(temp_path, bbox_inches='tight', dpi=300)
plt.close()

# 用PIL添加中文
img = Image.open(temp_path)
draw = ImageDraw.Draw(img)
font_title = ImageFont.truetype(FONT_PATH, 42)
font_label = ImageFont.truetype(FONT_PATH, 36)
font_legend = ImageFont.truetype(FONT_PATH, 30)
font_xtick = ImageFont.truetype(FONT_PATH, 28)

# 左上 - 饼图
draw.text((img.width//4, 80), '数据集划分', font=font_title, fill='black', anchor='mm')
draw.text((img.width//4-200, img.height//4-50), '训练集', font=font_legend, fill='black', anchor='mm')
draw.text((img.width//4+150, img.height//4-150), '验证集', font=font_legend, fill='black', anchor='mm')
draw.text((img.width//4+200, img.height//4+100), '测试集', font=font_legend, fill='black', anchor='mm')

# 右上 - 类别分布
draw.text((img.width*3//4, 80), '训练集类别分布', font=font_title, fill='black', anchor='mm')
draw.text((img.width//2+120, img.height//4), '样本数量', font=font_label, fill='black', anchor='mm', angle=90)
draw.text((img.width*3//4-200, img.height//2-150), '留任', font=font_legend, fill='black', anchor='mm')
draw.text((img.width*3//4+200, img.height//2-150), '离职', font=font_legend, fill='black', anchor='mm')

# 左下 - SMOTE
draw.text((img.width//4, img.height//2+80), 'SMOTE数据平衡', font=font_title, fill='black', anchor='mm')
draw.text((100, img.height*3//4), '样本数量', font=font_label, fill='black', anchor='mm', angle=90)
draw.text((img.width//4-200, img.height-150), 'SMOTE前', font=font_xtick, fill='black', anchor='mm')
draw.text((img.width//4+200, img.height-150), 'SMOTE后', font=font_xtick, fill='black', anchor='mm')
draw.text((img.width//4-200, img.height*3//4-250), '留任', font=font_legend, fill='#96CEB4', anchor='lm')
draw.text((img.width//4-200, img.height*3//4-210), '离职', font=font_legend, fill='#FF6B6B', anchor='lm')

# 右下 - 特征工程
draw.text((img.width*3//4, img.height//2+80), '特征工程进展', font=font_title, fill='black', anchor='mm')
draw.text((img.width//2+120, img.height*3//4), '特征数量', font=font_label, fill='black', anchor='mm', angle=90)
draw.text((img.width*3//4-400, img.height-130), '原始特征', font=font_xtick, fill='black', anchor='mm', angle=15)
draw.text((img.width*3//4-130, img.height-110), '外部数据', font=font_xtick, fill='black', anchor='mm', angle=15)
draw.text((img.width*3//4+140, img.height-110), '特征工程', font=font_xtick, fill='black', anchor='mm', angle=15)
draw.text((img.width*3//4+410, img.height-110), '最终特征', font=font_xtick, fill='black', anchor='mm', angle=15)

img.save('diagrams/data_distribution.png', dpi=(300, 300))
os.remove(temp_path)
print("✅ 已保存: diagrams/data_distribution.png")

# ============================================================================
# 图6: 性能雷达图
# ============================================================================
print("\n生成图6: 性能雷达图...")

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

categories = ['Accuracy', 'Ret. Precision', 'Ret. Recall', 'Attr. Precision', 'Attr. Recall']
values = [0.8971, 0.91, 0.98, 0.79, 0.43]

values += values[:1]
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

ax.plot(angles, values, 'o-', linewidth=2, color='#4ECDC4', label='Ensemble')
ax.fill(angles, values, alpha=0.25, color='#4ECDC4')

baseline = [0.85] * len(angles)
ax.plot(angles, baseline, '--', linewidth=1.5, color='red', alpha=0.5, label='85% baseline')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
ax.set_title('Performance Radar', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax.grid(True)

plt.tight_layout()
temp_path = 'diagrams/performance_radar_temp.png'
plt.savefig(temp_path, bbox_inches='tight', dpi=300)
plt.close()

# 用PIL添加中文
img = Image.open(temp_path)
draw = ImageDraw.Draw(img)
font_title = ImageFont.truetype(FONT_PATH, 42)
font_label = ImageFont.truetype(FONT_PATH, 32)
font_legend = ImageFont.truetype(FONT_PATH, 30)

# 添加中文标题
draw.text((img.width//2, 50), '模型性能指标雷达图', font=font_title, fill='black', anchor='mm')

# 添加中文标签（替换英文）
draw.text((img.width//2, 200), '准确率', font=font_label, fill='black', anchor='mm')
draw.text((img.width-250, img.height//2-200), '留任精确率', font=font_label, fill='black', anchor='lm')
draw.text((img.width-250, img.height//2+200), '留任召回率', font=font_label, fill='black', anchor='lm')
draw.text((250, img.height//2+200), '离职精确率', font=font_label, fill='black', anchor='rm')
draw.text((250, img.height//2-200), '离职召回率', font=font_label, fill='black', anchor='rm')

# 添加中文图例
draw.text((img.width-280, 150), '集成模型', font=font_legend, fill='#4ECDC4', anchor='lm')
draw.text((img.width-280, 190), '85%基准', font=font_legend, fill='red', anchor='lm')

img.save('diagrams/performance_radar.png', dpi=(300, 300))
os.remove(temp_path)
print("✅ 已保存: diagrams/performance_radar.png")

print("\n" + "=" * 80)
print("所有图表生成完成！")
print("=" * 80)
