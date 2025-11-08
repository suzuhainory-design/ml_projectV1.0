"""
生成所有模型准确度评估对比图
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# 字体路径
FONT_PATH = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'

os.makedirs('diagrams', exist_ok=True)

print("=" * 80)
print("生成模型准确度评估图")
print("=" * 80)

# ============================================================================
# 综合准确度评估图
# ============================================================================
print("\n生成综合准确度评估图...")

fig = plt.figure(figsize=(16, 12))

# 创建2x2子图布局
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

# ============================================================================
# 子图1: 所有模型在训练集和测试集上的表现
# ============================================================================
ax1 = fig.add_subplot(gs[0, :])

models = ['CatBoost', 'XGBoost', 'LightGBM', 'Ensemble']
train_acc = [0.9205, 0.9193, 0.9159, 0.9250]
test_acc = [0.8886, 0.8857, 0.8886, 0.8971]

x = np.arange(len(models))
width = 0.35

bars1 = ax1.bar(x - width/2, train_acc, width, label='Train', 
                color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax1.bar(x + width/2, test_acc, width, label='Test',
                color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1.5)

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{height:.2%}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax1.set_ylabel('')
ax1.set_title('', fontsize=1)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=12)
ax1.legend(loc='lower right', fontsize=11)
ax1.set_ylim([0.85, 0.95])
ax1.grid(axis='y', alpha=0.3)
ax1.axhline(y=0.90, color='green', linestyle='--', linewidth=1.5, alpha=0.5)

# ============================================================================
# 子图2: 不同优化阶段的测试集准确率
# ============================================================================
ax2 = fig.add_subplot(gs[1, 0])

stages = ['Baseline', 'Hyperparameter\nTuning', 'Advanced\nEnsemble', 'Final\nOptimization']
catboost_stages = [0.8914, 0.8886, 0.8914, 0.8886]
xgboost_stages = [0.8829, 0.8714, 0.8829, 0.8857]
lightgbm_stages = [0.8771, 0.8686, 0.8886, 0.8886]
ensemble_stages = [0.8743, 0.8686, 0.8886, 0.8971]

x2 = np.arange(len(stages))
ax2.plot(x2, catboost_stages, marker='o', linewidth=2, markersize=8, 
         label='CatBoost', color='#FF6B6B')
ax2.plot(x2, xgboost_stages, marker='s', linewidth=2, markersize=8,
         label='XGBoost', color='#4ECDC4')
ax2.plot(x2, lightgbm_stages, marker='^', linewidth=2, markersize=8,
         label='LightGBM', color='#45B7D1')
ax2.plot(x2, ensemble_stages, marker='D', linewidth=2.5, markersize=8,
         label='Ensemble', color='#96CEB4')

ax2.set_ylabel('')
ax2.set_title('', fontsize=1)
ax2.set_xticks(x2)
ax2.set_xticklabels(stages, fontsize=9)
ax2.legend(loc='lower right', fontsize=9)
ax2.set_ylim([0.86, 0.91])
ax2.grid(alpha=0.3)

# ============================================================================
# 子图3: 不同采样策略的准确率对比
# ============================================================================
ax3 = fig.add_subplot(gs[1, 1])

sampling = ['SMOTE', 'ADASYN', 'SMOTETomek']
cb_sampling = [0.8914, 0.8857, 0.8743]
xgb_sampling = [0.8829, 0.8829, 0.8800]
lgbm_sampling = [0.8771, 0.8686, 0.8771]

x3 = np.arange(len(sampling))
width3 = 0.25

ax3.bar(x3 - width3, cb_sampling, width3, label='CatBoost',
        color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1)
ax3.bar(x3, xgb_sampling, width3, label='XGBoost',
        color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=1)
ax3.bar(x3 + width3, lgbm_sampling, width3, label='LightGBM',
        color='#45B7D1', alpha=0.8, edgecolor='black', linewidth=1)

ax3.set_ylabel('')
ax3.set_title('', fontsize=1)
ax3.set_xticks(x3)
ax3.set_xticklabels(sampling, fontsize=10)
ax3.legend(loc='lower right', fontsize=9)
ax3.set_ylim([0.86, 0.90])
ax3.grid(axis='y', alpha=0.3)

plt.tight_layout()
temp_path = 'diagrams/accuracy_evaluation_temp.png'
plt.savefig(temp_path, bbox_inches='tight', dpi=300)
plt.close()

# 用PIL添加中文
img = Image.open(temp_path)
draw = ImageDraw.Draw(img)
font_main_title = ImageFont.truetype(FONT_PATH, 60)
font_title = ImageFont.truetype(FONT_PATH, 44)
font_label = ImageFont.truetype(FONT_PATH, 40)

# 主标题
draw.text((img.width//2, 60), '模型准确度综合评估', font=font_main_title, fill='black', anchor='mm')

# 子图1标题和标签
draw.text((img.width//2, 180), '训练集 vs 测试集准确率对比', font=font_title, fill='black', anchor='mm')
draw.text((150, img.height//4+50), '准确率', font=font_label, fill='black', anchor='mm')
draw.text((img.width-350, img.height//4-80), '90%基准线', font=ImageFont.truetype(FONT_PATH, 32), fill='green', anchor='lm')

# 子图2标题和标签
draw.text((img.width//4, img.height//2+100), '优化过程准确率变化', font=font_title, fill='black', anchor='mm')
draw.text((120, img.height*3//4), '准确率', font=font_label, fill='black', anchor='mm')
# 阶段标签
draw.text((img.width//4-450, img.height-120), '基准模型', font=ImageFont.truetype(FONT_PATH, 28), fill='black', anchor='mm')
draw.text((img.width//4-150, img.height-105), '超参数\n优化', font=ImageFont.truetype(FONT_PATH, 26), fill='black', anchor='mm')
draw.text((img.width//4+150, img.height-105), '高级\n集成', font=ImageFont.truetype(FONT_PATH, 26), fill='black', anchor='mm')
draw.text((img.width//4+450, img.height-105), '最终\n优化', font=ImageFont.truetype(FONT_PATH, 26), fill='black', anchor='mm')

# 子图3标题和标签
draw.text((img.width*3//4, img.height//2+100), '采样策略准确率对比', font=font_title, fill='black', anchor='mm')
draw.text((img.width//2+150, img.height*3//4), '准确率', font=font_label, fill='black', anchor='mm')

img.save('diagrams/accuracy_evaluation.png', dpi=(300, 300))
os.remove(temp_path)
print("✅ 已保存: diagrams/accuracy_evaluation.png")

# ============================================================================
# 详细准确率对比表格图
# ============================================================================
print("\n生成详细准确率对比表格图...")

fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('tight')
ax.axis('off')

# 准备表格数据
table_data = [
    ['模型', '训练集准确率', '验证集准确率', '测试集准确率', '过拟合程度'],
    ['CatBoost', '92.05%', '89.09%', '88.86%', '3.19%'],
    ['XGBoost', '91.93%', '88.64%', '88.57%', '3.36%'],
    ['LightGBM', '91.59%', '88.64%', '88.86%', '2.73%'],
    ['集成模型', '92.50%', '89.55%', '89.71%', '2.79%'],
]

# 创建表格
table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 3)

# 设置表头样式
for i in range(5):
    cell = table[(0, i)]
    cell.set_facecolor('#4ECDC4')
    cell.set_text_props(weight='bold', color='white', fontsize=14)

# 设置数据行样式
colors = ['#FFE5E5', '#E5F5FF', '#E5F5FF', '#E5FFE5']
for i in range(1, 5):
    for j in range(5):
        cell = table[(i, j)]
        cell.set_facecolor(colors[i-1])
        if j == 0:
            cell.set_text_props(weight='bold', fontsize=13)
        # 高亮最佳值
        if i == 4 and j > 0:  # 集成模型行
            cell.set_text_props(weight='bold', color='green', fontsize=13)

plt.tight_layout()
temp_path = 'diagrams/accuracy_table_temp.png'
plt.savefig(temp_path, bbox_inches='tight', dpi=300)
plt.close()

# 用PIL添加中文标题
img = Image.open(temp_path)
draw = ImageDraw.Draw(img)
font_title = ImageFont.truetype(FONT_PATH, 56)

draw.text((img.width//2, 80), '模型准确率详细对比表', font=font_title, fill='black', anchor='mm')

# 添加说明
font_note = ImageFont.truetype(FONT_PATH, 32)
draw.text((img.width//2, img.height-80), '注: 过拟合程度 = 训练集准确率 - 测试集准确率', 
          font=font_note, fill='gray', anchor='mm')

img.save('diagrams/accuracy_table.png', dpi=(300, 300))
os.remove(temp_path)
print("✅ 已保存: diagrams/accuracy_table.png")

print("\n" + "=" * 80)
print("所有准确度评估图生成完成！")
print("=" * 80)
