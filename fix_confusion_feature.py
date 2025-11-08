"""
修复混淆矩阵和特征重要性图的文字重叠问题
完全使用PIL绘制，避免任何重叠
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
print("修复混淆矩阵和特征重要性图")
print("=" * 80)

# ============================================================================
# 图1: 混淆矩阵（完全重绘，无任何英文）
# ============================================================================
print("\n生成混淆矩阵...")

fig, ax = plt.subplots(figsize=(10, 8))

confusion_matrix = np.array([[291, 6], [30, 23]])

# 绘制热图
im = ax.imshow(confusion_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=300)

# 完全清空所有标签
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['', ''])  # 清空x轴标签
ax.set_yticklabels(['', ''])  # 清空y轴标签

# 添加数值
for i in range(2):
    for j in range(2):
        text_color = "white" if confusion_matrix[i, j] > 150 else "black"
        ax.text(j, i, str(confusion_matrix[i, j]),
               ha="center", va="center", color=text_color,
               fontsize=24, fontweight='bold')

ax.set_xlabel('', fontsize=1)  # 清空
ax.set_ylabel('', fontsize=1)  # 清空
ax.set_title('', fontsize=1)  # 清空

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('', fontsize=1)  # 清空colorbar标签

plt.tight_layout()
temp_path = 'diagrams/confusion_matrix_temp.png'
plt.savefig(temp_path, bbox_inches='tight', dpi=300)
plt.close()

# 用PIL添加所有中文（精确定位，完全不重叠）
img = Image.open(temp_path)
draw = ImageDraw.Draw(img)
font_title = ImageFont.truetype(FONT_PATH, 56)
font_label = ImageFont.truetype(FONT_PATH, 48)
font_tick = ImageFont.truetype(FONT_PATH, 44)
font_info = ImageFont.truetype(FONT_PATH, 36)
font_colorbar = ImageFont.truetype(FONT_PATH, 40)

# 添加中文标题（顶部，留足空间）
draw.text((img.width//2, 80), '集成模型混淆矩阵', font=font_title, fill='black', anchor='mm')

# 添加中文轴标签（底部和左侧，远离刻度）
draw.text((img.width//2, img.height-80), '预测类别', font=font_label, fill='black', anchor='mm')
draw.text((150, img.height//2), '实际类别', font=font_label, fill='black', anchor='mm')

# 添加中文刻度标签（精确定位，不重叠）
# x轴刻度
draw.text((img.width//2-280, img.height-200), '留任', font=font_tick, fill='black', anchor='mm')
draw.text((img.width//2+280, img.height-200), '离职', font=font_tick, fill='black', anchor='mm')
# y轴刻度
draw.text((320, img.height//2-240), '留任', font=font_tick, fill='black', anchor='mm')
draw.text((320, img.height//2+240), '离职', font=font_tick, fill='black', anchor='mm')

# 添加统计信息（右下角，不重叠）
total = confusion_matrix.sum()
accuracy = (confusion_matrix[0,0] + confusion_matrix[1,1]) / total
precision_0 = confusion_matrix[0,0] / (confusion_matrix[0,0] + confusion_matrix[1,0])
recall_0 = confusion_matrix[0,0] / (confusion_matrix[0,0] + confusion_matrix[0,1])

info_y = img.height - 450
draw.text((img.width-280, info_y), f'准确率: {accuracy:.2%}', font=font_info, fill='black', anchor='rm')
draw.text((img.width-280, info_y+50), f'留任精确率: {precision_0:.2%}', font=font_info, fill='black', anchor='rm')
draw.text((img.width-280, info_y+100), f'留任召回率: {recall_0:.2%}', font=font_info, fill='black', anchor='rm')

# 添加colorbar标签（右侧垂直）
draw.text((img.width-100, img.height//2), '样本数量', font=font_colorbar, fill='black', anchor='mm')

img.save('diagrams/confusion_matrix.png', dpi=(300, 300))
os.remove(temp_path)
print("✅ 已保存: diagrams/confusion_matrix.png")

# ============================================================================
# 图2: 特征重要性（完全重绘，无任何英文）
# ============================================================================
print("\n生成特征重要性...")

if os.path.exists('feature_importance.csv'):
    feature_imp = pd.read_csv('feature_importance.csv')
    top_features = feature_imp.head(15)
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    colors_gradient = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    
    y_pos = np.arange(len(top_features))
    bars = ax.barh(y_pos, top_features['importance'], 
                   color=colors_gradient, edgecolor='black', linewidth=1)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['feature'], fontsize=11)
    ax.set_xlabel('', fontsize=1)  # 清空
    ax.set_title('', fontsize=1)  # 清空
    ax.invert_yaxis()
    
    for i, (bar, val) in enumerate(zip(bars, top_features['importance'])):
        width = bar.get_width()
        ax.text(width + 0.001, bar.get_y() + bar.get_height()/2.,
                f'{val:.4f}',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    temp_path = 'diagrams/feature_importance_temp.png'
    plt.savefig(temp_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    # 用PIL添加中文（精确定位，完全不重叠）
    img = Image.open(temp_path)
    draw = ImageDraw.Draw(img)
    font_title = ImageFont.truetype(FONT_PATH, 52)
    font_label = ImageFont.truetype(FONT_PATH, 44)
    
    # 添加中文标题（顶部，留足空间）
    draw.text((img.width//2, 80), 'Top 15 最重要特征', font=font_title, fill='black', anchor='mm')
    
    # 添加中文x轴标签（底部，远离刻度）
    draw.text((img.width//2, img.height-60), '重要性得分', font=font_label, fill='black', anchor='mm')
    
    img.save('diagrams/feature_importance.png', dpi=(300, 300))
    os.remove(temp_path)
    print("✅ 已保存: diagrams/feature_importance.png")
else:
    print("⚠️  未找到feature_importance.csv文件")

print("\n" + "=" * 80)
print("修复完成！")
print("=" * 80)
