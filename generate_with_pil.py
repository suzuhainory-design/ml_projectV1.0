"""
使用matplotlib生成图表骨架，然后用PIL添加中文标签
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

def add_chinese_text_to_image(img_path, texts_config):
    """
    在图片上添加中文文本
    texts_config: list of dict with keys: text, position, font_size, color, align
    """
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    
    for config in texts_config:
        font = ImageFont.truetype(FONT_PATH, config.get('font_size', 20))
        draw.text(
            config['position'],
            config['text'],
            font=font,
            fill=config.get('color', 'black'),
            anchor=config.get('anchor', 'lt')
        )
    
    img.save(img_path, dpi=(300, 300))
    print(f"✅ 已添加中文标签: {img_path}")

os.makedirs('diagrams', exist_ok=True)

print("=" * 80)
print("使用PIL重新生成图表")
print("=" * 80)

# ============================================================================
# 图2: 混淆矩阵（重新生成）
# ============================================================================
print("\n生成混淆矩阵...")

fig, ax = plt.subplots(figsize=(8, 6))

confusion_matrix = np.array([[291, 6], [30, 23]])

# 绘制热图
im = ax.imshow(confusion_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=300)

# 使用英文标签
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Retention', 'Attrition'], fontsize=12)
ax.set_yticklabels(['Retention', 'Attrition'], fontsize=12)

# 添加数值
for i in range(2):
    for j in range(2):
        text_color = "white" if confusion_matrix[i, j] > 150 else "black"
        ax.text(j, i, str(confusion_matrix[i, j]),
               ha="center", va="center", color=text_color,
               fontsize=20, fontweight='bold')

ax.set_xlabel('Predicted', fontsize=14, fontweight='bold')
ax.set_ylabel('Actual', fontsize=14, fontweight='bold')
ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Count', fontsize=12)

plt.tight_layout()
temp_path = 'diagrams/confusion_matrix_temp.png'
plt.savefig(temp_path, bbox_inches='tight', dpi=300)
plt.close()

# 用PIL添加中文
img = Image.open(temp_path)
draw = ImageDraw.Draw(img)
font_title = ImageFont.truetype(FONT_PATH, 48)
font_label = ImageFont.truetype(FONT_PATH, 42)
font_info = ImageFont.truetype(FONT_PATH, 33)

# 添加中文标题
draw.text((img.width//2, 40), '集成模型混淆矩阵', font=font_title, fill='black', anchor='mm')

# 添加中文轴标签
draw.text((img.width//2, img.height-60), '预测类别', font=font_label, fill='black', anchor='mm')
draw.text((120, img.height//2), '实际类别', font=font_label, fill='black', anchor='mm', angle=90)

# 添加中文刻度标签
draw.text((img.width//2-300, img.height-180), '留任', font=font_label, fill='black', anchor='mm')
draw.text((img.width//2+300, img.height-180), '离职', font=font_label, fill='black', anchor='mm')
draw.text((280, img.height//2-200), '留任', font=font_label, fill='black', anchor='mm')
draw.text((280, img.height//2+200), '离职', font=font_label, fill='black', anchor='mm')

# 添加统计信息
total = confusion_matrix.sum()
accuracy = (confusion_matrix[0,0] + confusion_matrix[1,1]) / total
precision_0 = confusion_matrix[0,0] / (confusion_matrix[0,0] + confusion_matrix[1,0])
recall_0 = confusion_matrix[0,0] / (confusion_matrix[0,0] + confusion_matrix[0,1])

info_text = f'准确率: {accuracy:.2%}\n留任精确率: {precision_0:.2%}\n留任召回率: {recall_0:.2%}'
y_start = img.height - 400
for i, line in enumerate(info_text.split('\n')):
    draw.text((img.width-300, y_start + i*40), line, font=font_info, fill='black', anchor='rm')

# 添加colorbar标签
draw.text((img.width-120, img.height//2), '样本数量', font=font_label, fill='black', anchor='mm', angle=270)

img.save('diagrams/confusion_matrix.png', dpi=(300, 300))
os.remove(temp_path)
print("✅ 已保存: diagrams/confusion_matrix.png")

# ============================================================================
# 图3: 特征重要性（重新生成）
# ============================================================================
print("\n生成特征重要性...")

if os.path.exists('feature_importance.csv'):
    feature_imp = pd.read_csv('feature_importance.csv')
    top_features = feature_imp.head(15)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors_gradient = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    
    y_pos = np.arange(len(top_features))
    bars = ax.barh(y_pos, top_features['importance'], 
                   color=colors_gradient, edgecolor='black', linewidth=1)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['feature'], fontsize=10)
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 Features', fontsize=14, fontweight='bold', pad=15)
    ax.invert_yaxis()
    
    for i, (bar, val) in enumerate(zip(bars, top_features['importance'])):
        width = bar.get_width()
        ax.text(width + 0.001, bar.get_y() + bar.get_height()/2.,
                f'{val:.4f}',
                ha='left', va='center', fontsize=9, fontweight='bold')
    
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    temp_path = 'diagrams/feature_importance_temp.png'
    plt.savefig(temp_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    # 用PIL添加中文
    img = Image.open(temp_path)
    draw = ImageDraw.Draw(img)
    font_title = ImageFont.truetype(FONT_PATH, 42)
    font_label = ImageFont.truetype(FONT_PATH, 36)
    
    # 添加中文标题
    draw.text((img.width//2, 40), 'Top 15 最重要特征', font=font_title, fill='black', anchor='mm')
    
    # 添加中文x轴标签
    draw.text((img.width//2, img.height-40), '重要性得分', font=font_label, fill='black', anchor='mm')
    
    img.save('diagrams/feature_importance.png', dpi=(300, 300))
    os.remove(temp_path)
    print("✅ 已保存: diagrams/feature_importance.png")

print("\n" + "=" * 80)
print("图表生成完成！")
print("=" * 80)
