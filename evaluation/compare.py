#!/usr/bin/env python3
"""
降噪测评对比分析脚本
用于对比两个CSV文件中同名文件的评估指标差异

使用方法:
    python compare_noise_reduction.py <csv1_path> <csv2_path> [--name1 NAME1] [--name2 NAME2] [--output OUTPUT_DIR]

示例:
    python compare_noise_reduction.py method_a.csv method_b.csv --name1 "Method A" --name2 "Method B" --output ./results
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import argparse
import os
from datetime import datetime
from matplotlib.lines import Line2D

# 设置matplotlib
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def parse_filename(filename):
    """解析文件名，提取说话者、噪声类型和SNR信息"""
    pattern = r'^(\d+)_(\d+)_(.+)_snr([+-]?\d+)dB_noisy\.wav$'
    match = re.match(pattern, filename)
    if match:
        return {
            'id': match.group(1),
            'speaker': int(match.group(2)),
            'noise_type': match.group(3),
            'snr': int(match.group(4))
        }
    return None


def load_and_parse_csv(csv_path):
    """加载CSV文件并解析文件名"""
    df = pd.read_csv(csv_path)
    
    parsed = df['filename'].apply(parse_filename)
    df['speaker'] = parsed.apply(lambda x: x['speaker'] if x else None)
    df['noise_type'] = parsed.apply(lambda x: x['noise_type'] if x else None)
    df['snr'] = parsed.apply(lambda x: x['snr'] if x else None)
    
    return df


def create_comparison_charts(df1, df2, name1, name2, output_dir):
    """生成对比图表"""
    
    metrics = ['SI-SNR', 'PESQ', 'OVRL', 'SIG', 'BAK', 'P808_MOS']
    metric_labels = {
        'SI-SNR': 'SI-SNR (dB)',
        'PESQ': 'PESQ Score',
        'OVRL': 'Overall Quality',
        'SIG': 'Signal Quality',
        'BAK': 'Background Quality',
        'P808_MOS': 'P.808 MOS'
    }
    
    noise_labels = {
        'fan_noise_level1': 'Fan Noise L1',
        'fan_noise_level2': 'Fan Noise L2',
        'clapping': 'Clapping',
        'door_knock': 'Door Knock',
        'keyboard_loud': 'Keyboard Loud',
        'keyboard_soft': 'Keyboard Soft'
    }
    
    snr_levels = sorted(df1['snr'].dropna().unique())
    noise_types = df1['noise_type'].dropna().unique()
    
    # 颜色方案：蓝色系 vs 红色系
    color1 = '#1f77b4'  # 蓝色
    color2 = '#d62728'  # 红色
    
    # ========== 图1: 按噪声类型对比各指标（所有说话者平均）==========
    fig1, axes1 = plt.subplots(2, 3, figsize=(16, 10))
    fig1.suptitle(f'Performance Comparison by Noise Type\n{name1} (Blue) vs {name2} (Red)', 
                  fontsize=14, fontweight='bold')
    
    for idx, metric in enumerate(metrics):
        ax = axes1[idx // 3, idx % 3]
        
        for noise_type in noise_types:
            # 数据集1
            noise_data1 = df1[df1['noise_type'] == noise_type]
            avg_by_snr1 = noise_data1.groupby('snr')[metric].mean()
            
            # 数据集2
            noise_data2 = df2[df2['noise_type'] == noise_type]
            avg_by_snr2 = noise_data2.groupby('snr')[metric].mean()
            
            label = noise_labels.get(noise_type, noise_type)
            
            # 用实线表示数据集1，虚线表示数据集2
            ax.plot(avg_by_snr1.index, avg_by_snr1.values, 
                    marker='o', linewidth=2, markersize=5, linestyle='-',
                    color=color1, alpha=0.7)
            ax.plot(avg_by_snr2.index, avg_by_snr2.values, 
                    marker='s', linewidth=2, markersize=5, linestyle='--',
                    color=color2, alpha=0.7)
        
        ax.set_xlabel('SNR (dB)', fontsize=10)
        ax.set_ylabel(metric_labels[metric], fontsize=10)
        ax.set_title(metric_labels[metric], fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(snr_levels)
    
    # 添加图例
    legend_elements = [
        Line2D([0], [0], color=color1, linestyle='-', marker='o', label=name1),
        Line2D([0], [0], color=color2, linestyle='--', marker='s', label=name2)
    ]
    fig1.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    chart1_path = os.path.join(output_dir, 'chart1_comparison_by_noise_type.png')
    fig1.savefig(chart1_path, dpi=150, bbox_inches='tight')
    print(f"已保存: {chart1_path}")
    
    # ========== 图2: 各噪声类型分开对比 ==========
    fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))
    fig2.suptitle(f'SI-SNR Comparison by Noise Type\n{name1} (Blue ●) vs {name2} (Red ■)', 
                  fontsize=14, fontweight='bold')
    
    for idx, noise_type in enumerate(list(noise_types)[:6]):
        ax = axes2[idx // 3, idx % 3]
        
        noise_data1 = df1[df1['noise_type'] == noise_type]
        noise_data2 = df2[df2['noise_type'] == noise_type]
        
        avg1 = noise_data1.groupby('snr')['SI-SNR'].mean()
        avg2 = noise_data2.groupby('snr')['SI-SNR'].mean()
        
        ax.plot(avg1.index, avg1.values, marker='o', linewidth=2.5, markersize=8,
                color=color1, label=name1)
        ax.plot(avg2.index, avg2.values, marker='s', linewidth=2.5, markersize=8,
                color=color2, label=name2)
        
        # 填充差异区域
        common_snr = sorted(set(avg1.index) & set(avg2.index))
        if common_snr:
            v1 = [avg1[s] for s in common_snr]
            v2 = [avg2[s] for s in common_snr]
            ax.fill_between(common_snr, v1, v2, alpha=0.2, color='gray')
        
        ax.set_xlabel('SNR (dB)', fontsize=10)
        ax.set_ylabel('SI-SNR (dB)', fontsize=10)
        ax.set_title(noise_labels.get(noise_type, noise_type), fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(snr_levels)
        ax.legend(loc='lower right', fontsize=8)
    
    plt.tight_layout()
    chart2_path = os.path.join(output_dir, 'chart2_sisnr_by_noise_type.png')
    fig2.savefig(chart2_path, dpi=150, bbox_inches='tight')
    print(f"已保存: {chart2_path}")
    
    # ========== 图3: 柱状图对比（按噪声类型平均）==========
    fig3, axes3 = plt.subplots(2, 3, figsize=(16, 10))
    fig3.suptitle(f'Average Metrics Comparison by Noise Type\n{name1} vs {name2}', 
                  fontsize=14, fontweight='bold')
    
    x = np.arange(len(noise_types))
    width = 0.35
    
    for idx, metric in enumerate(metrics):
        ax = axes3[idx // 3, idx % 3]
        
        avg1 = df1.groupby('noise_type')[metric].mean()
        avg2 = df2.groupby('noise_type')[metric].mean()
        
        vals1 = [avg1.get(n, 0) for n in noise_types]
        vals2 = [avg2.get(n, 0) for n in noise_types]
        
        bars1 = ax.bar(x - width/2, vals1, width, label=name1, color=color1, alpha=0.8)
        bars2 = ax.bar(x + width/2, vals2, width, label=name2, color=color2, alpha=0.8)
        
        ax.set_xlabel('Noise Type', fontsize=10)
        ax.set_ylabel(metric_labels[metric], fontsize=10)
        ax.set_title(metric_labels[metric], fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([noise_labels.get(n, n)[:10] for n in noise_types], rotation=45, ha='right', fontsize=8)
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    chart3_path = os.path.join(output_dir, 'chart3_bar_comparison.png')
    fig3.savefig(chart3_path, dpi=150, bbox_inches='tight')
    print(f"已保存: {chart3_path}")
    
    # ========== 图4: 差异热力图 ==========
    fig4, ax4 = plt.subplots(figsize=(12, 8))
    
    # 计算各噪声类型和SNR下的SI-SNR差异
    diff_matrix = []
    for noise_type in noise_types:
        row = []
        for snr in snr_levels:
            v1 = df1[(df1['noise_type'] == noise_type) & (df1['snr'] == snr)]['SI-SNR'].mean()
            v2 = df2[(df2['noise_type'] == noise_type) & (df2['snr'] == snr)]['SI-SNR'].mean()
            diff = v2 - v1  # 正值表示name2更好
            row.append(diff if not np.isnan(diff) else 0)
        diff_matrix.append(row)
    
    diff_matrix = np.array(diff_matrix)
    
    im = ax4.imshow(diff_matrix, cmap='RdBu_r', aspect='auto', vmin=-3, vmax=3)
    
    ax4.set_xticks(range(len(snr_levels)))
    ax4.set_xticklabels([f'{s}dB' for s in snr_levels])
    ax4.set_yticks(range(len(noise_types)))
    ax4.set_yticklabels([noise_labels.get(n, n) for n in noise_types])
    
    # 添加数值标注
    for i in range(len(noise_types)):
        for j in range(len(snr_levels)):
            text = ax4.text(j, i, f'{diff_matrix[i, j]:.2f}',
                           ha='center', va='center', color='black', fontsize=9)
    
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label(f'SI-SNR Difference ({name2} - {name1}) dB', fontsize=10)
    
    ax4.set_xlabel('SNR Level', fontsize=11)
    ax4.set_ylabel('Noise Type', fontsize=11)
    ax4.set_title(f'SI-SNR Difference Heatmap\nPositive (Red) = {name2} better, Negative (Blue) = {name1} better', 
                  fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    chart4_path = os.path.join(output_dir, 'chart4_difference_heatmap.png')
    fig4.savefig(chart4_path, dpi=150, bbox_inches='tight')
    print(f"已保存: {chart4_path}")
    
    plt.close('all')
    
    return [chart1_path, chart2_path, chart3_path, chart4_path]


def generate_report(df1, df2, name1, name2, output_dir):
    """生成对比分析报告"""
    
    metrics = ['SI-SNR', 'PESQ', 'OVRL', 'SIG', 'BAK', 'P808_MOS']
    noise_labels = {
        'fan_noise_level1': 'Fan Noise Level 1',
        'fan_noise_level2': 'Fan Noise Level 2',
        'clapping': 'Clapping',
        'door_knock': 'Door Knock',
        'keyboard_loud': 'Keyboard Loud',
        'keyboard_soft': 'Keyboard Soft'
    }
    
    noise_types = df1['noise_type'].dropna().unique()
    snr_levels = sorted(df1['snr'].dropna().unique())
    
    report = []
    report.append(f"# 降噪测评对比分析报告")
    report.append(f"\n**对比对象**: {name1} vs {name2}")
    report.append(f"\n**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n---\n")
    
    # 1. 数据概览
    report.append(f"## 1. 数据概览\n")
    report.append(f"| 项目 | {name1} | {name2} |")
    report.append(f"|------|---------|---------|")
    report.append(f"| 样本数量 | {len(df1)} | {len(df2)} |")
    report.append(f"| 噪声类型数 | {df1['noise_type'].nunique()} | {df2['noise_type'].nunique()} |")
    report.append(f"| 说话者数 | {df1['speaker'].nunique()} | {df2['speaker'].nunique()} |")
    report.append(f"| SNR级别数 | {df1['snr'].nunique()} | {df2['snr'].nunique()} |")
    
    # 2. 整体指标对比
    report.append(f"\n---\n")
    report.append(f"## 2. 整体指标对比\n")
    report.append(f"| 指标 | {name1} | {name2} | 差异 | 胜出 |")
    report.append(f"|------|---------|---------|------|------|")
    
    for metric in metrics:
        avg1 = df1[metric].mean()
        avg2 = df2[metric].mean()
        diff = avg2 - avg1
        winner = name2 if diff > 0 else name1 if diff < 0 else "平局"
        symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="
        report.append(f"| {metric} | {avg1:.3f} | {avg2:.3f} | {diff:+.3f} {symbol} | **{winner}** |")
    
    # 3. 按噪声类型对比
    report.append(f"\n---\n")
    report.append(f"## 3. 按噪声类型对比 (SI-SNR)\n")
    report.append(f"| 噪声类型 | {name1} | {name2} | 差异 | 胜出 |")
    report.append(f"|----------|---------|---------|------|------|")
    
    for noise_type in noise_types:
        avg1 = df1[df1['noise_type'] == noise_type]['SI-SNR'].mean()
        avg2 = df2[df2['noise_type'] == noise_type]['SI-SNR'].mean()
        diff = avg2 - avg1
        winner = name2 if diff > 0 else name1 if diff < 0 else "平局"
        label = noise_labels.get(noise_type, noise_type)
        report.append(f"| {label} | {avg1:.2f} | {avg2:.2f} | {diff:+.2f} | **{winner}** |")
    
    # 4. 按SNR级别对比
    report.append(f"\n---\n")
    report.append(f"## 4. 按SNR级别对比 (SI-SNR)\n")
    report.append(f"| SNR | {name1} | {name2} | 差异 | 胜出 |")
    report.append(f"|-----|---------|---------|------|------|")
    
    for snr in snr_levels:
        avg1 = df1[df1['snr'] == snr]['SI-SNR'].mean()
        avg2 = df2[df2['snr'] == snr]['SI-SNR'].mean()
        diff = avg2 - avg1
        winner = name2 if diff > 0 else name1 if diff < 0 else "平局"
        report.append(f"| {snr:+d}dB | {avg1:.2f} | {avg2:.2f} | {diff:+.2f} | **{winner}** |")
    
    # 5. 详细数据表（按文件对比）
    report.append(f"\n---\n")
    report.append(f"## 5. 详细文件对比\n")
    report.append(f"以下表格显示同名文件的SI-SNR差异（{name2} - {name1}）：\n")
    
    # 合并两个数据集
    merged = pd.merge(df1[['filename', 'SI-SNR', 'PESQ', 'noise_type', 'snr']], 
                      df2[['filename', 'SI-SNR', 'PESQ']], 
                      on='filename', suffixes=('_1', '_2'))
    merged['SI-SNR_diff'] = merged['SI-SNR_2'] - merged['SI-SNR_1']
    merged['PESQ_diff'] = merged['PESQ_2'] - merged['PESQ_1']
    
    # 找出差异最大的10个文件
    report.append(f"### 5.1 差异最大的文件（{name2}显著优于{name1}）\n")
    top_positive = merged.nlargest(5, 'SI-SNR_diff')
    report.append(f"| 文件名 | {name1} SI-SNR | {name2} SI-SNR | 差异 |")
    report.append(f"|--------|---------------|---------------|------|")
    for _, row in top_positive.iterrows():
        report.append(f"| {row['filename'][:40]}... | {row['SI-SNR_1']:.2f} | {row['SI-SNR_2']:.2f} | {row['SI-SNR_diff']:+.2f} |")
    
    report.append(f"\n### 5.2 差异最大的文件（{name1}显著优于{name2}）\n")
    top_negative = merged.nsmallest(5, 'SI-SNR_diff')
    report.append(f"| 文件名 | {name1} SI-SNR | {name2} SI-SNR | 差异 |")
    report.append(f"|--------|---------------|---------------|------|")
    for _, row in top_negative.iterrows():
        report.append(f"| {row['filename'][:40]}... | {row['SI-SNR_1']:.2f} | {row['SI-SNR_2']:.2f} | {row['SI-SNR_diff']:+.2f} |")
    
    # 6. 结论
    report.append(f"\n---\n")
    report.append(f"## 6. 结论\n")
    
    # 统计胜出情况
    wins1 = sum(1 for m in metrics if df1[m].mean() > df2[m].mean())
    wins2 = sum(1 for m in metrics if df2[m].mean() > df1[m].mean())
    
    avg_diff = merged['SI-SNR_diff'].mean()
    
    report.append(f"### 整体表现\n")
    report.append(f"- **{name1}** 在 {wins1} 个指标上领先")
    report.append(f"- **{name2}** 在 {wins2} 个指标上领先")
    report.append(f"- SI-SNR 平均差异: {avg_diff:+.2f} dB")
    
    if avg_diff > 0.5:
        report.append(f"\n**结论**: {name2} 整体表现更优")
    elif avg_diff < -0.5:
        report.append(f"\n**结论**: {name1} 整体表现更优")
    else:
        report.append(f"\n**结论**: 两者表现相近")
    
    # 保存报告
    report_text = '\n'.join(report)
    report_path = os.path.join(output_dir, 'comparison_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"已保存报告: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description='对比两个降噪评估CSV文件')
    parser.add_argument('csv1', help='第一个CSV文件路径')
    parser.add_argument('csv2', help='第二个CSV文件路径')
    parser.add_argument('--name1', default='Method A', help='第一个数据集的名称 (默认: Method A)')
    parser.add_argument('--name2', default='Method B', help='第二个数据集的名称 (默认: Method B)')
    parser.add_argument('--output', default='./comparison_results', help='输出目录 (默认: ./comparison_results)')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    print(f"正在加载数据...")
    print(f"  - {args.name1}: {args.csv1}")
    print(f"  - {args.name2}: {args.csv2}")
    
    # 加载数据
    df1 = load_and_parse_csv(args.csv1)
    df2 = load_and_parse_csv(args.csv2)
    
    print(f"\n数据加载完成:")
    print(f"  - {args.name1}: {len(df1)} 条记录")
    print(f"  - {args.name2}: {len(df2)} 条记录")
    
    # 生成图表
    print(f"\n正在生成对比图表...")
    charts = create_comparison_charts(df1, df2, args.name1, args.name2, args.output)
    
    # 生成报告
    print(f"\n正在生成对比报告...")
    report = generate_report(df1, df2, args.name1, args.name2, args.output)
    
    print(f"\n===== 完成 =====")
    print(f"所有文件已保存到: {args.output}")
    print(f"生成的文件:")
    for chart in charts:
        print(f"  - {os.path.basename(chart)}")
    print(f"  - {os.path.basename(report)}")


if __name__ == '__main__':
    main()