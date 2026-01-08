#!/usr/bin/env python3
"""
多模型降噪效果对比脚本

使用方法:
    python compare_models.py model1_results.csv model2_results.csv [model3_results.csv ...]

示例:
    python compare_models.py gtcrn_output/evaluation_results.csv \
                            model_a_output/evaluation_results.csv \
                            model_b_output/evaluation_results.csv
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_results(csv_path, model_name=None):
    """加载评估结果"""
    if not Path(csv_path).exists():
        print(f"错误: 文件不存在: {csv_path}")
        return None

    df = pd.read_csv(csv_path)

    # 如果没有指定模型名，使用文件所在目录名
    if model_name is None:
        model_name = Path(csv_path).parent.name

    return df, model_name


def calculate_statistics(df, metrics):
    """计算统计信息"""
    stats = {}
    for metric in metrics:
        if metric in df.columns:
            values = pd.to_numeric(df[metric], errors='coerce').dropna()
            if len(values) > 0:
                stats[metric] = {
                    'mean': values.mean(),
                    'std': values.std(),
                    'min': values.min(),
                    'max': values.max(),
                    'median': values.median(),
                }
    return stats


def print_comparison_table(all_stats, model_names):
    """打印对比表格"""
    metrics = ["SI-SNR", "PESQ", "OVRL", "SIG", "BAK", "P808_MOS"]

    print("\n" + "=" * 100)
    print("模型对比结果（平均值）")
    print("=" * 100)

    # 表头
    header = f"{'指标':<12}"
    for name in model_names:
        header += f"{name:<20}"
    header += f"{'最佳模型':<20}"
    print(header)
    print("-" * 100)

    # 每个指标的对比
    for metric in metrics:
        row = f"{metric:<12}"
        values = []

        for stats in all_stats:
            if metric in stats:
                value = stats[metric]['mean']
                values.append(value)
                row += f"{value:<20.3f}"
            else:
                values.append(None)
                row += f"{'N/A':<20}"

        # 找出最佳值
        valid_values = [v for v in values if v is not None]
        if valid_values:
            best_idx = values.index(max(valid_values))
            row += f"{model_names[best_idx]:<20}"
        else:
            row += f"{'N/A':<20}"

        print(row)

    print("=" * 100)


def print_detailed_statistics(all_stats, model_names):
    """打印详细统计信息"""
    metrics = ["SI-SNR", "PESQ", "OVRL", "SIG", "BAK", "P808_MOS"]

    print("\n" + "=" * 100)
    print("详细统计信息")
    print("=" * 100)

    for i, (stats, name) in enumerate(zip(all_stats, model_names)):
        print(f"\n模型: {name}")
        print("-" * 100)
        print(f"{'指标':<12} {'平均值':<12} {'标准差':<12} {'最小值':<12} {'最大值':<12} {'中位数':<12}")
        print("-" * 100)

        for metric in metrics:
            if metric in stats:
                s = stats[metric]
                print(f"{metric:<12} {s['mean']:<12.3f} {s['std']:<12.3f} "
                      f"{s['min']:<12.3f} {s['max']:<12.3f} {s['median']:<12.3f}")
            else:
                print(f"{metric:<12} {'N/A':<12}")

    print("=" * 100)


def plot_comparison(all_stats, model_names, output_path="model_comparison.png"):
    """绘制对比图表"""
    metrics = ["SI-SNR", "PESQ", "OVRL", "SIG", "BAK", "P808_MOS"]

    # 准备数据
    data = {}
    for metric in metrics:
        data[metric] = []
        for stats in all_stats:
            if metric in stats:
                data[metric].append(stats[metric]['mean'])
            else:
                data[metric].append(0)

    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Model Comparison - Audio Quality Metrics', fontsize=16, fontweight='bold')

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        values = data[metric]

        # 绘制柱状图
        bars = ax.bar(model_names, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(model_names)])

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=9)

        ax.set_title(metric, fontweight='bold')
        ax.set_ylabel('Score')
        ax.grid(axis='y', alpha=0.3)

        # 旋转 x 轴标签
        ax.set_xticklabels(model_names, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n对比图表已保存: {output_path}")


def plot_radar_chart(all_stats, model_names, output_path="model_comparison_radar.png"):
    """绘制雷达图"""
    metrics = ["SI-SNR", "PESQ", "OVRL", "SIG", "BAK", "P808_MOS"]

    # 准备数据并归一化
    data = []
    for stats in all_stats:
        values = []
        for metric in metrics:
            if metric in stats:
                values.append(stats[metric]['mean'])
            else:
                values.append(0)
        data.append(values)

    # 归一化到 0-1 范围（用于雷达图）
    data_normalized = []
    for values in data:
        normalized = []
        for i, v in enumerate(values):
            if metrics[i] == "SI-SNR":
                # SI-SNR 通常在 -10 到 30 范围
                normalized.append((v + 10) / 40)
            else:
                # 其他指标通常在 1 到 5 范围
                normalized.append((v - 1) / 4)
        data_normalized.append(normalized)

    # 创建雷达图
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (values, name) in enumerate(zip(data_normalized, model_names)):
        values += values[:1]  # 闭合图形
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, size=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=10)
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_title('Model Comparison - Radar Chart', size=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"雷达图已保存: {output_path}")


def export_comparison_csv(all_stats, model_names, output_path="model_comparison.csv"):
    """导出对比结果为 CSV"""
    metrics = ["SI-SNR", "PESQ", "OVRL", "SIG", "BAK", "P808_MOS"]

    rows = []
    for metric in metrics:
        row = {"Metric": metric}
        for name, stats in zip(model_names, all_stats):
            if metric in stats:
                row[name] = f"{stats[metric]['mean']:.3f}"
            else:
                row[name] = "N/A"
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"对比结果已导出: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="多模型降噪效果对比工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 对比两个模型
  python compare_models.py gtcrn_output/evaluation_results.csv model_a_output/evaluation_results.csv

  # 对比多个模型并指定模型名称
  python compare_models.py gtcrn_output/evaluation_results.csv model_a_output/evaluation_results.csv \\
      --names "GTCRN-Micro" "Model A"

  # 对比并生成图表
  python compare_models.py gtcrn_output/evaluation_results.csv model_a_output/evaluation_results.csv \\
      --plot --output comparison_results/
        """
    )

    parser.add_argument(
        "csv_files",
        nargs="+",
        help="评估结果 CSV 文件路径（至少 2 个）"
    )
    parser.add_argument(
        "--names",
        nargs="+",
        default=None,
        help="模型名称（可选，默认使用目录名）"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="生成对比图表"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./",
        help="输出目录（默认: 当前目录）"
    )

    args = parser.parse_args()

    if len(args.csv_files) < 2:
        print("错误: 至少需要 2 个 CSV 文件进行对比")
        sys.exit(1)

    # 加载所有结果
    all_data = []
    model_names = []

    for i, csv_file in enumerate(args.csv_files):
        if args.names and i < len(args.names):
            model_name = args.names[i]
        else:
            model_name = None

        result = load_results(csv_file, model_name)
        if result is None:
            sys.exit(1)

        df, name = result
        all_data.append(df)
        model_names.append(name)

    # 计算统计信息
    metrics = ["SI-SNR", "PESQ", "OVRL", "SIG", "BAK", "P808_MOS"]
    all_stats = [calculate_statistics(df, metrics) for df in all_data]

    # 打印对比表格
    print_comparison_table(all_stats, model_names)
    print_detailed_statistics(all_stats, model_names)

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 导出 CSV
    export_comparison_csv(all_stats, model_names, output_dir / "model_comparison.csv")

    # 生成图表
    if args.plot:
        try:
            plot_comparison(all_stats, model_names, output_dir / "model_comparison_bar.png")
            plot_radar_chart(all_stats, model_names, output_dir / "model_comparison_radar.png")
        except Exception as e:
            print(f"警告: 生成图表失败: {e}")

    print("\n对比完成！")


if __name__ == "__main__":
    main()
