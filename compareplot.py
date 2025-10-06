"""
Script to compare ALaST and traditional fine-tuning results.
Run this after training both models to generate comparison metrics and visualizations.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def load_metrics(file_path):
    """Load metrics from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def millions_formatter(x, pos):
    """Format numbers in millions for plot axes"""
    return f'{x / 1e6:.0f}M'


def billions_formatter(x, pos):
    """Format numbers in billions for plot axes"""
    return f'{x / 1e9:.1f}B'


def generate_comparison(traditional_metrics_path, alast_metrics_path, output_dir="comparison_results"):
    """
    Generate comparison charts and summary metrics

    Args:
        traditional_metrics_path: Path to traditional fine-tuning metrics JSON
        alast_metrics_path: Path to ALaST metrics JSON
        output_dir: Directory to save comparison results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load metrics
    traditional_metrics = load_metrics(traditional_metrics_path)
    alast_metrics = load_metrics(alast_metrics_path)

    # Summary statistics
    summary = {
        "accuracy": {
            "traditional": traditional_metrics["val_acc"][-1],
            "alast": alast_metrics["val_acc"][-1],
            "difference": alast_metrics["val_acc"][-1] - traditional_metrics["val_acc"][-1],
            "percent_change": (alast_metrics["val_acc"][-1] - traditional_metrics["val_acc"][-1]) /
                              traditional_metrics["val_acc"][-1] * 100
        },
        "training_time": {
            "traditional": sum(traditional_metrics["time_per_epoch"]),
            "alast": sum(alast_metrics["time_per_epoch"]),
            "speedup": sum(traditional_metrics["time_per_epoch"]) / sum(alast_metrics["time_per_epoch"])
        },
        "average_epoch_time": {
            "traditional": np.mean(traditional_metrics["time_per_epoch"]),
            "alast": np.mean(alast_metrics["time_per_epoch"]),
            "speedup": np.mean(traditional_metrics["time_per_epoch"]) / np.mean(alast_metrics["time_per_epoch"])
        }
    }

    # Add memory comparison if available
    if "peak_memory" in traditional_metrics and "peak_memory" in alast_metrics:
        if traditional_metrics["peak_memory"] and alast_metrics["peak_memory"]:
            summary["peak_memory"] = {
                "traditional": max(traditional_metrics["peak_memory"]),
                "alast": max(alast_metrics["peak_memory"]),
                "reduction": 1 - max(alast_metrics["peak_memory"]) / max(traditional_metrics["peak_memory"]),
                "percent_reduction": (1 - max(alast_metrics["peak_memory"]) / max(
                    traditional_metrics["peak_memory"])) * 100
            }

    # Add throughput comparison if available
    if "throughput" in traditional_metrics and "throughput" in alast_metrics:
        if traditional_metrics["throughput"] and alast_metrics["throughput"]:
            summary["throughput"] = {
                "traditional": np.mean(traditional_metrics["throughput"]),
                "alast": np.mean(alast_metrics["throughput"]),
                "speedup": np.mean(alast_metrics["throughput"]) / np.mean(traditional_metrics["throughput"])
            }

    # Add FLOPs comparison if available
    if "flops_per_epoch" in traditional_metrics and "flops_per_epoch" in alast_metrics:
        if traditional_metrics["flops_per_epoch"] and alast_metrics["flops_per_epoch"]:
            summary["flops"] = {
                "traditional": np.mean(traditional_metrics["flops_per_epoch"]),
                "alast": np.mean(alast_metrics["flops_per_epoch"]),
                "reduction": 1 - np.mean(alast_metrics["flops_per_epoch"]) / np.mean(
                    traditional_metrics["flops_per_epoch"]),
                "percent_reduction": (1 - np.mean(alast_metrics["flops_per_epoch"]) / np.mean(
                    traditional_metrics["flops_per_epoch"])) * 100
            }

    # Save summary to JSON
    with open(os.path.join(output_dir, "summary_comparison.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Save summary as text
    with open(os.path.join(output_dir, "summary_comparison.txt"), "w") as f:
        f.write("Comparison: Traditional vs ALaST Fine-tuning\n")
        f.write("=" * 50 + "\n\n")

        # Accuracy
        f.write("Validation Accuracy:\n")
        f.write(f"  Traditional: {summary['accuracy']['traditional']:.2f}%\n")
        f.write(f"  ALaST: {summary['accuracy']['alast']:.2f}%\n")
        f.write(f"  Difference: {summary['accuracy']['difference']:.2f}%\n")
        f.write(f"  Percent Change: {summary['accuracy']['percent_change']:.2f}%\n\n")

        # Training time
        f.write("Training Time:\n")
        f.write(f"  Traditional: {summary['training_time']['traditional']:.2f} seconds\n")
        f.write(f"  ALaST: {summary['training_time']['alast']:.2f} seconds\n")
        f.write(f"  Speedup: {summary['training_time']['speedup']:.2f}x\n\n")

        # Average epoch time
        f.write("Average Epoch Time:\n")
        f.write(f"  Traditional: {summary['average_epoch_time']['traditional']:.2f} seconds\n")
        f.write(f"  ALaST: {summary['average_epoch_time']['alast']:.2f} seconds\n")
        f.write(f"  Speedup: {summary['average_epoch_time']['speedup']:.2f}x\n\n")

        # Memory usage if available
        if "peak_memory" in summary:
            f.write("Peak Memory Usage:\n")
            f.write(f"  Traditional: {summary['peak_memory']['traditional']:.2f} GB\n")
            f.write(f"  ALaST: {summary['peak_memory']['alast']:.2f} GB\n")
            f.write(f"  Reduction: {summary['peak_memory']['percent_reduction']:.2f}%\n\n")

        # Throughput if available
        if "throughput" in summary:
            f.write("Throughput (samples/second):\n")
            f.write(f"  Traditional: {summary['throughput']['traditional']:.2f}\n")
            f.write(f"  ALaST: {summary['throughput']['alast']:.2f}\n")
            f.write(f"  Improvement: {summary['throughput']['speedup']:.2f}x\n\n")

        # FLOPs if available
        if "flops" in summary:
            f.write("Computational Cost (FLOPs per epoch):\n")
            f.write(f"  Traditional: {summary['flops']['traditional']:.2e}\n")
            f.write(f"  ALaST: {summary['flops']['alast']:.2e}\n")
            f.write(f"  Reduction: {summary['flops']['percent_reduction']:.2f}%\n\n")

        # Overall assessment
        f.write("Summary Assessment:\n")
        if summary['accuracy']['difference'] >= -0.5:  # Allow small accuracy drop
            efficiency_gain = max(
                summary['training_time']['speedup'] if 'training_time' in summary else 1,
                summary['peak_memory']['reduction'] * 100 if 'peak_memory' in summary else 1,
                summary['flops']['percent_reduction'] if 'flops' in summary else 1
            )
            if efficiency_gain > 20:  # 20% improvement
                assessment = "ALaST provides significant efficiency gains while maintaining accuracy."
            elif efficiency_gain > 10:  # 10% improvement
                assessment = "ALaST provides moderate efficiency gains while maintaining accuracy."
            else:
                assessment = "ALaST provides modest efficiency gains while maintaining accuracy."
        else:
            assessment = "ALaST trades some accuracy for efficiency gains."

        f.write(f"  {assessment}\n")

    # Create visualizations

    # 1. Validation Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(traditional_metrics["val_acc"], 'b-', marker='o', label='Traditional')
    plt.plot(alast_metrics["val_acc"], 'r-', marker='x', label='ALaST')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')

    # 2. Training Loss
    if "train_loss" in traditional_metrics and "train_loss" in alast_metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(traditional_metrics["train_loss"], 'b-', marker='o', label='Traditional')
        plt.plot(alast_metrics["train_loss"], 'r-', marker='x', label='ALaST')
        plt.title('Training Loss Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'loss_comparison.png'), dpi=300, bbox_inches='tight')

    # 3. Training Time per Epoch
    plt.figure(figsize=(10, 6))
    plt.bar(
        np.arange(len(traditional_metrics["time_per_epoch"])) - 0.2,
        traditional_metrics["time_per_epoch"],
        width=0.4,
        label='Traditional'
    )
    plt.bar(
        np.arange(len(alast_metrics["time_per_epoch"])) + 0.2,
        alast_metrics["time_per_epoch"],
        width=0.4,
        label='ALaST'
    )
    plt.title('Training Time per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.xticks(np.arange(len(traditional_metrics["time_per_epoch"])))
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'training_time_comparison.png'), dpi=300, bbox_inches='tight')

    # 4. Memory Usage if available
    if "peak_memory" in traditional_metrics and "peak_memory" in alast_metrics and traditional_metrics[
        "peak_memory"] and alast_metrics["peak_memory"]:
        plt.figure(figsize=(10, 6))
        plt.bar(
            np.arange(len(traditional_metrics["peak_memory"])) - 0.2,
            traditional_metrics["peak_memory"],
            width=0.4,
            label='Traditional'
        )
        plt.bar(
            np.arange(len(alast_metrics["peak_memory"])) + 0.2,
            alast_metrics["peak_memory"],
            width=0.4,
            label='ALaST'
        )
        plt.title('Peak Memory Usage per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Memory (GB)')
        plt.xticks(np.arange(len(traditional_metrics["peak_memory"])))
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'memory_comparison.png'), dpi=300, bbox_inches='tight')

    # 5. FLOPs comparison if available
    if "flops_per_epoch" in traditional_metrics and "flops_per_epoch" in alast_metrics and traditional_metrics[
        "flops_per_epoch"] and alast_metrics["flops_per_epoch"]:
        plt.figure(figsize=(10, 6))
        plt.bar(
            np.arange(len(traditional_metrics["flops_per_epoch"])) - 0.2,
            traditional_metrics["flops_per_epoch"],
            width=0.4,
            label='Traditional'
        )
        plt.bar(
            np.arange(len(alast_metrics["flops_per_epoch"])) + 0.2,
            alast_metrics["flops_per_epoch"],
            width=0.4,
            label='ALaST'
        )
        plt.title('Computational Cost per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('FLOPs')
        plt.xticks(np.arange(len(traditional_metrics["flops_per_epoch"])))
        plt.gca().yaxis.set_major_formatter(FuncFormatter(billions_formatter))
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'flops_comparison.png'), dpi=300, bbox_inches='tight')

    # 6. Layer budgets evolution (ALaST only)
    if "layer_budgets" in alast_metrics:
        plt.figure(figsize=(12, 6))
        layer_budgets = alast_metrics["layer_budgets"]

        # Convert to numpy for easier manipulation
        if isinstance(layer_budgets[0], list):
            num_layers = len(layer_budgets[0])
            for i in range(num_layers):
                # Extract budget for layer i across all epochs
                layer_data = [budget[i] for budget in layer_budgets]
                plt.plot(layer_data, marker='.', label=f'Layer {i + 1}')

        plt.title('ALaST Layer Budget Evolution')
        plt.xlabel('Epoch')
        plt.ylabel('Budget Value')
        plt.ylim(0, 1.05)  # Budget values are in [0,1]
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'layer_budgets_evolution.png'), dpi=300, bbox_inches='tight')

    # 7. Summary bar chart with only the metrics that exist in both
    plt.figure(figsize=(12, 6))
    metrics_to_plot = [
        ('Accuracy', summary['accuracy']['traditional'], summary['accuracy']['alast']),
        ('Training Time', summary['average_epoch_time']['traditional'], summary['average_epoch_time']['alast']),
    ]

    if "peak_memory" in summary:
        metrics_to_plot.append(
            ('Peak Memory', summary['peak_memory']['traditional'], summary['peak_memory']['alast'])
        )

    if "flops" in summary:
        # Scale down FLOPs to make it fit on the same chart
        flops_scale = 1e-9  # Convert to billions
        metrics_to_plot.append(
            ('FLOPs (billions)', summary['flops']['traditional'] * flops_scale, summary['flops']['alast'] * flops_scale)
        )

    labels = [m[0] for m in metrics_to_plot]
    traditional_values = [m[1] for m in metrics_to_plot]
    alast_values = [m[2] for m in metrics_to_plot]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width / 2, traditional_values, width, label='Traditional')
    rects2 = ax.bar(x + width / 2, alast_values, width, label='ALaST')

    ax.set_ylabel('Value')
    ax.set_title('Summary Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Add value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_bar_chart.png'), dpi=300, bbox_inches='tight')

    print(f"Comparison results saved to {output_dir}/")
    return summary


def main():
    """Main function to run the comparison"""
    # Check for result files
    traditional_metrics_path = None
    alast_metrics_path = None

    # Look in the results directory
    if os.path.exists('results'):
        for file in os.listdir('results'):
            if file.startswith('traditional') and file.endswith('metrics.json'):
                traditional_metrics_path = os.path.join('results', file)
            elif file.startswith('alast') and file.endswith('metrics.json'):
                alast_metrics_path = os.path.join('results', file)

    # If not found, ask for paths
    if not traditional_metrics_path:
        print("Traditional metrics file not found in results directory.")
        traditional_metrics_path = input("Enter path to traditional metrics JSON file: ")

    if not alast_metrics_path:
        print("ALaST metrics file not found in results directory.")
        alast_metrics_path = input("Enter path to ALaST metrics JSON file: ")

    # Generate comparison
    if os.path.exists(traditional_metrics_path) and os.path.exists(alast_metrics_path):
        print(f"Comparing metrics from:")
        print(f"  Traditional: {traditional_metrics_path}")
        print(f"  ALaST: {alast_metrics_path}")

        summary = generate_comparison(traditional_metrics_path, alast_metrics_path)

        # Print key findings
        print("\nKey Findings:")
        print(f"- Accuracy: {'Improved' if summary['accuracy']['difference'] > 0 else 'Decreased'} by {abs(summary['accuracy']['difference']):.2f}%")
        print(f"- Training Speed: {summary['average_epoch_time']['speedup']:.2f}x faster with ALaST")

        if "peak_memory" in summary:
            print(f"- Memory Usage: Reduced by {summary['peak_memory']['percent_reduction']:.2f}% with ALaST")

        if "flops" in summary:
            print(f"- Computational Cost: Reduced by {summary['flops']['percent_reduction']:.2f}% with ALaST")

        else:
            print("Error: Could not find metric files.")
        print("Please run both traditional fine-tuning and ALaST training first.")

if __name__ == "__main__":
    main()

