#!/usr/bin/env python3
"""
Visualize RAG experiment results with grouped bar charts.
Supports both single-dataset and multi-dataset result formats.
Results are aggregated by configuration and logged to Weights & Biases.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

# Try to import matplotlib and wandb
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Will skip local plotting.")

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not installed. Will skip wandb logging.")


def load_results(results_path: str) -> dict:
    """Load experiment results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def aggregate_by_config(results: dict) -> dict:
    """Aggregate metrics by configuration (similarity_threshold, max_chunks)."""
    config_metrics = defaultdict(lambda: {
        'vanilla_rag': defaultdict(list),
        'kg_rag': defaultdict(list)
    })
    
    # Check if this is a multi-dataset result
    if 'per_dataset_results' in results:
        # Multi-dataset format
        for dataset_name, dataset_data in results.get('per_dataset_results', {}).items():
            for result in dataset_data.get('results', []):
                config = result.get('config', {})
                threshold = config.get('similarity_threshold', 'unknown')
                max_chunks = config.get('max_chunks', 'unknown')
                
                config_key = f"thresh={threshold}_chunks={max_chunks}"
                
                # Vanilla RAG metrics
                if 'vanilla_rag_metrics' in result.get('evaluation', {}):
                    metrics = result['evaluation']['vanilla_rag_metrics']
                    for metric_name, value in metrics.items():
                        config_metrics[config_key]['vanilla_rag'][metric_name].append(value)
                
                # KG-RAG metrics
                if 'kg_rag_metrics' in result.get('evaluation', {}):
                    metrics = result['evaluation']['kg_rag_metrics']
                    for metric_name, value in metrics.items():
                        config_metrics[config_key]['kg_rag'][metric_name].append(value)
    else:
        # Single dataset format
        for result in results.get('results', []):
            config = result.get('config', {})
            threshold = config.get('similarity_threshold', 'unknown')
            max_chunks = config.get('max_chunks', 'unknown')
            
            config_key = f"thresh={threshold}_chunks={max_chunks}"
            
            # Vanilla RAG metrics
            if 'vanilla_rag_metrics' in result.get('evaluation', {}):
                metrics = result['evaluation']['vanilla_rag_metrics']
                for metric_name, value in metrics.items():
                    config_metrics[config_key]['vanilla_rag'][metric_name].append(value)
            
            # KG-RAG metrics
            if 'kg_rag_metrics' in result.get('evaluation', {}):
                metrics = result['evaluation']['kg_rag_metrics']
                for metric_name, value in metrics.items():
                    config_metrics[config_key]['kg_rag'][metric_name].append(value)
    
    # Compute averages
    aggregated = {}
    for config_key, systems in config_metrics.items():
        aggregated[config_key] = {}
        for system, metrics in systems.items():
            aggregated[config_key][system] = {
                metric: np.mean(values) if values else 0 
                for metric, values in metrics.items()
            }
    
    return aggregated


def aggregate_overall(results: dict) -> dict:
    """Aggregate overall metrics across all configurations."""
    overall = {
        'vanilla_rag': defaultdict(list),
        'kg_rag': defaultdict(list)
    }
    
    # Check if this is a multi-dataset result
    if 'per_dataset_results' in results:
        # Multi-dataset format - use combined_analysis if available
        combined = results.get('combined_analysis', {})
        if combined:
            overall_vanilla = combined.get('overall_vanilla_rag', {})
            overall_kg = combined.get('overall_kg_rag', {})
            return {
                'vanilla_rag': overall_vanilla,
                'kg_rag': overall_kg
            }
        
        # Otherwise aggregate manually
        for dataset_name, dataset_data in results.get('per_dataset_results', {}).items():
            for result in dataset_data.get('results', []):
                # Vanilla RAG metrics
                if 'vanilla_rag_metrics' in result.get('evaluation', {}):
                    metrics = result['evaluation']['vanilla_rag_metrics']
                    for metric_name, value in metrics.items():
                        overall['vanilla_rag'][metric_name].append(value)
                
                # KG-RAG metrics
                if 'kg_rag_metrics' in result.get('evaluation', {}):
                    metrics = result['evaluation']['kg_rag_metrics']
                    for metric_name, value in metrics.items():
                        overall['kg_rag'][metric_name].append(value)
    else:
        # Single dataset format
        for result in results.get('results', []):
            # Vanilla RAG metrics
            if 'vanilla_rag_metrics' in result.get('evaluation', {}):
                metrics = result['evaluation']['vanilla_rag_metrics']
                for metric_name, value in metrics.items():
                    overall['vanilla_rag'][metric_name].append(value)
            
            # KG-RAG metrics
            if 'kg_rag_metrics' in result.get('evaluation', {}):
                metrics = result['evaluation']['kg_rag_metrics']
                for metric_name, value in metrics.items():
                    overall['kg_rag'][metric_name].append(value)
    
    # Compute averages
    return {
        'vanilla_rag': {
            metric: np.mean(values) if values else 0 
            for metric, values in overall['vanilla_rag'].items()
        },
        'kg_rag': {
            metric: np.mean(values) if values else 0 
            for metric, values in overall['kg_rag'].items()
        }
    }


def print_per_dataset_summary(results: dict):
    """Print summary per dataset for multi-dataset results."""
    if 'per_dataset_results' not in results:
        return
    
    print("\n=== Per Dataset Summary ===")
    per_dataset = results.get('combined_analysis', {}).get('per_dataset', {})
    
    for dataset_name, analysis in per_dataset.items():
        print(f"\n{dataset_name}:")
        vanilla_stats = analysis.get('vanilla_rag_stats', {})
        kg_stats = analysis.get('kg_rag_stats', {})
        
        print(f"  Vanilla RAG:")
        for metric, value in vanilla_stats.items():
            print(f"    {metric}: {value:.2f}")
        
        print(f"  KG-RAG:")
        for metric, value in kg_stats.items():
            print(f"    {metric}: {value:.2f}")
        
        comparisons = analysis.get('comparisons', {})
        print(f"  Comparisons: vanilla_better={comparisons.get('vanilla_better', 0)}, "
              f"kg_better={comparisons.get('kg_better', 0)}, "
              f"tie={comparisons.get('tie', 0)}")


def create_comparison_chart(aggregated: dict, output_path: str = None):
    """Create grouped bar chart comparing Vanilla RAG vs KG-RAG by config."""
    if not HAS_MATPLOTLIB:
        print("Skipping chart creation (matplotlib not available)")
        return
    
    configs = sorted(aggregated.keys())
    metrics = ['correctness', 'completeness', 'relevance', 'coherence', 'factuality', 'hallucination_level']
    
    n_configs = len(configs)
    n_metrics = len(metrics)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    x = np.arange(n_configs)
    width = 0.35
    
    colors = {'vanilla_rag': '#1f77b4', 'kg_rag': '#ff7f0e'}
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        vanilla_values = [aggregated[c]['vanilla_rag'].get(metric, 0) for c in configs]
        kg_values = [aggregated[c]['kg_rag'].get(metric, 0) for c in configs]
        
        bars1 = ax.bar(x - width/2, vanilla_values, width, label='Vanilla RAG', color=colors['vanilla_rag'])
        bars2 = ax.bar(x + width/2, kg_values, width, label='KG-RAG', color=colors['kg_rag'])
        
        ax.set_xlabel('Configuration')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} by Configuration')
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace('_', '\n') for c in configs], rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 11)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved chart to {output_path}")
    
    return fig


def create_overall_comparison_chart(overall: dict, output_path: str = None):
    """Create overall comparison bar chart."""
    if not HAS_MATPLOTLIB:
        print("Skipping chart creation (matplotlib not available)")
        return
    
    metrics = ['correctness', 'completeness', 'relevance', 'coherence', 'factuality', 'hallucination_level']
    
    vanilla_values = [overall['vanilla_rag'].get(m, 0) for m in metrics]
    kg_values = [overall['kg_rag'].get(m, 0) for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, vanilla_values, width, label='Vanilla RAG', color='#1f77b4')
    bars2 = ax.bar(x + width/2, kg_values, width, label='KG-RAG', color='#ff7f0e')
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Score (1-10)')
    ax.set_title('Overall RAG Comparison: Vanilla RAG vs KG-RAG')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.legend()
    ax.set_ylim(0, 11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved chart to {output_path}")
    
    return fig


def log_to_wandb(results: dict, aggregated: dict, overall: dict, project_name: str = "kg-rag-evaluation"):
    """Log results to Weights & Biases."""
    if not HAS_WANDB:
        print("Skipping wandb logging (wandb not installed)")
        return
    
    # Initialize wandb
    run = wandb.init(project=project_name, job_type="evaluation")
    
    # Log overall metrics as a table
    metrics_table = wandb.Table(columns=["Metric", "Vanilla RAG", "KG-RAG", "Difference"])
    for metric in ['correctness', 'completeness', 'relevance', 'coherence', 'factuality', 'hallucination_level']:
        v = overall['vanilla_rag'].get(metric, 0)
        k = overall['kg_rag'].get(metric, 0)
        diff = v - k
        metrics_table.add_data(metric.replace('_', ' ').title(), f"{v:.2f}", f"{k:.2f}", f"{diff:+.2f}")
    
    wandb.log({"overall_metrics": metrics_table})
    
    # Log aggregated metrics by config
    for config, systems in aggregated.items():
        wandb.log({
            f"config/{config}/vanilla_rag_correctness": systems['vanilla_rag'].get('correctness', 0),
            f"config/{config}/kg_rag_correctness": systems['kg_rag'].get('correctness', 0),
            f"config/{config}/vanilla_rag_hallucination": systems['vanilla_rag'].get('hallucination_level', 0),
            f"config/{config}/kg_rag_hallucination": systems['kg_rag'].get('hallucination_level', 0),
        })
    
    # Log per-dataset metrics
    if 'per_dataset_results' in results:
        for dataset_name, dataset_data in results['per_dataset_results'].items():
            analysis = dataset_data.get('analysis', {})
            for metric, value in analysis.get('vanilla_rag_stats', {}).items():
                wandb.log({f"dataset/{dataset_name}/vanilla_rag_{metric}": value})
            for metric, value in analysis.get('kg_rag_stats', {}).items():
                wandb.log({f"dataset/{dataset_name}/kg_rag_{metric}": value})
    
    # Log comparison chart
    if HAS_MATPLOTLIB:
        fig = create_overall_comparison_chart(overall)
        wandb.log({"overall_comparison": wandb.Image(fig)})
        plt.close(fig)
        
        fig2 = create_comparison_chart(aggregated)
        wandb.log({"comparison_by_config": wandb.Image(fig2)})
        plt.close(fig2)
    
    # Log experiment metadata
    wandb.log({
        "experiment_id": results.get('experiment_id', 'unknown'),
        "total_experiments": results.get('combined_analysis', {}).get('total_experiments', 0) or results.get('analysis', {}).get('total_experiments', 0),
    })
    
    run.finish()
    print(f"Logged to wandb project: {project_name}")


def main():
    parser = argparse.ArgumentParser(description="Visualize RAG experiment results")
    parser.add_argument('--results', type=str, required=True, help='Path to results JSON file')
    parser.add_argument('--output-dir', type=str, default='results/visualizations', help='Output directory for charts')
    parser.add_argument('--project', type=str, default='kg-rag-evaluation', help='Wandb project name')
    parser.add_argument('--no-wandb', action='store_true', help='Skip wandb logging')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.results}...")
    results = load_results(args.results)
    
    # Check if multi-dataset
    is_multi_dataset = 'per_dataset_results' in results
    
    # Aggregate results
    print("Aggregating results by configuration...")
    aggregated = aggregate_by_config(results)
    overall = aggregate_overall(results)
    
    # Print summary
    print("\n=== Overall Results ===")
    print(f"Vanilla RAG: {overall['vanilla_rag']}")
    print(f"KG-RAG: {overall['kg_rag']}")
    
    # Print per-dataset summary if available
    if is_multi_dataset:
        print_per_dataset_summary(results)
    
    print("\n=== By Configuration ===")
    for config, systems in sorted(aggregated.items()):
        print(f"\n{config}:")
        print(f"  Vanilla RAG: correctness={systems['vanilla_rag'].get('correctness', 0):.2f}, hallucination={systems['vanilla_rag'].get('hallucination_level', 0):.2f}")
        print(f"  KG-RAG:      correctness={systems['kg_rag'].get('correctness', 0):.2f}, hallucination={systems['kg_rag'].get('hallucination_level', 0):.2f}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate charts
    if HAS_MATPLOTLIB:
        print("\nGenerating charts...")
        overall_chart_path = output_dir / "overall_comparison.png"
        config_chart_path = output_dir / "comparison_by_config.png"
        
        create_overall_comparison_chart(overall, str(overall_chart_path))
        create_comparison_chart(aggregated, str(config_chart_path))
    
    # Log to wandb
    if not args.no_wandb and HAS_WANDB:
        print("\nLogging to Weights & Biases...")
        log_to_wandb(results, aggregated, overall, args.project)
    elif args.no_wandb:
        print("\nSkipping wandb logging (--no-wandb flag)")
    else:
        print("\nSkipping wandb logging (wandb not installed)")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
