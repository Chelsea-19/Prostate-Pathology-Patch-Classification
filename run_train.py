import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import set_seed, get_device, ensure_dir, plot_loss_curve
from src.data import get_data_loaders
from src.model import create_model
from src.train import train_model, validate
from src.eval import evaluate_and_save
from src.explain import save_class_visualizations


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prostate Pathology Classification with UNI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data paths
    parser.add_argument(
        "--train_dirs",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--val_dirs",
        nargs="+",
        required=True,
    )
    
    # Hyperparameters setting
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    
    # Model options
    parser.add_argument(
        "--use_sampler",
        action="store_true",
        default=True,
        help="Use WeightedRandomSampler for class imbalance",
    )
    parser.add_argument(
        "--no_sampler",
        action="store_true",
        help="Disable WeightedRandomSampler",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        default=True,
        help="Freeze UNI backbone (only train head)",
    )
    parser.add_argument(
        "--unfreeze_backbone",
        action="store_true",
        help="Unfreeze UNI backbone (fine-tune all)",
    )
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    
    # Output
    parser.add_argument("--out_dir", type=str, default="outputs")
    
    # Ablation mode
    parser.add_argument(
        "--run_ablation",
        action="store_true",
        help="Run ablation experiments (sampler on/off, lr comparison)",
    )
    
    args = parser.parse_args()
    
    # Handle negation flags
    if args.no_sampler:
        args.use_sampler = False
    if args.unfreeze_backbone:
        args.freeze_backbone = False
    
    return args


def run_single_experiment(
    train_dirs: list,
    val_dirs: list,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    use_sampler: bool,
    freeze_backbone: bool,
    seed: int,
    num_workers: int,
    out_dir: str,
    device,
) -> dict:
    """
    Run a single training experiment.
    
    Returns:
        Dictionary with experiment configuration and results.
    """
    print("\n" + "="*60)
    print("Experiment Configuration:")
    print(f"  use_sampler={use_sampler}, lr={lr}, freeze_backbone={freeze_backbone}")
    print("="*60)
    
    # Set seed
    set_seed(seed)
    
    # Create data loaders
    train_loader, val_loader, train_ds, val_ds, classes = get_data_loaders(
        train_dirs=train_dirs,
        val_dirs=val_dirs,
        batch_size=batch_size,
        num_workers=num_workers,
        use_sampler=use_sampler,
        seed=seed,
    )
    
    # Create model
    model = create_model(
        num_classes=len(classes),
        freeze_backbone=freeze_backbone,
        dropout=dropout,
        device=device,
    )
    
    # Train
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
        out_dir=out_dir,
    )
    
    # Save loss curve
    plot_loss_curve(history, os.path.join(out_dir, "loss_curve.png"))
    
    # Evaluate
    metrics = evaluate_and_save(
        model=model,
        loader=val_loader,
        class_names=classes,
        device=device,
        out_dir=out_dir,
    )
    
    # Generate interpretability visualizations
    save_class_visualizations(
        model=model,
        loader=val_loader,
        class_names=classes,
        device=device,
        out_dir=out_dir,
    )
    
    return {
        "use_sampler": use_sampler,
        "lr": lr,
        "freeze_backbone": freeze_backbone,
        "epochs": epochs,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "best_val_f1": max(history["macro_f1"]),
    }


def run_ablation(args, device):
    """
    Run ablation experiments.
    
    Experiments:
    1. Baseline without WeightedRandomSampler
    2. With WeightedRandomSampler (default)
    3. lr=1e-4 vs lr=3e-4
    """
    print("\n" + "="*60)
    print("Running Ablation Experiments")
    print("="*60)
    
    results = []
    
    # Ablation 1: Sampler effect
    experiments = [
        {"name": "no_sampler", "use_sampler": False, "lr": args.lr},
        {"name": "with_sampler", "use_sampler": True, "lr": args.lr},
    ]
    
    # Ablation 2: Learning rate effect
    experiments.extend([
        {"name": "lr_1e-4", "use_sampler": True, "lr": 1e-4},
        {"name": "lr_3e-4", "use_sampler": True, "lr": 3e-4},
    ])
    
    for exp in experiments:
        exp_name = exp["name"]
        exp_dir = os.path.join(args.out_dir, exp_name)
        ensure_dir(exp_dir)
        
        print(f"\n>>> Running experiment: {exp_name}")
        
        result = run_single_experiment(
            train_dirs=args.train_dirs,
            val_dirs=args.val_dirs,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=exp["lr"],
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            use_sampler=exp["use_sampler"],
            freeze_backbone=args.freeze_backbone,
            seed=args.seed,
            num_workers=args.num_workers,
            out_dir=exp_dir,
            device=device,
        )
        result["experiment"] = exp_name
        results.append(result)
    
    # Save ablation summary
    df = pd.DataFrame(results)
    df = df.sort_values("macro_f1", ascending=False)
    summary_path = os.path.join(args.out_dir, "ablation_summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"\nSaved ablation summary to {summary_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("Ablation Results (sorted by macro_f1):")
    print("="*60)
    print(df.to_string(index=False))
    
    # Plot ablation comparison
    plt.figure(figsize=(10, 5))
    
    # Sampler comparison
    plt.subplot(1, 2, 1)
    sampler_results = df[df['experiment'].isin(['no_sampler', 'with_sampler'])]
    plt.bar(sampler_results['experiment'], sampler_results['macro_f1'])
    plt.title('WeightedRandomSampler Effect')
    plt.ylabel('Macro F1')
    plt.ylim(0.8, 1.0)
    
    # LR comparison
    plt.subplot(1, 2, 2)
    lr_results = df[df['experiment'].isin(['lr_1e-4', 'lr_3e-4'])]
    plt.bar(lr_results['experiment'], lr_results['macro_f1'])
    plt.title('Learning Rate Effect')
    plt.ylabel('Macro F1')
    plt.ylim(0.8, 1.0)
    
    plt.tight_layout()
    plot_path = os.path.join(args.out_dir, "ablation_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved ablation plot to {plot_path}")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup
    ensure_dir(args.out_dir)
    device = get_device()
    
    print("="*60)
    print("Prostate Pathology Patch Classification")
    print("="*60)
    print(f"Train dirs: {args.train_dirs}")
    print(f"Val dirs: {args.val_dirs}")
    print(f"Output dir: {args.out_dir}")
    
    if args.run_ablation:
        run_ablation(args, device)
    else:
        # Single experiment
        run_single_experiment(
            train_dirs=args.train_dirs,
            val_dirs=args.val_dirs,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            use_sampler=args.use_sampler,
            freeze_backbone=args.freeze_backbone,
            seed=args.seed,
            num_workers=args.num_workers,
            out_dir=args.out_dir,
            device=device,
        )
    
    print("\n" + "="*60)
    print("Done! Check outputs in:", args.out_dir)
    print("="*60)


if __name__ == "__main__":
    main()

