import os, glob, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
)
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve

warnings.filterwarnings("ignore")

# ─── Config ──────────────────────────────────────────────────────────────────
DATA_DIR      = "./data"
USE_SYNTHETIC = True
CORNERS       = ["Early Rise", "Early Fall", "Late Rise", "Late Fall"]
os.makedirs("per_model", exist_ok=True)

# =============================================================================
# DATA LOADING
# =============================================================================

def load_real_dataset(data_dir):
    try:
        import torch
        def _load_split(split):
            files = sorted(glob.glob(os.path.join(data_dir, split, "*.pt")))
            if not files:
                raise FileNotFoundError(f"No .pt files in {data_dir}/{split}/")
            X_all, y_all = [], []
            for f in files:
                g = torch.load(f)
                X_all.append(g.x.numpy())
                y_all.append(g.y.numpy())
            return np.vstack(X_all), np.vstack(y_all)
        print("Loading real dataset...")
        X_train, y_train = _load_split("train")
        X_test,  y_test  = _load_split("test")
        return X_train, y_train, X_test, y_test
    except Exception as e:
        print(f"Could not load real dataset ({e}). Falling back to synthetic.")
        return None


def make_synthetic_dataset(n_train=5000, n_test=1500, n_features=18, n_targets=4):
    """
    Synthetic data mimicking TimingPredict node feature distribution.
    Kept small (5k/1.5k) to avoid memory issues on low-RAM machines.
    """
    rng = np.random.RandomState(42)

    def gen(n):
        X = np.zeros((n, n_features))
        cell_type = rng.randint(0, 8, n)
        X[np.arange(n), cell_type] = 1
        X[:, 8:12]  = rng.exponential(0.1, (n, 4))
        X[:, 12:14] = rng.exponential(0.05, (n, 2))
        X[:, 14]    = rng.randint(1, 8, n)
        X[:, 15]    = rng.randint(0, 50, n)
        X[:, 16]    = rng.exponential(0.02, n)
        X[:, 17]    = (rng.rand(n) < 0.1).astype(float)
        base = (
            0.5
            - 3.0  * X[:, 8]
            - 1.5  * X[:, 12]
            - 0.02 * X[:, 15]
            + 0.1  * X[np.arange(n), cell_type % 4 + 4]
        )
        y = np.column_stack([
            base + rng.normal(0, 0.02, n),
            base * 0.95 + rng.normal(0, 0.02, n),
            base - 0.1  + rng.normal(0, 0.02, n),
            base * 0.90 + rng.normal(0, 0.02, n),
        ])
        return X, y

    print(f"Generating synthetic dataset ({n_train} train, {n_test} test pins)...")
    X_train, y_train = gen(n_train)
    X_test,  y_test  = gen(n_test)
    return X_train, y_train, X_test, y_test


# =============================================================================
# MODEL DEFINITIONS  (n_jobs=1 — avoids SIGKILL OOM)
# =============================================================================

def get_models():
    models = {

        "RandomForest [Breiman2001]": (
            MultiOutputRegressor(
                RandomForestRegressor(n_estimators=100, n_jobs=1, random_state=42),
                n_jobs=1
            ),
            "Breiman, ML 2001 | Guo et al. DAC 2022"
        ),

        "ExtraTrees [Geurts2006]": (
            MultiOutputRegressor(
                ExtraTreesRegressor(n_estimators=100, n_jobs=1, random_state=42),
                n_jobs=1
            ),
            "Geurts et al., Machine Learning 2006"
        ),

        "GradBoost [Friedman2001]": (
            MultiOutputRegressor(
                GradientBoostingRegressor(
                    n_estimators=100, learning_rate=0.05,
                    max_depth=4, random_state=42
                ),
                n_jobs=1
            ),
            "Friedman, Annals of Statistics 2001"
        ),

        "MLP [DAC2022-baseline]": (
            MLPRegressor(
                hidden_layer_sizes=(128, 128, 64),
                activation="relu",
                max_iter=300,
                learning_rate_init=1e-3,
                early_stopping=True,
                random_state=42
            ),
            "Guo et al., DAC 2022 (MLP flat baseline)"
        ),

        "Ridge [linear-baseline]": (
            MultiOutputRegressor(Ridge(alpha=1.0), n_jobs=1),
            "Linear baseline (Tikhonov regularization)"
        ),
    }

    try:
        from xgboost import XGBRegressor
        models["XGBoost [KDD2016]"] = (
            MultiOutputRegressor(
                XGBRegressor(
                    n_estimators=100, learning_rate=0.05,
                    max_depth=5, tree_method="hist",
                    n_jobs=1, random_state=42, verbosity=0
                ),
                n_jobs=1
            ),
            "Chen & Guestrin, KDD 2016"
        )
        print("XGBoost found — adding to model list.")
    except ImportError:
        print("XGBoost not installed (optional). Run: pip install xgboost")

    return models


# =============================================================================
# PER-MODEL GRAPH: Per-corner R² bar chart + Learning curve
# One PNG saved per model in ./per_model/
# =============================================================================

def plot_per_model(name, model, X_train, y_train, X_test, y_test, pred, paper):
    """
    Saves one PNG with two subplots:
      Left:  Per-corner R² bar chart
      Right: Learning curve (train vs val score vs training size)
    """
    short_name = name.split(" [")[0]
    corner_r2  = [r2_score(y_test[:, i], pred[:, i]) for i in range(4)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"{short_name}  —  Per-Corner R² & Learning Curve\nRef: {paper}",
        fontsize=11, fontweight="bold"
    )

    # ── Left: Per-corner R² bar chart ────────────────────────────────────────
    bar_colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]
    bars = axes[0].bar(CORNERS, corner_r2, color=bar_colors, edgecolor="white", linewidth=0.8)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("R² Score")
    axes[0].set_title("Per-Corner R² Score")
    axes[0].axhline(y=1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
    axes[0].tick_params(axis="x", labelsize=9)
    for bar, val in zip(bars, corner_r2):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    # ── Right: Learning curve ─────────────────────────────────────────────────
    # Use only corner 0 (early_rise) for learning curve to keep it fast
    train_sizes = np.linspace(0.1, 1.0, 6)

    # For MultiOutputRegressor wrap, use single-output model clone for speed
    from sklearn.base import clone
    from sklearn.multioutput import MultiOutputRegressor as MOR

    if isinstance(model, MOR):
        lc_model = clone(model.estimator)
        y_lc = y_train[:, 0]
        y_lc_label = f"{CORNERS[0]} slack"
    else:
        # MLP supports multi-output natively
        lc_model = clone(model)
        y_lc = y_train[:, 0]
        y_lc_label = f"{CORNERS[0]} slack"

    try:
        train_sizes_abs, train_scores, val_scores = learning_curve(
            lc_model, X_train, y_lc,
            train_sizes=train_sizes,
            cv=3,
            scoring="r2",
            n_jobs=1
        )
        train_mean = train_scores.mean(axis=1)
        train_std  = train_scores.std(axis=1)
        val_mean   = val_scores.mean(axis=1)
        val_std    = val_scores.std(axis=1)

        axes[1].plot(train_sizes_abs, train_mean, "o-", color="#3498db", label="Train R²")
        axes[1].fill_between(train_sizes_abs,
                             train_mean - train_std,
                             train_mean + train_std,
                             alpha=0.15, color="#3498db")
        axes[1].plot(train_sizes_abs, val_mean, "s-", color="#e74c3c", label="Val R²")
        axes[1].fill_between(train_sizes_abs,
                             val_mean - val_std,
                             val_mean + val_std,
                             alpha=0.15, color="#e74c3c")
        axes[1].set_xlabel("Training Samples")
        axes[1].set_ylabel("R² Score")
        axes[1].set_title(f"Learning Curve ({y_lc_label})")
        axes[1].legend(fontsize=9)
        axes[1].set_ylim(-0.1, 1.05)
        axes[1].grid(True, alpha=0.3)

    except Exception as e:
        axes[1].text(0.5, 0.5, f"Learning curve\nnot available\n({e})",
                     ha="center", va="center", transform=axes[1].transAxes, fontsize=10)
        axes[1].set_title("Learning Curve")

    plt.tight_layout()
    safe_name = short_name.replace(" ", "_").replace("/", "_")
    path = os.path.join("per_model", f"{safe_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# OVERALL COMPARISON GRAPH
# =============================================================================

def plot_overall(results):
    df = pd.DataFrame(results).sort_values("R²", ascending=False)
    names  = [n.split(" [")[0] for n in df["Model"]]
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e67e22", "#e74c3c", "#1abc9c"][:len(df)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Pre-Routing Slack Prediction — Overall Model Comparison\n(TimingPredict Dataset, DAC 2022)",
        fontsize=13, fontweight="bold"
    )

    # R²
    axes[0].barh(names, df["R²"], color=colors)
    axes[0].set_xlabel("R² Score (higher = better)")
    axes[0].set_title("R² Score by Model")
    axes[0].set_xlim(0, 1.1)
    axes[0].axvline(x=1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
    for i, v in enumerate(df["R²"]):
        axes[0].text(v + 0.01, i, f"{v:.4f}", va="center", fontsize=9)

    # MAE
    axes[1].barh(names, df["MAE"], color=colors)
    axes[1].set_xlabel("MAE (lower = better)")
    axes[1].set_title("MAE by Model")
    for i, v in enumerate(df["MAE"]):
        axes[1].text(v + 0.00005, i, f"{v:.5f}", va="center", fontsize=9)

    # Time
    axes[2].barh(names, df["Time(s)"], color=colors)
    axes[2].set_xlabel("Training Time (seconds)")
    axes[2].set_title("Training Time by Model")
    for i, v in enumerate(df["Time(s)"]):
        axes[2].text(v + 0.1, i, f"{v:.1f}s", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: model_comparison.png")


# =============================================================================
# TRAINING & EVALUATION
# =============================================================================

def run_all_models(X_train, y_train, X_test, y_test):
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    models  = get_models()
    results = []

    for name, (model, paper) in models.items():
        print(f"\n[{name}]")
        print(f"  ref: {paper}")
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        pred = model.predict(X_test)
        r2   = r2_score(y_test, pred, multioutput="uniform_average")
        mae  = mean_absolute_error(y_test, pred)

        print(f"  R²={r2:.4f}  MAE={mae:.5f}  Time={train_time:.1f}s")
        for i, c in enumerate(CORNERS):
            cr2 = r2_score(y_test[:, i], pred[:, i])
            print(f"    {c}: R²={cr2:.4f}")

        # Save per-model graph
        print(f"  Generating per-model graph...")
        plot_per_model(name, model, X_train, y_train, X_test, y_test, pred, paper)

        results.append({
            "Model":   name,
            "R²":      round(r2, 4),
            "MAE":     round(mae, 5),
            "Time(s)": round(train_time, 1),
            "Paper":   paper
        })

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    data = None
    if not USE_SYNTHETIC:
        data = load_real_dataset(DATA_DIR)

    if data is None:
        X_train, y_train, X_test, y_test = make_synthetic_dataset()
    else:
        X_train, y_train, X_test, y_test = data

    print(f"\nTrain: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"Targets (slack corners): {y_train.shape[1]}\n")

    results = run_all_models(X_train, y_train, X_test, y_test)

    df = pd.DataFrame(results).sort_values("R²", ascending=False)
    print("\n" + "=" * 70)
    print(f"{'MODEL':<35} {'R² (↑)':>8} {'MAE (↓)':>10} {'Time(s)':>8}")
    print("=" * 70)
    for _, row in df.iterrows():
        print(f"{row['Model']:<35} {row['R²']:>8.4f} {row['MAE']:>10.5f} {row['Time(s)']:>8.1f}")
    print("=" * 70)

    print("\nKEY REFERENCES")
    print("[1] Guo et al., DAC 2022     https://dl.acm.org/doi/abs/10.1145/3489517.3530597")
    print("[2] Breiman, ML 2001         Random Forests")
    print("[3] Friedman, Ann.Stat 2001  Gradient Boosting")
    print("[4] Chen & Guestrin KDD 2016 https://arxiv.org/abs/1603.02754")
    print("[5] Geurts et al., ML 2006   Extremely Randomized Trees")
    print("[6] Zhong et al., AAAI 2024  https://arxiv.org/abs/2403.00012")

    df.to_csv("results_summary.csv", index=False)
    print("Results saved to results_summary.csv")

    print("\nGenerating overall comparison graph...")
    plot_overall(results)

    print("\nAll done! Files produced:")
    print("  model_comparison.png        — overall comparison")
    print("  results_summary.csv         — full results table")
    for r in results:
        short = r["Model"].split(" [")[0].replace(" ", "_")
        print(f"  per_model/{short}.png")


if __name__ == "__main__":
    main()
