import optuna
import Training_Config as cfg
from Training_Training_Loop_Heatmap import train_heatmap
import json
import os

STUDY_NAME = "heatmap_posenet_sct"
STORAGE_URL = "sqlite:///optuna_heatmap_posenet_sct.db"
RESULTS_JSON = f"optuna_results{STUDY_NAME}.json"
N_TRIALS = 40

def objective(trial: optuna.Trial):
    print("\n" + "=" * 60)
    print(f"STARTING TRIAL {trial.number}")
    print("=" * 60)

    cfg.LR = trial.suggest_float(
        "lr", 1e-5, 1e-3, log=True
    )
    if trial.suggest_categorical("use_dropout", [True, False]):
        cfg.DROPOUT = trial.suggest_float("dropout", 0.05, 0.3)
    else:
        cfg.DROPOUT = 0.0
    cfg.COORD_LOSS_WEIGHT = trial.suggest_float(
        "coord_loss_weight", 0.1, 2.5, log=True
    )
    cfg.HEATMAP_WARMUP_EPOCHS = trial.suggest_int(
        "heatmap_warmup_epochs", 10, 100
    )
    cfg.HEATMAP_SIGMA = trial.suggest_float(
        "heatmap_sigma", 3.0, 9.0
    )

    print("🔍 Trial parameters:")
    for k, v in trial.params.items():
        print(f"  {k}: {v}")

    os.makedirs("weights", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    cfg.FINAL_WEIGHTS_PATH = f"weights/best_trial_{trial.number}.pt"
    cfg.CHECKPOINT_PATH = f"checkpoints/trial_{trial.number}.pt"

    try:
        metrics = train_heatmap(
            load_initial=False,
            load_final=False,
            resume_checkpoint=False,
            trial=trial
        )
    except optuna.exceptions.TrialPruned:
        print(f"TRIAL {trial.number} PRUNED")
        raise

    print(
        f"TRIAL {trial.number} FINISHED | "
        f"best val px = {metrics['best_val_px']:.4f}"
    )

    save_trial_result(trial, metrics)
    return metrics["best_val_px"]

def save_trial_result(trial, metrics):
    record = {
        "trial": trial.number,
        **metrics,
        **trial.params
    }

    data = []
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON, "r") as f:
            data = json.load(f)

    data.append(record)

    with open(RESULTS_JSON, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Trial {trial.number} saved to {RESULTS_JSON}")

if __name__ == "__main__":
    print("\n Initializing Optuna study...")

    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="minimize",
        storage=STORAGE_URL,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=10,
            interval_steps=5
        )
    )

    print(f"Study name: {STUDY_NAME}")
    print(f"Storage: {STORAGE_URL}")
    print(f"Existing trials: {len(study.trials)}")

    study.optimize(objective, n_trials=N_TRIALS)

    print("\nBEST RESULT")
    print(f"Best val px: {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
