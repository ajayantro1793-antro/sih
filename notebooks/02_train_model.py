"""Train and evaluate the multi-district CNN+LSTM model with leakage-safe time splits.

Every district gets its own output unit, its own class-balance weight in
the loss, and its own tuned alert threshold -- a storm-prone coastal
district and a drier interior district should not share one cutoff.

UPDATED: normalize_from_train() now streams over batches to compute
mean/std and normalizes in place, instead of calling X_train.std(...)
directly. numpy's .std() internally allocates a second full-size array
((arr - mean) before squaring it) -- for your augmented (terrain-included)
X_train this briefly needed roughly double X_train's memory just for that
one line, which is what caused the ArrayMemoryError on the cloudburst
side before the same fix was applied there. Everything else in this file
is unchanged from your original.
"""

import json
import os
import random
import sys

import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (
    DATA_PROCESSED_DIR, MODELS_DIR, TRAIN_SPLIT, VAL_SPLIT,
    RANDOM_SEED, BATCH_SIZE, EPOCHS, LEARNING_RATE, SEQUENCE_TIMESTEPS,
    LEAD_TIME_HOURS,
)
from src.model import build_multi_district_model, get_callbacks


def set_all_seeds(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_processed_data():
    X = np.load(os.path.join(DATA_PROCESSED_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_PROCESSED_DIR, "y.npy"))  # (N, n_districts)
    feature_names = np.load(os.path.join(DATA_PROCESSED_DIR, "feature_names.npy"), allow_pickle=True)
    sample_times = np.load(os.path.join(DATA_PROCESSED_DIR, "sample_times.npy"), allow_pickle=True)
    district_names = np.load(os.path.join(DATA_PROCESSED_DIR, "district_names.npy"), allow_pickle=True)
    district_masks = np.load(os.path.join(DATA_PROCESSED_DIR, "district_masks.npy"))
    return X, y.astype("float32"), feature_names, sample_times, district_names, district_masks


def chronological_split_indices(
    n, y, train_split=TRAIN_SPLIT, val_split=VAL_SPLIT,
    purge=SEQUENCE_TIMESTEPS + LEAD_TIME_HOURS, min_positive_rate=0.02,
):
    """Same idea as the single-region version, but with multi-district y
    (N, D): a split is viable only if EVERY district clears
    min_positive_rate in EVERY split. This is a stricter bar with more
    districts, so min_positive_rate defaults slightly lower than the
    original single-region 0.03 -- raise it back up if you have a long
    enough record.
    """
    if n < 100:
        raise ValueError(f"Only {n} samples available. Download more data before final training/evaluation.")

    def build(train_split, val_split):
        train_end = int(n * train_split)
        val_end = int(n * (train_split + val_split))
        train_stop = max(0, train_end - purge)
        val_start = min(n, train_end + purge)
        val_stop = max(val_start, val_end - purge)
        test_start = min(n, val_end + purge)
        return (np.arange(0, train_stop), np.arange(val_start, val_stop), np.arange(test_start, n))

    def is_viable(idx_tuple):
        if min(len(i) for i in idx_tuple) == 0:
            return False
        return all(y[idx].mean(axis=0).min() >= min_positive_rate for idx in idx_tuple)

    candidate = build(train_split, val_split)
    if is_viable(candidate):
        return candidate

    print(f"NOTE: default split ({train_split:.0%}/{val_split:.0%}/"
          f"{1-train_split-val_split:.0%}) leaves some district below "
          f"{min_positive_rate:.0%} positive rate in some split -- "
          f"searching nearby split ratios...")

    train_options = sorted(
        [train_split + d for d in (0, -0.05, 0.05, -0.10, 0.10, -0.15, 0.15, -0.20, 0.20)],
        key=lambda t: abs(t - train_split),
    )
    val_options = sorted(
        [val_split + d for d in (0, -0.03, 0.03, -0.06, 0.06)],
        key=lambda v: abs(v - val_split),
    )

    for t in train_options:
        if not (0.4 <= t <= 0.85):
            continue
        for v in val_options:
            if not (0.05 <= v <= 0.30) or t + v >= 0.97:
                continue
            candidate = build(t, v)
            if is_viable(candidate):
                print(f"Found viable split: train={t:.0%}, val={v:.0%}, test={1-t-v:.0%}")
                return candidate

    raise ValueError(
        "No split ratio in the search grid gives EVERY district a positive "
        f"rate >= {min_positive_rate:.0%} in every split. Download more "
        "months/years spanning multiple monsoon cycles, drop the driest "
        "district(s) from src/districts.py, or lower min_positive_rate."
    )


def normalize_from_train(X_train, X_val, X_test, batch_size=500):
    """
    Fit per-feature normalization on TRAIN only and apply it everywhere.

    Memory note: numpy's array.std() internally allocates a SECOND
    full-size array (arr - mean) before squaring it -- for a train split
    this size, that briefly needs roughly double X_train's memory just for
    this one line, which can exceed available RAM (this is what caused
    the ArrayMemoryError seen on the cloudburst side of this project).
    This version instead streams over small batches to accumulate sum and
    sum-of-squares (mean/var computed from those at the end, no full-size
    temp array), then normalizes each array IN PLACE, batch by batch,
    instead of allocating a second full-size copy the way
    `(X - mean) / std` would.
    """
    n_features = X_train.shape[-1]
    n_samples = X_train.shape[0]

    count = 0
    sum_ = np.zeros(n_features, dtype=np.float64)
    sum_sq = np.zeros(n_features, dtype=np.float64)

    for start in range(0, n_samples, batch_size):
        batch = X_train[start:start + batch_size].astype(np.float64)
        count += batch.shape[0] * batch.shape[1] * batch.shape[2] * batch.shape[3]
        sum_ += batch.sum(axis=(0, 1, 2, 3))
        sum_sq += (batch ** 2).sum(axis=(0, 1, 2, 3))

    mean_flat = sum_ / count
    var_flat = np.maximum(sum_sq / count - mean_flat ** 2, 0.0)  # guard tiny negative from fp error
    std_flat = np.sqrt(var_flat)
    std_flat = np.where(std_flat < 1e-8, 1.0, std_flat)

    mean = mean_flat.astype("float32").reshape(1, 1, 1, 1, n_features)
    std = std_flat.astype("float32").reshape(1, 1, 1, 1, n_features)

    def normalize_in_place(X):
        for start in range(0, X.shape[0], batch_size):
            end = start + batch_size
            X[start:end] = (X[start:end] - mean) / std
        return X

    X_train = normalize_in_place(X_train)
    X_val = normalize_in_place(X_val)
    X_test = normalize_in_place(X_test)

    return X_train, X_val, X_test, mean, std


def compute_district_pos_weights(y_train, cap=20.0):
    """
    Per-district positive-class weight (n_negative / n_positive), the
    multi-label equivalent of sklearn's 'balanced' class_weight -- capped
    so a near-empty district can't blow up the loss.
    """
    n_pos = y_train.sum(axis=0)
    n_neg = (1 - y_train).sum(axis=0)
    weights = np.where(n_pos > 0, n_neg / np.maximum(n_pos, 1), cap)
    weights = np.clip(weights, 1.0, cap)
    return weights.astype("float32")


def make_weighted_bce(pos_weight_per_district):
    pos_weight = tf.constant(pos_weight_per_district, dtype=tf.float32)  # (D,)

    def weighted_bce(y_true, y_pred):
        eps = 1e-7
        y_pred = tf.clip_by_value(y_pred, eps, 1 - eps)
        loss = -(pos_weight * y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        return tf.reduce_mean(loss)

    return weighted_bce


def find_best_threshold_per_district(model, X_val, y_val, target_recall=0.70):
    """Same F1/recall-target threshold logic as the single-region version,
    run once per district column."""
    from sklearn.metrics import precision_recall_curve

    proba_all = model.predict(X_val, verbose=0)  # (N, D)
    n_districts = y_val.shape[1]
    thresholds = []

    for d in range(n_districts):
        y_col = y_val[:, d]
        proba = proba_all[:, d]
        if len(np.unique(y_col)) < 2:
            thresholds.append(0.5)
            continue
        precision, recall, thr = precision_recall_curve(y_col, proba)
        candidates = np.where(recall[:-1] >= target_recall)[0]
        if len(candidates):
            idx = candidates[np.argmax(precision[:-1][candidates])]
            thresholds.append(float(thr[idx]))
        else:
            idx = int(np.argmax(recall[:-1])) if len(thr) else 0
            thresholds.append(float(thr[idx]) if len(thr) else 0.5)

    return np.array(thresholds, dtype="float32")


def evaluate_model_per_district(model, X_test, y_test, thresholds, district_names):
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, confusion_matrix,
        brier_score_loss,
    )

    proba_all = model.predict(X_test, verbose=0)  # (N, D)
    per_district_metrics = {}

    print("\n" + "=" * 90)
    print("LEAKAGE-SAFE TEST EVALUATION -- PER DISTRICT")
    print("=" * 90)
    header = f"{'District':16s}{'Thr':>6s}{'Acc':>7s}{'Prec':>7s}{'Rec':>7s}{'F1':>7s}{'AUC':>7s}{'TP':>5s}{'FP':>5s}{'FN':>5s}"
    print(header)

    for d, name in enumerate(district_names):
        proba = proba_all[:, d]
        y_col = y_test[:, d]
        thr = float(thresholds[d])
        pred = (proba >= thr).astype(int)

        metrics = {
            "threshold": thr,
            "accuracy": float(accuracy_score(y_col, pred)),
            "precision": float(precision_score(y_col, pred, zero_division=0)),
            "recall": float(recall_score(y_col, pred, zero_division=0)),
            "f1": float(f1_score(y_col, pred, zero_division=0)),
            "brier_score": float(brier_score_loss(y_col, proba)),
        }
        if len(np.unique(y_col)) == 2:
            metrics["auc_roc"] = float(roc_auc_score(y_col, proba))
            metrics["auc_pr"] = float(average_precision_score(y_col, proba))
        else:
            metrics["auc_roc"] = None
            metrics["auc_pr"] = None

        cm = confusion_matrix(y_col, pred, labels=[0, 1])
        metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"] = [int(x) for x in cm.ravel()]
        per_district_metrics[str(name)] = metrics

        auc_str = f"{metrics['auc_roc']:.3f}" if metrics["auc_roc"] is not None else "  n/a"
        print(f"{str(name):16s}{thr:6.2f}{metrics['accuracy']:7.2f}{metrics['precision']:7.2f}"
              f"{metrics['recall']:7.2f}{metrics['f1']:7.2f}{auc_str:>7s}"
              f"{metrics['tp']:5d}{metrics['fp']:5d}{metrics['fn']:5d}")

    print("=" * 90)
    return per_district_metrics


def main():
    set_all_seeds()
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

    X, y, feature_names, sample_times, district_names, district_masks = load_processed_data()
    print(f"X shape: {X.shape}, y shape: {y.shape} (samples, n_districts={len(district_names)})")
    print(f"Features: {list(feature_names)}")
    print(f"Districts: {list(district_names)}")

    train_idx, val_idx, test_idx = chronological_split_indices(len(X), y)
    np.save(os.path.join(DATA_PROCESSED_DIR, "train_indices.npy"), train_idx)
    np.save(os.path.join(DATA_PROCESSED_DIR, "val_indices.npy"), val_idx)
    np.save(os.path.join(DATA_PROCESSED_DIR, "test_indices.npy"), test_idx)

    print("\nChronological split with purge gap:")
    for name, idx in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
        print(f"{name:5s}: {len(idx):5d} | {sample_times[idx[0]]} -> {sample_times[idx[-1]]}")
        for d, dname in enumerate(district_names):
            print(f"       {dname:16s} positive={y[idx, d].mean():.2%}")

    X_train, X_val, X_test, mean, std = normalize_from_train(X[train_idx], X[val_idx], X[test_idx])
    np.save(os.path.join(DATA_PROCESSED_DIR, "feature_mean.npy"), mean)
    np.save(os.path.join(DATA_PROCESSED_DIR, "feature_std.npy"), std)

    model = build_multi_district_model(input_shape=X.shape[1:], district_masks=district_masks)
    pos_weights = compute_district_pos_weights(y[train_idx])
    print(f"\nPer-district positive-class weights: "
          f"{dict(zip([str(n) for n in district_names], pos_weights.tolist()))}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=make_weighted_bce(pos_weights),
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    model_path = os.path.join(MODELS_DIR, "thunderstorm_mvp.h5")
    history = model.fit(
        X_train, y[train_idx],
        validation_data=(X_val, y[val_idx]),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks(model_path),
        verbose=1,
    )
    # ModelCheckpoint already wrote the best-epoch weights to model_path
    # during training; explicit save here just guarantees the file matches
    # exactly what's in memory (best weights, via EarlyStopping's
    # restore_best_weights=True) even if checkpointing never fired.
    model.save_weights(model_path)

    thresholds = find_best_threshold_per_district(model, X_val, y[val_idx])
    np.save(os.path.join(DATA_PROCESSED_DIR, "alert_threshold.npy"), thresholds)
    print(f"\nPer-district tuned thresholds: {dict(zip([str(n) for n in district_names], thresholds.tolist()))}")

    per_district_metrics = evaluate_model_per_district(model, X_test, y[test_idx], thresholds, district_names)

    metadata = {
        "features": [str(x) for x in feature_names],
        "districts": [str(x) for x in district_names],
        "sequence_timesteps": SEQUENCE_TIMESTEPS,
        "lead_time_hours": LEAD_TIME_HOURS,
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "test_samples": int(len(test_idx)),
        "train_start": str(sample_times[train_idx[0]]),
        "train_end": str(sample_times[train_idx[-1]]),
        "val_start": str(sample_times[val_idx[0]]),
        "val_end": str(sample_times[val_idx[-1]]),
        "test_start": str(sample_times[test_idx[0]]),
        "test_end": str(sample_times[test_idx[-1]]),
        "normalization_fit": "training split only",
        "label_type": "per-district area-mean heavy-rainfall proxy",
        "per_district_metrics": per_district_metrics,
    }
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    history_json = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(os.path.join(MODELS_DIR, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history_json, f, indent=2)

    print(f"\nSaved model weights: {model_path}")
    print("(Full architecture is NOT saved to this file -- it's weights-only, see")
    print(" src/model.py's get_callbacks() docstring. Reload with")
    print(" src/model.load_multi_district_model(), which rebuilds the architecture")
    print(" from district_masks.npy before loading these weights.)")
    print("Saved per-district thresholds, train/val/test indices, normalization stats, metadata, history.")


if __name__ == "__main__":
    main()
