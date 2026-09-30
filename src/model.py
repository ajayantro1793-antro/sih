"""
MVP model: simple CNN + LSTM for single-hazard (thunderstorm) prediction.

This is intentionally simpler than the full multi-task U-Net design —
single output, fewer layers — so it trains fast and is easy to debug
on a small dataset. Scale up once this pipeline works end-to-end.

UPDATED: every layer now gets an explicit, optionally-prefixed name
instead of relying on Keras's automatic layer-naming. Reason: app.py
builds TWO models in the same Python process (Thunderstorm, then
Cloudburst) via @st.cache_resource, and Keras's auto-naming counter
isn't reliably scoped per-model across sequential builds in that
setup -- the second build's first TimeDistributed layer collided with
the name the first build already used ("time_distributed" used twice
in the model"). Explicit names sidestep that entirely, since they no
longer depend on any shared global counter. Pass name_prefix (e.g.
"thunderstorm", "cloudburst") when building more than one model in the
same process; omitting it preserves the exact names this file used
before, so single-model callers (02_train_model.py etc.) don't need
any changes.

Usage:
    from src.model import build_mvp_model
    model = build_mvp_model(input_shape=(24, 23, 17, 6))  # (timesteps, H, W, features)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def _prefix(name_prefix: str) -> str:
    return f"{name_prefix}_" if name_prefix else ""


def build_mvp_model(input_shape: tuple, name_prefix: str = "") -> tf.keras.Model:
    """
    Build a simplified spatiotemporal model for thunderstorm prediction.

    Architecture: TimeDistributed CNN -> LSTM -> Dense -> sigmoid output
    (This mirrors the full design's CNN+LSTM idea, just with 1 output
    instead of 3 task heads, and fewer filters/layers.)

    Args:
        input_shape: (timesteps, height, width, n_features)
                     e.g. (24, 23, 17, 6) = 6 timesteps, 50x50 grid, 6 features
        name_prefix: optional prefix for every layer name (e.g. "thunderstorm"),
                     so this can be safely called more than once in the same
                     process without layer-name collisions. Defaults to "" for
                     the original unprefixed names.

    Returns:
        Uncompiled Keras Model. Compile before training:
            model.compile(optimizer='adam', loss='binary_crossentropy',
                          metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    """
    p = _prefix(name_prefix)
    inputs = layers.Input(shape=input_shape, name=f"{p}input")

    x = layers.TimeDistributed(
        layers.Conv2D(16, (3, 3), padding="same", activation="relu", name=f"{p}conv1"),
        name=f"{p}td_conv1",
    )(inputs)
    x = layers.TimeDistributed(layers.BatchNormalization(name=f"{p}bn1"), name=f"{p}td_bn1")(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2), name=f"{p}pool1"), name=f"{p}td_pool1")(x)

    x = layers.TimeDistributed(
        layers.Conv2D(32, (3, 3), padding="same", activation="relu", name=f"{p}conv2"),
        name=f"{p}td_conv2",
    )(x)
    x = layers.TimeDistributed(layers.BatchNormalization(name=f"{p}bn2"), name=f"{p}td_bn2")(x)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D(name=f"{p}gap"), name=f"{p}td_gap")(x)
    # Shape now: (batch, timesteps, 32)

    # LSTM: learn temporal evolution across timesteps
    x = layers.LSTM(64, return_sequences=False, dropout=0.2, name=f"{p}lstm")(x)

    # Prediction head
    x = layers.Dense(32, activation="relu", name=f"{p}dense1")(x)
    x = layers.Dropout(0.2, name=f"{p}dropout1")(x)
    output = layers.Dense(1, activation="sigmoid", name=f"{p}thunderstorm_probability")(x)

    model_name = f"{name_prefix}_mvp" if name_prefix else "thunderstorm_mvp"
    model = models.Model(inputs=inputs, outputs=output, name=model_name)
    return model


def build_multi_district_model(input_shape: tuple, district_masks, name_prefix: str = "") -> tf.keras.Model:
    """
    Per-district version of build_mvp_model(). One shared CNN backbone
    scans the whole grid (as before), but instead of GlobalAveragePooling2D
    collapsing the ENTIRE grid into one vector (which would throw away
    exactly the spatial information needed to tell districts apart), each
    district gets its own masked average-pooled feature vector per
    timestep, computed from src.districts.build_district_masks(). A
    single shared LSTM (same weights) then processes each district's
    sequence independently, and a shared Dense head outputs one storm
    probability per district.

    Args:
        input_shape: (timesteps, height, width, n_features)
        district_masks: np.ndarray (n_districts, height, width), the
                         NORMALIZED masks from
                         src.districts.build_district_masks() -- must be
                         built from the SAME lat/lon grid as input_shape.
        name_prefix: optional prefix for every layer name (e.g. "thunderstorm",
                     "cloudburst"), so this can be safely called more than once
                     in the same process (e.g. app.py building both hazards'
                     models) without layer-name collisions. Defaults to "" for
                     the original unprefixed names.

    Returns:
        Uncompiled Keras Model with output shape (batch, n_districts).
        Compile with loss='binary_crossentropy' (applied elementwise
        across districts) or a custom weighted loss -- see
        notebooks/02_train_model.py's make_weighted_bce().
    """
    import numpy as np
    p = _prefix(name_prefix)
    district_masks = np.asarray(district_masks, dtype="float32")
    n_districts = district_masks.shape[0]
    masks_tensor = tf.constant(district_masks)  # (D, H, W)

    inputs = layers.Input(shape=input_shape, name=f"{p}input")  # (T, H, W, F)

    x = layers.TimeDistributed(
        layers.Conv2D(16, (3, 3), padding="same", activation="relu", name=f"{p}conv1"),
        name=f"{p}td_conv1",
    )(inputs)
    x = layers.TimeDistributed(layers.BatchNormalization(name=f"{p}bn1"), name=f"{p}td_bn1")(x)
    x = layers.TimeDistributed(
        layers.Conv2D(32, (3, 3), padding="same", activation="relu", name=f"{p}conv2"),
        name=f"{p}td_conv2",
    )(x)
    x = layers.TimeDistributed(layers.BatchNormalization(name=f"{p}bn2"), name=f"{p}td_bn2")(x)
    # x: (batch, T, H, W, C)

    def masked_pool(feat_map):
        # (batch,T,H,W,C) x (D,H,W) -> (batch,T,D,C): average-pool each
        # district's grid cells independently, per timestep.
        return tf.einsum("bthwc,dhw->btdc", feat_map, masks_tensor)

    x = layers.Lambda(masked_pool, name=f"{p}district_masked_pool")(x)  # (batch, T, D, C)
    x = layers.Permute((2, 1, 3), name=f"{p}permute_dt")(x)  # (batch, D, T, C)

    def to_batch_times_district(t):
        shp = t.shape  # static (batch, D, T, C) except batch
        return tf.reshape(t, (-1, shp[2], shp[3]))  # (batch*D, T, C)

    x = layers.Lambda(to_batch_times_district, name=f"{p}reshape_to_bd")(x)  # (batch*D, T, C)
    x = layers.LSTM(64, return_sequences=False, dropout=0.2, name=f"{p}lstm")(x)  # (batch*D, 64)

    def from_batch_times_district(t):
        return tf.reshape(t, (-1, n_districts, 64))  # (batch, D, 64)

    x = layers.Lambda(from_batch_times_district, name=f"{p}reshape_from_bd")(x)  # (batch, D, 64)

    # Dense layers apply to the last axis regardless of rank, so this is
    # a per-district head with SHARED weights across districts.
    x = layers.Dense(32, activation="relu", name=f"{p}dense1")(x)
    x = layers.Dropout(0.2, name=f"{p}dropout1")(x)
    per_district = layers.Dense(1, activation="sigmoid", name=f"{p}dense_out")(x)  # (batch, D, 1)
    outputs = layers.Reshape((n_districts,), name=f"{p}district_probabilities")(per_district)

    model_name = f"{name_prefix}_multi_district" if name_prefix else "thunderstorm_multi_district"
    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)
    return model


def get_callbacks(model_save_path: str = "models/best_mvp_model.h5"):
    """
    Standard training callbacks: early stopping + save best model + LR decay.

    IMPORTANT: save_weights_only=True. The multi-district model's masked
    pooling layer captures the district_masks array in a Python closure
    (see build_multi_district_model()'s Lambda layers), and Keras cannot
    JSON-serialize a captured TensorFlow tensor when saving a FULL model
    (architecture + weights) to .h5 -- that crashes mid-training with
    "Unable to serialize ... EagerTensor". Saving weights only sidesteps
    this entirely: there's no architecture to serialize, just numbers.
    Always reload with load_multi_district_model() below, which rebuilds
    the exact same architecture from district_masks.npy before loading
    these weights back in.
    """
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_save_path, monitor="val_loss", save_best_only=True, save_weights_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6
        ),
    ]


def load_multi_district_model(
    weights_path: str, input_shape: tuple, district_masks, name_prefix: str = ""
) -> tf.keras.Model:
    """
    Reconstruct a multi-district model and load previously-saved weights
    into it. Use this everywhere instead of tf.keras.models.load_model()
    for this model -- see the save_weights_only note in get_callbacks().

    Args:
        weights_path: path to the .h5 WEIGHTS file saved during training
        input_shape: (timesteps, height, width, n_features) -- must match
                     what the model was trained with
        district_masks: (n_districts, height, width) array -- must be the
                         SAME masks used during training (load
                         district_masks.npy, don't rebuild from scratch,
                         so weight shapes line up)
        name_prefix: optional prefix for every layer name -- pass a
                     hazard-specific value (e.g. "thunderstorm", "cloudburst")
                     when loading more than one model in the same process
                     (e.g. app.py), to avoid layer-name collisions. Weight
                     VALUES load correctly regardless of prefix, since
                     load_weights() matches by layer order/shape, not name,
                     for a freshly-built architecture like this.

    Returns:
        Model with trained weights loaded, ready for model.predict().
    """
    model = build_multi_district_model(input_shape=input_shape, district_masks=district_masks, name_prefix=name_prefix)
    model.load_weights(weights_path)
    return model


if __name__ == "__main__":
    # Sanity check: build the model with dummy shape and print summary
    model = build_mvp_model(input_shape=(24, 23, 17, 6))
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    model.summary()
