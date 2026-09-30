"""
Cloudburst model architecture for the Weather Nowcasting MVP.

Same TimeDistributed CNN + LSTM design as your thunderstorm model, sized
for your actual config: SEQUENCE_TIMESTEPS=24 (hourly data, not 6-hourly)
and a Tamil Nadu-wide grid (~23 x 17 at 0.25 deg: (13.5-8.0)/0.25+1=23
lat points, (80.5-76.5)/0.25+1=17 lon points). The channel count is 8
instead of your thunderstorm model's 6, for the two extra features from
src/features_cloudburst.py.

input_shape is a parameter, not hardcoded -- your training script should
pass X.shape[1:] from the actual saved array rather than relying on the
default here, since grid size will shift if REGION_NAME/LAT/LON bounds in
config.py ever change.

Differences vs. your thunderstorm model
----------------------------------------
1. 8 input channels instead of 6.
2. Binary focal loss by default instead of plain BCE. Cloudburst
   positives will be rarer than thunderstorm positives (top ~3% vs ~15%
   of windows), and at that level of imbalance on a modest dataset, focal
   loss tends to train more stably than class-weighted BCE alone.
   use_focal_loss=False in compile_cloudburst_model() switches back to
   plain BCE for a direct comparison against your thunderstorm setup.
3. LSTM dropout 0.3 instead of 0.2 -- mild extra regularization given how
   few positive cloudburst examples your dataset will contain.

Everything else (Conv2D filter counts, pooling, Dense head shape, sigmoid
output) matches your thunderstorm model.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models


def binary_focal_loss(gamma: float = 2.0, alpha: float = 0.75):
    """
    Standard binary focal loss. Down-weights easy (well-classified)
    examples and concentrates gradient on hard, rare positives.
    alpha=0.75 additionally up-weights the positive (cloudburst) class.
    """

    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        y_true = tf.cast(y_true, y_pred.dtype)

        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)

        focal_weight = alpha_t * tf.pow(1 - p_t, gamma)
        bce = -tf.math.log(p_t)

        return tf.reduce_mean(focal_weight * bce)

    return loss_fn


def build_cloudburst_model(
    input_shape: tuple = (24, 23, 17, 8),
    lstm_units: int = 64,
    lstm_dropout: float = 0.3,
    dense_units: int = 32,
    dense_dropout: float = 0.2,
) -> tf.keras.Model:
    """
    Build the cloudburst_mvp model.

    Parameters
    ----------
    input_shape : tuple
        (timesteps, lat, lon, channels). The default reflects your
        current config (24 timesteps, ~23x17 Tamil Nadu grid, 8 channels)
        but pass X.shape[1:] from your actual saved array in practice.
    """
    inputs = layers.Input(shape=input_shape, name="cloudburst_input")

    x = layers.TimeDistributed(
        layers.Conv2D(16, (3, 3), padding="same", activation="relu")
    )(inputs)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)

    x = layers.TimeDistributed(
        layers.Conv2D(32, (3, 3), padding="same", activation="relu")
    )(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)

    x = layers.LSTM(lstm_units, dropout=lstm_dropout, return_sequences=False)(x)

    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dense_dropout)(x)

    outputs = layers.Dense(1, activation="sigmoid", name="cloudburst_probability")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="cloudburst_mvp")
    return model


def compile_cloudburst_model(
    model: tf.keras.Model,
    learning_rate: float = 0.001,
    use_focal_loss: bool = True,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.75,
) -> tf.keras.Model:
    """Compile with focal loss by default (see module docstring)."""
    loss = (
        binary_focal_loss(gamma=focal_gamma, alpha=focal_alpha)
        if use_focal_loss
        else "binary_crossentropy"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model
