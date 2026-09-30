"""
app.py — Streamlit demo dashboard for the PER-DISTRICT nowcasting MVP,
covering BOTH hazards: thunderstorm and cloudburst.

Run with:
    streamlit run app.py

Lets you pick a hazard (top-level tabs), then a date from that hazard's
test period to see the model's predicted probability for EVERY Tamil
Nadu district (rectangular approximations -- see src/districts.py), plus
a live "right now" tab per hazard.

Both hazards share the exact same rendering code (render_historical_tab,
render_live_tab) -- only the data/model/threshold/log SOURCE differs,
via the HAZARDS config dict below. This keeps the two dashboards from
drifting out of sync the way two copy-pasted files would.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from configs.config import DATA_PROCESSED_DIR, MODELS_DIR
from src.model import load_multi_district_model

from src.thresholds import get_alert_thresholds, get_district_names
from src.live_predict_core import run_live_prediction
from src.live_log import (
    append_log as append_log_thunderstorm,
    load_log as load_log_thunderstorm,
    save_log as save_log_thunderstorm,
)

from src.thresholds_cloudburst import get_cloudburst_alert_thresholds, get_cloudburst_district_names
from src.live_predict_core_cloudburst import run_live_cloudburst_prediction
from src.live_log_cloudburst import (
    append_log as append_log_cloudburst,
    load_log as load_log_cloudburst,
    save_log as save_log_cloudburst,
)

st.set_page_config(page_title="Weather Nowcasting MVP — Per District", layout="centered")


# --- Per-hazard configuration -------------------------------------------
# Everything that differs between thunderstorm and cloudburst lives here.
# render_historical_tab() / render_live_tab() below are hazard-agnostic --
# they just read whatever this dict points them at.
HAZARDS = {
    "Thunderstorm": {
        "title": "Thunderstorm",
        "tab_label": "🌩️ Thunderstorm",
        "header_emoji": "🌩️",
        "model_filename": "thunderstorm_mvp.h5",
        "metadata_filename": "model_metadata.json",
        "prefix": "",  # X.npy, y.npy, etc. -- unprefixed
        "get_district_names": get_district_names,
        "get_thresholds": get_alert_thresholds,
        "run_live_prediction": run_live_prediction,
        "append_log": append_log_thunderstorm,
        "load_log": load_log_thunderstorm,
        "save_log": save_log_thunderstorm,
        "event_true_label": "storm",
        "event_false_label": "no_storm",
        "event_true_display": "🌩️ Storm",
        "event_false_display": "☀️ No storm",
        "alert_word": "storm",
        "live_notebook": "notebooks/03_live_predict.py",
        "caveat": (
            "Uses Open-Meteo operational forecast data -- the same source this "
            "model was trained on -- plus rectangular district-box approximations. "
            "Proof of concept only."
        ),
    },
    "Cloudburst": {
        "title": "Cloudburst",
        "tab_label": "🌧️ Cloudburst",
        "header_emoji": "🌧️",
        "model_filename": "cloudburst_mvp.h5",
        "metadata_filename": "cloudburst_model_metadata.json",
        "prefix": "cloudburst_",  # cloudburst_X.npy, cloudburst_y.npy, etc.
        "get_district_names": get_cloudburst_district_names,
        "get_thresholds": get_cloudburst_alert_thresholds,
        "run_live_prediction": run_live_cloudburst_prediction,
        "append_log": append_log_cloudburst,
        "load_log": load_log_cloudburst,
        "save_log": save_log_cloudburst,
        "event_true_label": "cloudburst",
        "event_false_label": "no_cloudburst",
        "event_true_display": "🌧️ Cloudburst",
        "event_false_display": "🌤️ No cloudburst",
        "alert_word": "cloudburst",
        "live_notebook": "notebooks/03b_live_predict_cloudburst.py",
        "caveat": (
            "Uses Open-Meteo operational forecast data -- the same source this "
            "model was trained on -- plus rectangular district-box approximations. "
            "Cloudburst labels are an extreme-rainfall PROXY (top ~3% of forward "
            "rainfall windows per district), not a confirmed cloudburst event "
            "record -- true cloudbursts are sub-grid-scale, sub-hourly events this "
            "hourly/0.25-degree pipeline cannot resolve directly. Proof of concept only."
        ),
    },
}


@st.cache_resource
def load_model(input_shape, _district_masks, hazard: str):
    """
    Rebuild the multi-district architecture and load its trained weights
    for the given hazard. Both thunderstorm_mvp.h5 and cloudburst_mvp.h5
    are WEIGHTS-ONLY (see src/model.py's get_callbacks() docstring), so
    tf.keras.models.load_model() won't work -- the architecture has to be
    reconstructed from that hazard's district_masks first.

    _district_masks is prefixed with an underscore so Streamlit's
    @st.cache_resource doesn't try to hash a large numpy array as a cache
    key (st.cache_resource skips leading-underscore args); input_shape and
    hazard are small/hashable and fine to use as cache keys, so switching
    hazards or retraining with a different shape correctly busts the cache.
    """
    cfg = HAZARDS[hazard]
    model_path = os.path.join(MODELS_DIR, cfg["model_filename"])
    return load_multi_district_model(model_path, input_shape, _district_masks, name_prefix=hazard.lower())


@st.cache_data
def load_data(hazard: str):
    """
    Load the RAW feature array plus the ACTUAL test split indices,
    normalization statistics, and district names/thresholds saved by
    that hazard's dataset-build + training scripts.
    """
    cfg = HAZARDS[hazard]
    prefix = cfg["prefix"]

    X = np.load(os.path.join(DATA_PROCESSED_DIR, f"{prefix}X.npy"))
    y = np.load(os.path.join(DATA_PROCESSED_DIR, f"{prefix}y.npy"))  # (N, n_districts)
    sample_times = np.load(os.path.join(DATA_PROCESSED_DIR, f"{prefix}sample_times.npy"), allow_pickle=True)
    test_idx = np.load(os.path.join(DATA_PROCESSED_DIR, f"{prefix}test_indices.npy"))

    mean = np.load(os.path.join(DATA_PROCESSED_DIR, f"{prefix}feature_mean.npy"))
    std = np.load(os.path.join(DATA_PROCESSED_DIR, f"{prefix}feature_std.npy"))

    X_test_raw = X[test_idx]
    X_test_norm = ((X_test_raw - mean) / std).astype("float32")
    y_test = y[test_idx]
    times_test = sample_times[test_idx]

    district_names = [str(n) for n in cfg["get_district_names"]()]
    thresholds = cfg["get_thresholds"]()
    district_masks = np.load(os.path.join(DATA_PROCESSED_DIR, f"{prefix}district_masks.npy"))

    return X_test_norm, y_test, times_test, district_names, thresholds, district_masks


def render_historical_tab(model, X_test, y_test, times_test, district_names, thresholds, cfg, key_prefix):
    st.subheader("Pick a date to see every district's prediction")

    dates_display = [pd.Timestamp(t).strftime("%Y-%m-%d %H:%M") for t in times_test]
    idx_key = f"selected_idx_{key_prefix}"

    if idx_key not in st.session_state:
        st.session_state[idx_key] = 0

    all_probs = model.predict(X_test, verbose=0)  # (N, D)
    any_alert = (all_probs >= thresholds[np.newaxis, :]).any(axis=1)
    any_actual_event = (y_test > 0.5).any(axis=1)
    correct_alert_idxs = np.where(any_alert & any_actual_event)[0]
    correct_quiet_idxs = np.where(~any_alert & ~any_actual_event)[0]

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if len(correct_alert_idxs) > 0 and st.button(
            f"{cfg['header_emoji']} Show a date with a correct alert", key=f"alert_btn_{key_prefix}"
        ):
            st.session_state[idx_key] = int(correct_alert_idxs[0])
    with col_b:
        if len(correct_quiet_idxs) > 0 and st.button(
            "☀️ Show a correctly quiet date", key=f"quiet_btn_{key_prefix}"
        ):
            st.session_state[idx_key] = int(correct_quiet_idxs[0])
    with col_c:
        if st.button("🔀 Random date", key=f"random_btn_{key_prefix}"):
            st.session_state[idx_key] = int(np.random.randint(len(X_test)))

    selected_idx = st.selectbox(
        "Or pick a specific test-period timestamp (unseen by the model during training):",
        options=range(len(dates_display)),
        format_func=lambda i: dates_display[i],
        index=st.session_state[idx_key],
        key=f"date_selectbox_{key_prefix}",
    )
    st.session_state[idx_key] = selected_idx

    sample_X = X_test[selected_idx: selected_idx + 1]
    predicted_probs = model.predict(sample_X, verbose=0)[0]  # (D,)
    actual_labels = y_test[selected_idx]  # (D,)

    rows = []
    for name, prob, thr, actual in zip(district_names, predicted_probs, thresholds, actual_labels):
        predicted_event = prob >= thr
        if predicted_event and actual > 0.5:
            verdict = "✅ Correct alert"
        elif not predicted_event and actual <= 0.5:
            verdict = "✅ Correctly quiet"
        elif predicted_event and actual <= 0.5:
            verdict = "⚠️ False alarm"
        else:
            verdict = "❌ Missed event"
        rows.append({
            "District": name,
            "Predicted probability": f"{prob:.1%}",
            "Threshold": f"{thr:.1%}",
            "Actual outcome": cfg["event_true_display"] if actual > 0.5 else cfg["event_false_display"],
            "Verdict": verdict,
        })

    df = pd.DataFrame(rows).sort_values("Predicted probability", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.bar_chart(pd.Series(predicted_probs, index=district_names, name="Predicted probability"))

    st.divider()
    st.subheader("Overall test-set performance")
    st.write(
        f"Test set: {len(X_test)} timestamps x {len(district_names)} districts | "
        f"Positive rate (any district {cfg['alert_word']}): {any_actual_event.mean():.1%}"
    )
    st.caption(
        f"This is an MVP trained on Open-Meteo historical forecast data for Tamil "
        f"Nadu, with rectangular district-box approximations (src/districts.py). "
        f"See per-district AUC/precision/recall in models/{cfg['metadata_filename']} "
        f"and the project README for full methodology and next steps."
    )


def render_live_tab(cfg, key_prefix):
    st.subheader(f"Live prediction — right now, all districts ({cfg['title'].lower()})")
    st.warning(f"⚠️ {cfg['caveat']} See {cfg['live_notebook']} for full caveats.")

    result_key = f"last_live_result_{key_prefix}"

    if st.button("🔄 Fetch live data & predict now", type="primary", key=f"fetch_live_btn_{key_prefix}"):
        with st.spinner("Fetching live data from Open-Meteo across the state grid — this can take a minute..."):
            result = cfg["run_live_prediction"]()
            if result["error"]:
                st.error(f"Live fetch failed: {result['error']}")
            else:
                cfg["append_log"](districts=result["districts"], latest_time=result["latest_time"])
                st.session_state[result_key] = result

    if result_key in st.session_state:
        result = st.session_state[result_key]
        st.caption(f"Based on live atmospheric data through: {result['latest_time']}")
        live_df = pd.DataFrame(result["districts"]).sort_values("probability", ascending=False)
        live_df["probability"] = live_df["probability"].map(lambda p: f"{p:.1%}")
        live_df["threshold"] = live_df["threshold"].map(lambda p: f"{p:.1%}")
        live_df["alert"] = live_df["alert"].map(
            lambda a: f"⚠️ Elevated {cfg['alert_word']} risk" if a else "✅ No elevated risk"
        )
        live_df.columns = ["District", "Probability", "Threshold", "Verdict"]
        st.dataframe(live_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Live prediction track record")
    history_df = cfg["load_log"]()
    if len(history_df) == 0:
        st.info(
            f"No live predictions logged yet. Click the button above, or run "
            f"`python {cfg['live_notebook']}` a few times over the coming "
            f"hours/days to build up a track record here."
        )
    else:
        pivot = history_df.pivot_table(
            index="logged_at_utc", columns="district", values="probability", aggfunc="last"
        )
        st.line_chart(pivot)

        st.caption(
            f"Once you know what actually happened for a given row, mark it "
            f"below ({cfg['event_true_label']} / {cfg['event_false_label']}) — this builds a real "
            f"accuracy record over time, per district, separate from your held-out test-set metrics."
        )
        edited_df = st.data_editor(
            history_df,
            column_config={
                "actual_outcome": st.column_config.SelectboxColumn(
                    "actual_outcome", options=["", cfg["event_true_label"], cfg["event_false_label"]]
                )
            },
            disabled=[c for c in history_df.columns if c not in ("actual_outcome", "notes")],
            use_container_width=True,
            key=f"live_log_editor_{key_prefix}",
        )
        if st.button("💾 Save annotations", key=f"save_log_btn_{key_prefix}"):
            cfg["save_log"](edited_df)
            st.success("Saved.")

        annotated = edited_df[edited_df["actual_outcome"] != ""]
        if len(annotated) > 0:
            correct = (
                ((annotated["alert"] == True) & (annotated["actual_outcome"] == cfg["event_true_label"])) |
                ((annotated["alert"] == False) & (annotated["actual_outcome"] == cfg["event_false_label"]))
            ).sum()
            st.caption(
                f"Live track record so far: {correct}/{len(annotated)} correct "
                f"({correct/len(annotated):.0%}) across all annotated district-rows."
            )


def render_hazard_dashboard(hazard: str):
    cfg = HAZARDS[hazard]
    key_prefix = hazard.lower()

    X_test, y_test, times_test, district_names, thresholds, district_masks = load_data(hazard)
    model = load_model(input_shape=X_test.shape[1:], _district_masks=district_masks, hazard=hazard)

    sub_tab1, sub_tab2 = st.tabs(["📊 Historical Test Set", "🔴 Live Prediction"])
    with sub_tab1:
        render_historical_tab(model, X_test, y_test, times_test, district_names, thresholds, cfg, key_prefix)
    with sub_tab2:
        render_live_tab(cfg, key_prefix)


def main():
    st.title("🌩️🌧️ AI-Driven Weather Nowcasting — Per-District MVP")
    st.caption("Smart India Hackathon 2026 — Tamil Nadu, one prediction per district, per hazard")

    hazard_tab1, hazard_tab2 = st.tabs([HAZARDS["Thunderstorm"]["tab_label"], HAZARDS["Cloudburst"]["tab_label"]])
    with hazard_tab1:
        render_hazard_dashboard("Thunderstorm")
    with hazard_tab2:
        render_hazard_dashboard("Cloudburst")


if __name__ == "__main__":
    main()
