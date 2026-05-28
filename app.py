import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
import json
from datetime import datetime

MODEL_PATH = "mnist_model.keras"
CORRECTIONS_LOG_PATH = "corrections_log.json"

# Data

@st.cache_resource
def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return (x_train / 255.0, y_train), (x_test / 255.0, y_test)


# Model

def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def load_model_cached():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None


# Image preprocessing

def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("L")
    img = ImageOps.invert(img)      # MNIST: white digit on black background
    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.reshape(1, 28, 28)


# Persistent corrections log

def load_disk_log() -> dict:
    if os.path.exists(CORRECTIONS_LOG_PATH):
        with open(CORRECTIONS_LOG_PATH, "r") as f:
            return json.load(f)
    return {"total_corrections": 0, "total_confirmations": 0, "history": []}


def append_disk_log(predicted: int, label: int, correct: bool):
    data = load_disk_log()
    if correct:
        data["total_confirmations"] += 1
    else:
        data["total_corrections"] += 1
    data["history"].append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "predicted": predicted,
        "label": label,
        "correct": correct,
    })
    data["history"] = data["history"][-200:]   # keep last 200 entries
    with open(CORRECTIONS_LOG_PATH, "w") as f:
        json.dump(data, f, indent=2)


# Online learning helpers

def fine_tune(model, arr: np.ndarray, label: int, steps: int = 5):
    label_arr = np.array([label])
    for _ in range(steps):
        model.train_on_batch(arr, label_arr)
    model.save(MODEL_PATH)


def log_feedback(predicted: int, label: int, correct: bool):
    if "feedback_log" not in st.session_state:
        st.session_state["feedback_log"] = []
    st.session_state["feedback_log"].append(
        {"predicted": predicted, "label": label, "correct": correct}
    )
    append_disk_log(predicted, label, correct)


# Pages

def page_predict():
    st.title("Digit Predictor")

    model = st.session_state.get("model")
    if model is None:
        st.warning("No trained model found. Go to **Train Model** first.")
        return

    uploaded_file = st.file_uploader(
        "Upload a handwritten digit image (PNG or JPG)",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.markdown(
            "<div style='text-align:center; padding: 3rem; color: #888; "
            "border: 2px dashed #ccc; border-radius: 12px; margin-top: 1rem;'>"
            "<h3 style='margin:0'>Drop an image above to get started</h3>"
            "<p style='margin:0.5rem 0 0'>Works best with a dark digit on a white background</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # Reset state on new upload
    file_id = (uploaded_file.name, uploaded_file.size)
    if st.session_state.get("last_file_id") != file_id:
        st.session_state["last_file_id"] = file_id
        st.session_state["feedback_done"] = False
        img = Image.open(uploaded_file)
        st.session_state["current_img"] = img
        st.session_state["current_arr"] = preprocess_image(img)

    img = st.session_state["current_img"]
    arr = st.session_state["current_arr"]

    probs = model.predict(arr, verbose=0)[0]
    predicted = int(np.argmax(probs))
    confidence = float(probs[predicted]) * 100

    # Main result row: images + prediction + feedback side by side
    st.markdown("<div style='margin-top: 1.5rem'></div>", unsafe_allow_html=True)
    col_img, col_input, col_result = st.columns([1, 1, 2], gap="large")

    with col_img:
        st.markdown("**Your image**")
        st.image(img, use_container_width=True)

    with col_input:
        st.markdown("**28×28 model input**")
        st.image(arr.reshape(28, 28), use_container_width=True, clamp=True)

    with col_result:
        # Big prediction display
        st.markdown(
            f"<div style='background:#f0f2f6; border-radius:16px; padding:1.5rem 2rem; "
            f"text-align:center; margin-bottom:1rem;'>"
            f"<div style='font-size:0.85rem; color:#666; margin-bottom:0.25rem; "
            f"text-transform:uppercase; letter-spacing:0.08em;'>Prediction</div>"
            f"<div style='font-size:5rem; font-weight:800; line-height:1; color:#1a1a2e;'>{predicted}</div>"
            f"<div style='font-size:1rem; color:#555; margin-top:0.5rem;'>Confidence: "
            f"<strong>{confidence:.1f}%</strong></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Feedback
        if st.session_state.get("feedback_done"):
            last = st.session_state["last_feedback"]
            if last["correct"]:
                st.success(f"Confirmed **{last['label']}** — saved to disk.")
            else:
                st.success(
                    f"Corrected **{last['predicted']}** → **{last['label']}** "
                    f"— saved to disk. New confidence: **{confidence:.1f}%**"
                )
            if st.button("Try another image", use_container_width=True):
                st.session_state["feedback_done"] = False
                st.session_state["last_file_id"] = None
                st.rerun()
        else:
            st.markdown(
                "<div style='font-size:0.9rem; color:#444; margin-bottom:0.5rem;'>"
                "Was this correct?</div>",
                unsafe_allow_html=True,
            )
            if st.button("✓  Yes, correct!", use_container_width=True, type="primary"):
                fine_tune(model, arr, predicted, steps=3)
                log_feedback(predicted, predicted, correct=True)
                st.session_state["feedback_done"] = True
                st.session_state["last_feedback"] = {"correct": True, "label": predicted, "predicted": predicted}
                st.rerun()

            st.markdown(
                "<div style='font-size:0.85rem; color:#888; "
                "text-align:center; margin: 0.4rem 0;'>— or —</div>",
                unsafe_allow_html=True,
            )

            correct_label = st.selectbox(
                "It's actually a:",
                list(range(10)),
                index=predicted,
                key="correction_select",
            )
            if st.button("✗  Correct & teach model", use_container_width=True):
                fine_tune(model, arr, correct_label, steps=6)
                log_feedback(predicted, correct_label, correct=False)
                st.session_state["feedback_done"] = True
                st.session_state["last_feedback"] = {"correct": False, "label": correct_label, "predicted": predicted}
                st.rerun()

    # Probability bar chart
    st.markdown("---")
    st.markdown("**Confidence per digit**")
    st.bar_chart({str(i): float(probs[i]) for i in range(10)}, height=200)

    # Lifetime history (collapsed by default)
    disk_log = load_disk_log()
    lifetime_total = disk_log["total_corrections"] + disk_log["total_confirmations"]

    if lifetime_total > 0:
        with st.expander(f"Feedback history — {lifetime_total} total entries"):
            lm1, lm2, lm3 = st.columns(3)
            lm1.metric("Total", lifetime_total)
            lm2.metric("Confirmed correct", disk_log["total_confirmations"])
            lm3.metric("Corrections", disk_log["total_corrections"])
            st.markdown("")
            for entry in reversed(disk_log["history"][-20:]):
                ts = entry["timestamp"]
                if entry["correct"]:
                    st.write(f"✓  `{ts}`  Predicted **{entry['predicted']}** — confirmed")
                else:
                    st.write(f"✗  `{ts}`  Predicted **{entry['predicted']}** → corrected to **{entry['label']}**")


def page_train():
    st.title("Train Model")

    (x_train, y_train), (x_test, y_test) = load_data()

    epochs = st.slider("Epochs", min_value=1, max_value=20, value=5)
    retrain = st.button("Train / Retrain")

    if retrain:
        model = build_model()
        progress_bar = st.progress(0, text="Training…")
        epoch_placeholder = st.empty()

        history_log = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}

        for epoch in range(epochs):
            hist = model.fit(x_train, y_train, epochs=1, validation_split=0.2, verbose=0)
            for key in history_log:
                history_log[key].append(hist.history[key][0])

            progress_bar.progress((epoch + 1) / epochs, text=f"Epoch {epoch + 1}/{epochs}")
            epoch_placeholder.markdown(
                f"acc `{history_log['accuracy'][-1]:.4f}` | "
                f"val_acc `{history_log['val_accuracy'][-1]:.4f}` | "
                f"loss `{history_log['loss'][-1]:.4f}` | "
                f"val_loss `{history_log['val_loss'][-1]:.4f}`"
            )

        model.save(MODEL_PATH)
        st.session_state["model"] = model
        st.session_state["history"] = history_log
        # Clear any stale feedback state after a full retrain
        st.session_state.pop("feedback_log", None)
        st.session_state.pop("feedback_done", None)
        st.session_state.pop("last_file_id", None)

        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        st.success(f"Training complete — test accuracy: **{test_acc * 100:.2f}%**")

    history = st.session_state.get("history")
    if history:
        st.subheader("Training curves")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(history["accuracy"], marker="o", color="steelblue", label="Train")
        ax1.plot(history["val_accuracy"], marker="o", color="coral", label="Val")
        ax1.set_title("Accuracy over epochs")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(history["loss"], marker="o", color="steelblue", label="Train")
        ax2.plot(history["val_loss"], marker="o", color="coral", label="Val")
        ax2.set_title("Loss over epochs")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    elif not retrain:
        if st.session_state.get("model") is not None:
            st.info("Model loaded from disk. Click **Train / Retrain** to train fresh and see curves.")
        else:
            st.info("Click **Train / Retrain** to start.")


def page_eda():
    st.title("Exploratory Data Analysis")

    (x_train, y_train), (x_test, y_test) = load_data()

    # Class distribution
    st.subheader("Class distribution")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Training set**")
        unique, counts = np.unique(y_train, return_counts=True)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(unique, counts, color="steelblue")
        ax.set_title("Training samples per digit")
        ax.set_xlabel("Digit")
        ax.set_ylabel("Count")
        ax.set_xticks(range(10))
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.markdown("**Test set**")
        unique_t, counts_t = np.unique(y_test, return_counts=True)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(unique_t, counts_t, color="coral")
        ax.set_title("Test samples per digit")
        ax.set_xlabel("Digit")
        ax.set_ylabel("Count")
        ax.set_xticks(range(10))
        st.pyplot(fig)
        plt.close(fig)

    st.markdown(f"Training samples: **{len(x_train):,}** | Test samples: **{len(x_test):,}**")

    # Sample images
    st.subheader("Sample images from training set")
    digit_filter = st.selectbox("Show samples for digit", ["All"] + list(range(10)))

    if digit_filter == "All":
        indices = np.random.choice(len(x_train), 20, replace=False)
    else:
        pool = np.where(y_train == int(digit_filter))[0]
        indices = np.random.choice(pool, min(20, len(pool)), replace=False)

    fig = plt.figure(figsize=(12, 5))
    for i, idx in enumerate(indices):
        ax = fig.add_subplot(4, 5, i + 1)
        ax.imshow(x_train[idx], cmap="gray")
        ax.set_title(str(y_train[idx]), fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Average pixel intensity
    st.subheader("Average pixel intensity per digit")
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for digit in range(10):
        ax = axes[digit // 5][digit % 5]
        avg_img = x_train[y_train == digit].mean(axis=0)
        ax.imshow(avg_img, cmap="hot")
        ax.set_title(f"Digit {digit}")
        ax.axis("off")
    plt.suptitle("Average image per class")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def page_evaluate():
    st.title("Model Evaluation")

    model = st.session_state.get("model")
    if model is None:
        st.warning("No trained model found. Go to **Train Model** first.")
        return

    (_, _), (x_test, y_test) = load_data()

    with st.spinner("Running predictions on test set…"):
        y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)

    test_acc = (y_pred == y_test).mean() * 100
    st.metric("Test accuracy", f"{test_acc:.2f}%")

    # Confusion matrix
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=range(10), yticklabels=range(10))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
    plt.close(fig)

    # Classification report
    st.subheader("Classification report")
    report = classification_report(y_test, y_pred, output_dict=True)
    rows = []
    for label in [str(i) for i in range(10)]:
        r = report[label]
        rows.append({
            "Digit": label,
            "Precision": f"{r['precision']:.3f}",
            "Recall": f"{r['recall']:.3f}",
            "F1-score": f"{r['f1-score']:.3f}",
            "Support": int(r["support"]),
        })
    st.table(rows)

    # Most confused pairs
    st.subheader("Most confused digit pairs")
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    flat_idx = np.argsort(cm_no_diag.ravel())[::-1][:5]
    for rank, idx in enumerate(flat_idx, 1):
        actual, pred = divmod(idx, 10)
        st.write(f"{rank}. Actual **{actual}** predicted as **{pred}** — {cm_no_diag[actual, pred]} times")


# Sidebar navigation

st.set_page_config(page_title="MNIST Classifier", layout="wide")

PAGES = {
    "Predict": page_predict,
    "Train Model": page_train,
    "EDA": page_eda,
    "Evaluate": page_evaluate,
}

with st.sidebar:
    st.title("MNIST Classifier")
    st.markdown("---")
    page = st.radio("Navigate", list(PAGES.keys()))
    st.markdown("---")

    if "model" not in st.session_state:
        st.session_state["model"] = load_model_cached()

    if st.session_state.get("model") is not None:
        st.success("Model ready")
    else:
        st.warning("No model — train first")

    disk_log = load_disk_log()
    lifetime_total = disk_log["total_corrections"] + disk_log["total_confirmations"]
    if lifetime_total > 0:
        st.markdown("**Lifetime feedback (all restarts)**")
        st.markdown(f"Corrections: **{disk_log['total_corrections']}**")
        st.markdown(f"Confirmations: **{disk_log['total_confirmations']}**")

    session_log = st.session_state.get("feedback_log", [])
    if session_log:
        session_correct = sum(1 for e in session_log if e["correct"])
        st.markdown("**This session**")
        st.markdown(f"Feedback given: **{len(session_log)}**")
        st.markdown(f"Corrections: **{len(session_log) - session_correct}**")

PAGES[page]()
