import os
import uuid
import pickle

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pydub import AudioSegment
import noisereduce as nr

from tensorflow.keras.models import load_model
from lime import lime_tabular
import shap
from sklearn.linear_model import LogisticRegression

# ===========================
# CONFIG
# ===========================
UPLOAD_FOLDER = "uploads"
PLOT_FOLDER = os.path.join("static", "plots")
ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a", "mp4a"}  # match training data extensions

MODEL_PATH = "dialect_model_5feat.h5"          # your CNN+BiLSTM hybrid model
ENCODER_PATH = "label_encoder_5feat.pkl"
FEATURE_CACHE_PATH = "features_cache_5feat.npz"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

MAX_PAD_LEN = 200  # must match training

# 193 = 40 + 12 + 128 + 7 + 6
FEATURE_GROUPS = {
    "MFCC":    (0, 40),
    "Chroma":  (40, 52),
    "Mel":     (52, 180),
    "Contrast":(180, 187),
    "Tonnetz": (187, 193)
}

# ===========================
# FLASK APP
# ===========================
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ===========================
# HELPERS
# ===========================
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def is_valid_audio_buffer(arr):
    return (
        isinstance(arr, np.ndarray)
        and arr.ndim > 0
        and arr.dtype.kind in ("f",)
        and np.all(np.isfinite(arr))
    )


def extract_features_from_array(samples, sr, max_pad_len=MAX_PAD_LEN):
    """
    Feature pipeline identical to the training notebook:
    - Noise reduction
    - MFCC, Chroma, Mel (dB), Spectral Contrast, Tonnetz
    - Pad/truncate each to max_pad_len
    - Stack into (193, max_pad_len)
    """
    try:
        if len(samples) < 512:
            samples = np.pad(samples, (0, 512 - len(samples)), mode="constant")

        samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
        if not is_valid_audio_buffer(samples):
            return None, None

        # ----- Noise reduction (same as training) -----
        noise_len = max(1, int(0.5 * sr))
        reduced = nr.reduce_noise(y=samples, y_noise=samples[:noise_len], sr=sr)

        # 1. MFCC (40)
        mfccs = librosa.feature.mfcc(y=reduced, sr=sr, n_mfcc=40)
        # 2. STFT once for multiple features
        stft = np.abs(librosa.stft(reduced, n_fft=min(512, len(reduced))))
        # 3. Chroma (12)
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
        # 4. Mel (in dB, 128)
        mel = librosa.feature.melspectrogram(y=reduced, sr=sr)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        # 5. Spectral Contrast (7)
        contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
        # 6. Tonnetz (6)
        tonnetz = librosa.feature.tonnetz(
            y=librosa.effects.harmonic(reduced), sr=sr
        )

        def pad_feat(x):
            if x.shape[1] < max_pad_len:
                pad_width = max_pad_len - x.shape[1]
                x = np.pad(x, ((0, 0), (0, pad_width)), mode="constant")
            return x[:, :max_pad_len]

        mfccs   = pad_feat(mfccs)
        chroma  = pad_feat(chroma)
        mel_db  = pad_feat(mel_db)
        contrast= pad_feat(contrast)
        tonnetz = pad_feat(tonnetz)

        stacked = np.vstack([mfccs, chroma, mel_db, contrast, tonnetz]).astype(
            np.float32
        )
        return stacked, (
            mfccs.astype(np.float32),
            chroma.astype(np.float32),
            mel_db.astype(np.float32),
            contrast.astype(np.float32),
            tonnetz.astype(np.float32),
        )
    except Exception as e:
        print("Feature extraction error:", e)
        return None, None


def extract_features(file_path: str):
    """
    Load audio with pydub (same as training), normalize, and call
    extract_features_from_array.
    """
    try:
        audio = AudioSegment.from_file(file_path)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        max_abs = max(1e-6, np.max(np.abs(samples)))
        samples = samples / max_abs  # normalize to [-1, 1]
        sr = audio.frame_rate
        return extract_features_from_array(samples, sr)
    except Exception as e:
        print("File load error:", e)
        return None, None


def save_feature_plots(mfcc, chroma, mel, contrast, tonnetz, base_name: str):
    """Save feature heatmaps and return paths. (medium size)"""
    plot_paths = {}

    def _save(feature, title, key):
        feature = np.array(feature)
        if feature.ndim > 2:
            feature = np.squeeze(feature)
        if feature.ndim == 1:
            feature = feature[np.newaxis, :]

        fig, ax = plt.subplots(figsize=(5, 3))  # medium size
        im = ax.imshow(feature, aspect="auto", origin="lower")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Time frames", fontsize=9)
        ax.set_ylabel("Coefficients", fontsize=9)
        ax.tick_params(labelsize=8)
        fig.tight_layout()

        filename = f"{base_name}_{key}_{uuid.uuid4().hex[:8]}.png"
        full_path = os.path.join(PLOT_FOLDER, filename)
        fig.savefig(full_path, bbox_inches="tight", dpi=120)
        plt.close(fig)

        plot_paths[key] = os.path.join("static", "plots", filename)

    _save(mfcc, "MFCC", "mfcc")
    _save(chroma, "Chroma", "chroma")
    _save(mel, "Mel Spectrogram", "mel")
    _save(contrast, "Spectral Contrast", "contrast")
    _save(tonnetz, "Tonnetz", "tonnetz")

    return plot_paths


def index_to_group_name(idx: int) -> str:
    """Map 0..192 index to human-readable group name like MFCC_1."""
    for gname, (start, end) in FEATURE_GROUPS.items():
        if start <= idx < end:
            return f"{gname}_{idx - start + 1}"
    return f"F{idx + 1}"


# ===========================
# LOAD MAIN MODEL & ENCODER
# ===========================
print("Loading CNN+BiLSTM dialect model...")
model = load_model(MODEL_PATH)
print("Model loaded.")

with open(ENCODER_PATH, "rb") as f:
    le = pickle.load(f)
print("Label encoder loaded.")

# ===========================
# LOAD BACKGROUND FEATURES
# ===========================
npz = np.load(FEATURE_CACHE_PATH)
if "features" in npz.files:
    bg_features = npz["features"].astype(np.float32)   # (N, 193, 200)
else:
    bg_features = npz[npz.files[0]].astype(np.float32)

X_bg_mean = bg_features.mean(axis=2)  # (N, 193)


# ===========================
# COMMON PREDICT FUNCTION
# ===========================
def predict_tabular(x_tab: np.ndarray):
    """
    x_tab: (batch, 193)
    Reconstruct (batch, 193, 200, 1) by repeating along time axis,
    then call the original CNN+BiLSTM model.
    """
    x_full = np.repeat(x_tab[:, :, np.newaxis], MAX_PAD_LEN, axis=2)  # (batch, 193, 200)
    x_full = x_full[..., np.newaxis]                                  # (batch, 193, 200, 1)
    return model.predict(x_full, verbose=0)


# ===========================
# LIME & SHAP INITIALIZATION
# ===========================
print("Initializing LIME & SHAP...")
lime_explainer = lime_tabular.LimeTabularExplainer(
    X_bg_mean,
    mode="classification",
    feature_names=[f"F{i + 1}" for i in range(X_bg_mean.shape[1])],
    class_names=list(le.classes_),
    discretize_continuous=True
)

# surrogate model
y_bg_proba = predict_tabular(X_bg_mean)
y_bg = y_bg_proba.argmax(axis=1)

clf = LogisticRegression(multi_class="multinomial", max_iter=800)
clf.fit(X_bg_mean, y_bg)

shap_explainer = shap.LinearExplainer(clf, X_bg_mean)
print("Explainability ready.")


# ===========================
# LIME & SHAP OUTPUT
# ===========================
def generate_lime_output(features_2d: np.ndarray, pred_idx: int):
    """
    LIME-based explanation for the predicted class.
    Positive weight -> increases probability.
    Negative weight -> decreases probability.
    """
    x_mean = features_2d.mean(axis=1)  # (193,)
    exp = lime_explainer.explain_instance(
        x_mean,
        predict_tabular,
        labels=[pred_idx],
        num_features=15
    )

    contribs = exp.as_map()[pred_idx]  # list of (feature_index, weight)
    results = []
    for feat_idx, weight in contribs:
        feat_name = index_to_group_name(feat_idx)
        results.append((feat_name, float(weight)))

    results.sort(key=lambda x: abs(x[1]), reverse=True)
    return results[:15]


def generate_shap_output(features_2d: np.ndarray, pred_idx: int):
    """
    SHAP-based feature importance for predicted class on 193-d surrogate model.
    Handles various SHAP output shapes robustly.
    """
    x_mean = features_2d.mean(axis=1).reshape(1, -1)  # (1, 193)
    raw = shap_explainer.shap_values(x_mean)

    def _normalize(arr_like):
        arr = np.array(arr_like)
        # common shapes:
        #  (n_classes, 1, 193) -> (n_classes, 193)
        #  (n_classes, 193)
        #  (1, 193)
        #  (193,)
        if arr.ndim == 3 and arr.shape[1] == 1:
            arr = arr[:, 0, :]
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        return arr

    if isinstance(raw, list):
        arr = _normalize(raw)
        if arr.ndim == 2:
            idx = min(max(pred_idx, 0), arr.shape[0] - 1)
            sv = arr[idx]
        else:
            sv = arr
    else:
        arr = _normalize(raw)
        if arr.ndim == 2:
            sv = arr[0]
        else:
            sv = arr

    sv = np.asarray(sv).ravel()  # (193,)

    scores = []
    for i, v in enumerate(sv):
        feat_name = index_to_group_name(i)
        scores.append((feat_name, float(abs(v))))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:15]


# ===========================
# XAI SUMMARY PLOTS (MEDIUM SIZE)
# ===========================
def save_xai_summary_plots(preds_full, class_labels, lime_list, shap_list, base_name: str):
    """
    Create three graphs for a single prediction:
    1) Prediction probabilities (all dialects)
    2) LIME top features (signed weights)
    3) SHAP top features (absolute importance)

    Returns a dict with file paths: proba, lime, shap
    """
    plot_paths = {}

    # ----- 1) Prediction probabilities -----
    probs_percent = preds_full * 100.0
    labels = list(class_labels)

    fig, ax = plt.subplots(figsize=(6, 3))  # medium size
    ax.barh(labels, probs_percent)
    ax.set_xlabel("Prediction probability (%)", fontsize=9)
    ax.set_title("Dialect prediction probabilities", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.invert_yaxis()
    plt.tight_layout()

    fname = f"{base_name}_proba_{uuid.uuid4().hex[:6]}.png"
    full = os.path.join(PLOT_FOLDER, fname)
    fig.savefig(full, bbox_inches="tight", dpi=120)
    plt.close(fig)
    plot_paths["proba"] = os.path.join("static", "plots", fname)

    # ----- 2) LIME TOP FEATURES -----
    if lime_list:
        lime_names, lime_vals = zip(*lime_list)
        lime_names = list(lime_names)[::-1]
        lime_vals = list(lime_vals)[::-1]

        colors = ["green" if v > 0 else "red" for v in lime_vals]

        fig, ax = plt.subplots(figsize=(6, 3.2))  # medium
        ax.barh(lime_names, lime_vals, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("LIME – feature effects for detected dialect", fontsize=10)
        ax.set_xlabel("Effect on prediction (log-odds space)", fontsize=9)
        ax.tick_params(labelsize=8)
        plt.tight_layout()

        fname = f"{base_name}_lime_{uuid.uuid4().hex[:6]}.png"
        full = os.path.join(PLOT_FOLDER, fname)
        fig.savefig(full, bbox_inches="tight", dpi=120)
        plt.close(fig)
        plot_paths["lime"] = os.path.join("static", "plots", fname)

    # ----- 3) SHAP TOP FEATURES -----
    if shap_list:
        shap_names, shap_vals = zip(*shap_list)
        shap_names = list(shap_names)[::-1]
        shap_vals = list(shap_vals)[::-1]

        fig, ax = plt.subplots(figsize=(6, 3.2))  # medium
        ax.barh(shap_names, shap_vals)
        ax.set_title("SHAP – feature importance for detected dialect", fontsize=10)
        ax.set_xlabel("Absolute contribution (|SHAP value|)", fontsize=9)
        ax.tick_params(labelsize=8)
        plt.tight_layout()

        fname = f"{base_name}_shap_{uuid.uuid4().hex[:6]}.png"
        full = os.path.join(PLOT_FOLDER, fname)
        fig.savefig(full, bbox_inches="tight", dpi=120)
        plt.close(fig)
        plot_paths["shap"] = os.path.join("static", "plots", fname)

    return plot_paths


# ===========================
# FLASK ROUTES
# ===========================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template(
                "index.html",
                uploaded=False,
                error="No file part in request.",
            )

        file = request.files["file"]
        if file.filename == "":
            return render_template(
                "index.html",
                uploaded=False,
                error="No file selected.",
            )

        if not allowed_file(file.filename):
            return render_template(
                "index.html",
                uploaded=False,
                error="File type not allowed. Use WAV / MP3 / M4A / MP4A.",
            )

        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        file.save(file_path)

        # Features
        feat, features_split = extract_features(file_path)
        if feat is None:
            return render_template(
                "index.html",
                uploaded=False,
                error="Could not extract features from this audio file. Try another recording.",
            )

        mfcc, chroma, mel, contrast, tonnetz = features_split

        # Prediction from main CNN+BiLSTM model
        feat_input = feat[np.newaxis, ..., np.newaxis]
        preds_full = model.predict(feat_input, verbose=0)[0]  # shape (num_classes,)

        # Top-5 for textual display
        top5_idx = preds_full.argsort()[-5:][::-1]
        top5_labels = le.inverse_transform(top5_idx)
        top5_conf = preds_full[top5_idx] * 100.0

        primary_label = top5_labels[0]
        primary_conf_value = top5_conf[0]

        predictions = [
            {"label": label, "confidence": f"{conf:.2f}%"}
            for label, conf in zip(top5_labels, top5_conf)
        ]

        base_name = os.path.splitext(filename)[0]
        plot_paths = save_feature_plots(
            mfcc, chroma, mel, contrast, tonnetz, base_name
        )

        # XAI for top-1 predicted class
        pred_class_idx = int(top5_idx[0])
        xai_lime_output = generate_lime_output(feat, pred_class_idx)
        xai_shap_output = generate_shap_output(feat, pred_class_idx)

        # Medium-size XAI graphs (all classes for proba, 15 features for LIME/SHAP)
        xai_graph_paths = save_xai_summary_plots(
            preds_full,
            le.classes_,
            xai_lime_output,
            xai_shap_output,
            base_name,
        )

        return render_template(
            "index.html",
            uploaded=True,
            filename=filename,
            predictions=predictions,
            plot_paths=plot_paths,
            xai_shap_output=xai_shap_output,
            xai_lime_output=xai_lime_output,
            xai_graph_paths=xai_graph_paths,
            error=None,
            primary_label=primary_label,
            primary_conf=f"{primary_conf_value:.2f}%",
        )

    # GET
    return render_template(
        "index.html",
        uploaded=False,
        error=None,
        xai_shap_output=None,
        xai_lime_output=None,
        xai_graph_paths=None,
        primary_label=None,
        primary_conf=None,
    )


if __name__ == "__main__":
    # debug=False avoids double initialization
    app.run(host="0.0.0.0", port=5000, debug=False)
