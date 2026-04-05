import os
import sys
import subprocess
from flask import Flask, render_template, request, jsonify
import webbrowser
from threading import Timer

app = Flask(__name__)

# =========================
# CONFIG MODELOS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "MP3 Enhancer": {
        "ckpt": os.path.join(BASE_DIR, "model", "pytorch_model.bin"),
        "config": os.path.join(BASE_DIR, "configs", "apollo.yaml")
    },
    "Lew Vocal Enhancer": {
        "ckpt": os.path.join(BASE_DIR, "model", "apollo_model.ckpt"),
        "config": os.path.join(BASE_DIR, "configs", "apollo.yaml")
    },
    "Lew Vocal Enhancer v2": {
        "ckpt": os.path.join(BASE_DIR, "model", "apollo_model_v2.ckpt"),
        "config": os.path.join(BASE_DIR, "configs", "config_apollo_vocal.yaml")
    },
    "Universal": {
        "ckpt": os.path.join(BASE_DIR, "model", "apollo_model_uni.ckpt"),
        "config": os.path.join(BASE_DIR, "configs", "config_apollo_uni.yaml")
    }
}

# =========================
# ABRIR NAVEGADOR
# =========================
def open_browser():
    webbrowser.open("http://127.0.0.1:5000")

# =========================
# RUTAS
# =========================
@app.route("/")
def index():
    return render_template("index.html", models=list(MODELS.keys()))

@app.route("/process", methods=["POST"])
def process():
    data = request.json
    model_name = data.get("model")
    input_dir = data.get("input_dir")

    if model_name not in MODELS:
        return jsonify({"error": "Modelo inválido"}), 400

    ckpt = MODELS[model_name]["ckpt"]
    config = MODELS[model_name]["config"]

    python_exe = os.path.join(BASE_DIR, "venv", "Scripts", "python.exe")
    script = os.path.join(BASE_DIR, "inference.py")

    cmd = [
        python_exe,
        script,
        "--input_dir", input_dir,
        "--ckpt", ckpt,
        "--config", config
    ]

    try:
        subprocess.Popen(cmd)
        return jsonify({"status": "Proceso iniciado"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run(debug=False)