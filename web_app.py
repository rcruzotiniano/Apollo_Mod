"""Interfaz web local de Apollo (Flask).

Elige modelo, carpeta y cutoff, y lanza el motor de inferencia de inference.py
en segundo plano. Como el procesamiento puede durar horas, la salida del motor
se guarda en un archivo de log que se puede consultar desde la propia pagina.

Uso:
    venv\\Scripts\\python.exe web_app.py
    venv\\Scripts\\python.exe web_app.py --port 5000 --open
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from threading import Timer

from flask import Flask, jsonify, render_template, request

import inference

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
DEFAULT_PORT = 5000

# Clave interna -> datos para la interfaz. El nombre a mostrar y el sufijo se
# toman de inference.MODEL_CONFIGS para no duplicar la definicion de modelos.
MODELS = {
    key: {
        "label": config[0],
        "suffix": config[3],
    }
    for key, config in inference.MODEL_CONFIGS.items()
}

LAST_RUN = {"process": None, "log_path": None}


def log_path_for(model: str) -> Path:
    """Ruta del log de la corrida actual (una por modelo y arranque)."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return LOG_DIR / f"apollo-{model}-{stamp}.log"


def run_inference(command: list[str], log_path: Path) -> None:
    """Ejecuta el motor y guarda su salida en el log.

    CREATE_NO_WINDOW evita que se abra una consola negra en Windows; la salida
    se captura con tuberias y se copia al archivo de log.
    """
    creationflags = 0
    if sys.platform == "win32":
        creationflags = 0x08000000  # CREATE_NO_WINDOW

    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"$ {' '.join(command)}\n\n")
        log.flush()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=str(BASE_DIR),
            creationflags=creationflags,
        )
        LAST_RUN["process"] = process
        for line in process.stdout:
            log.write(line)
            log.flush()
        process.wait()
        log.write(f"\n[Proceso terminado con codigo {process.returncode}]\n")


@app.route("/")
def index():
    return render_template(
        "index.html",
        models=MODELS,
        default_cutoff=16000,
    )


@app.route("/process", methods=["POST"])
def process():
    data = request.get_json(silent=True) or {}
    model = data.get("model")
    input_dir = (data.get("input_dir") or "").strip()
    output_dir = (data.get("output_dir") or "").strip()
    cutoff_raw = str(data.get("cutoff", "") or "").strip()

    if model not in MODELS:
        return jsonify({"error": "Modelo invalido"}), 400

    if not input_dir:
        return jsonify({"error": "Falta la carpeta a procesar"}), 400
    if not Path(input_dir).expanduser().is_dir():
        return jsonify({"error": f"No existe la carpeta: {input_dir}"}), 400

    if output_dir and not Path(output_dir).expanduser().is_dir():
        return jsonify({"error": f"No existe la carpeta de salida: {output_dir}"}), 400

    try:
        cutoff = inference.normalize_cutoff(cutoff_raw or 0)
        inference.validate_cutoff(cutoff, inference.ASSUMED_SAMPLE_RATE)
    except inference.CutoffError as e:
        return jsonify({"error": str(e)}), 400

    current = LAST_RUN.get("process")
    if current is not None and current.poll() is None:
        return jsonify({
            "error": (
                "Apollo ya esta procesando un trabajo. Espera a que termine "
                "antes de lanzar otro."
            )
        }), 409

    command = inference.build_command(
        input_dir=input_dir,
        model=model,
        cutoff=cutoff,
        output_dir=output_dir or None,
    )

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = log_path_for(model)
    LAST_RUN["log_path"] = log_path

    Timer(0.2, run_inference, args=(command, log_path)).start()

    return jsonify({
        "status": (
            f"Proceso iniciado con el modelo {MODELS[model]['label']} "
            f"({MODELS[model]['suffix']})."
        ),
        "log": str(log_path),
    })


@app.route("/log")
def read_log():
    """Ultimas lineas del log de la corrida actual."""
    log_path = LAST_RUN.get("log_path")
    if not log_path or not Path(log_path).is_file():
        return jsonify({
            "status": "Todavia no hay ningun trabajo lanzado.",
            "running": False,
            "log": "",
        })

    limit = request.args.get("lines", default=200, type=int)
    lines = Path(log_path).read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()

    process = LAST_RUN.get("process")
    running = process is not None and process.poll() is None
    return jsonify({
        "status": "En curso..." if running else "Terminado.",
        "running": running,
        "log": "\n".join(lines[-limit:]),
    })


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Interfaz web local de Apollo.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--open",
        action="store_true",
        help="Abre el navegador al arrancar.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activa el modo debug de Flask.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    url = f"http://{args.host}:{args.port}"
    print(f"Apollo web en {url}")
    if args.open:
        Timer(1, lambda: webbrowser.open(url)).start()
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    sys.exit(main())
