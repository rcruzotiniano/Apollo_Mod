"""Inferencia de Apollo: restauracion de audio comprimido.

Script unico de inferencia, pensado para tres usos:

* Colab / linea de comandos:
    python inference.py --input_dir "..." --model normal --cutoff 16000
* Menu de consola: app.py (reutiliza MODEL_CONFIGS y build_command de aca)
* Interfaz web local: web_app.py (idem)

El cutoff es editable: cualquier valor de 1000 a 24000 Hz que quede por debajo
de Nyquist. 0 (o desactivado) conserva el audio de entrada sin filtrar.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import yaml
import warnings
from ml_collections import ConfigDict
from tqdm.auto import tqdm

import look2hear.models
from audio_filter import (
    CutoffError,
    apply_input_cutoff,
    normalize_cutoff,
    validate_cutoff,
)

warnings.filterwarnings("ignore")

# =========================================================
# MODELOS DISPONIBLES
# =========================================================
# Clave (--model) -> (nombre visible, checkpoint, config, sufijo de salida).
# El sufijo identifica con que modelo se proceso cada WAV, asi cambiar de modelo
# no reutiliza las salidas del otro. Las claves coinciden con las que ya usan el
# notebook de Colab y el pipeline de Kaggle.
MODEL_CONFIGS = {
    "normal": (
        "Normal (MP3 enhancer)",
        "model/pytorch_model.bin",
        "configs/apollo.yaml",
        "_N",
    ),
    "vocal": (
        "Lew Vocal Enhancer",
        "model/apollo_model.ckpt",
        "configs/apollo.yaml",
        "_V",
    ),
    "vocal_v2": (
        "Lew Vocal Enhancer v2",
        "model/apollo_model_v2.ckpt",
        "configs/config_apollo_vocal.yaml",
        "_V2",
    ),
    "universal": (
        "Universal",
        "model/apollo_model_uni.ckpt",
        "configs/config_apollo_uni.yaml",
        "_U",
    ),
}
# Salidas del Apollo viejo (local). No son entradas nuevas ni carpetas que haya
# que recorrer: si se ignoraran solo por archivo, el procesador entraria en las
# carpetas "_IA" y reprocesaria las salidas.
LEGACY_SUFFIXES = ("_IA",)
# Sufijos que marcan un archivo como salida de Apollo (de cualquier modelo).
IGNORED_STEM_SUFFIXES = (
    *(s.lower() for s in LEGACY_SUFFIXES),
    *(config[3].lower() for config in MODEL_CONFIGS.values()),
)
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".aiff", ".m4a")
PROJECT_DIR = Path(__file__).resolve().parent
# Frecuencia de muestreo de todos los modelos Apollo. La usan las interfaces
# para validar el cutoff antes de cargar el checkpoint.
ASSUMED_SAMPLE_RATE = 44100

# Parametros de segmentacion (compartidos por el CLI y las interfaces).
CHUNK_SIZE = 10  # segundos
OVERLAP = 2


# =========================================================
# UTILIDADES COMPARTIDAS
# =========================================================
def resolve_project_path(path_like):
    """Devuelve una ruta existente del proyecto.

    Acepta rutas absolutas y, si la ruta relativa no existe desde el directorio
    actual, la resuelve contra la carpeta del script: asi funciona igual en
    Colab (que hace %cd al repo) que desde otra carpeta.
    """
    path = Path(path_like)
    if path.is_file():
        return path
    candidate = PROJECT_DIR / path
    return candidate if candidate.is_file() else path


def normalize_suffix(suffix):
    """Garantiza que el sufijo empiece con guion bajo (o quede vacio)."""
    suffix = suffix.strip()
    if suffix and not suffix.startswith("_"):
        suffix = "_" + suffix
    return suffix


def suffix_for_model(model):
    """Sufijo de salida por defecto del modelo elegido."""
    return MODEL_CONFIGS[model][3]


def build_command(input_dir, model, cutoff=0, output_dir=None, suffix=None,
                  ckpt=None, config=None):
    """Arma el comando del motor de inferencia.

    Lo usan app.py y web_app.py para no duplicar rutas ni nombres de modelo.
    """
    if model not in MODEL_CONFIGS:
        raise ValueError(f"Modelo desconocido: {model}")

    python_exe = Path(sys.executable)
    # Cuando app.py/web_app.py corren desde su propio venv, sys.executable ya es
    # el correcto; si no, se prefiere el venv del proyecto si existe.
    venv_python = PROJECT_DIR / "venv" / "Scripts" / "python.exe"
    if venv_python.is_file() and "venv" not in str(python_exe).lower():
        python_exe = venv_python

    command = [
        str(python_exe),
        str(PROJECT_DIR / "inference.py"),
        "--input_dir", str(input_dir),
        "--model", model,
        "--cutoff", str(int(cutoff or 0)),
    ]
    if output_dir:
        command += ["--output_dir", str(output_dir)]
    if suffix:
        command += ["--suffix", normalize_suffix(str(suffix))]
    if ckpt:
        command += ["--ckpt", str(ckpt)]
    if config:
        command += ["--config", str(config)]
    return command


def is_ignored_stem(stem):
    """True si el nombre de archivo ya es una salida de Apollo."""
    return stem.lower().endswith(IGNORED_STEM_SUFFIXES)


def is_ignored_dir(name):
    """True si la carpeta es una salida vieja de Apollo (por ejemplo "x_IA")."""
    return name.lower().endswith(tuple(s.lower() for s in LEGACY_SUFFIXES))


def find_audio_files(input_dir):
    """Audios a procesar, recursivo, salteando salidas propias y carpetas viejas."""
    input_dir = Path(input_dir)
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in AUDIO_EXTS
        and not is_ignored_stem(path.stem)
        and not any(
            is_ignored_dir(part)
            for part in path.relative_to(input_dir).parts[:-1]
        )
    )


def print_models():
    print("Modelos disponibles:")
    for key, (label, ckpt, config, suffix) in sorted(MODEL_CONFIGS.items()):
        print(
            f"  {key:<10} {label:<22} sufijo={suffix:<4} "
            f"ckpt={ckpt} config={config}"
        )


# =========================================================
# ARGUMENTOS
# =========================================================
def build_parser():
    parser = argparse.ArgumentParser(
        description="Apollo: restauracion de audio comprimido.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  python inference.py --input_dir \"/content/drive/MyDrive/Apollo\" "
            "--model normal --cutoff 16000\n"
            "  python inference.py --input_dir ./audios --model universal "
            "--cutoff 0 --output_dir ./restaurados"
        ),
    )
    parser.add_argument(
        "--input_dir",
        default="/content/drive/MyDrive/Apollo",
        help="Carpeta con archivos de audio y subcarpetas (recursivo).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Carpeta para los audios restaurados. Por defecto, usa input_dir.",
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_CONFIGS),
        default="normal",
        help="Modelo de Apollo a aplicar (por defecto: normal).",
    )
    parser.add_argument(
        "--cutoff",
        type=int,
        default=0,
        help=(
            "Cutoff del filtro pasabajos de entrada, en Hz. Acepta cualquier "
            "valor de 1000 a 24000 (por debajo de Nyquist); usa 0 para no filtrar."
        ),
    )
    parser.add_argument(
        "--suffix",
        default=None,
        help=(
            "Sufijo de los WAV de salida. Si no se indica, se usa el sufijo del "
            f"modelo ({', '.join(m[3] for m in MODEL_CONFIGS.values())})."
        ),
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Ruta explicita al checkpoint (reemplaza la del modelo elegido).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Ruta explicita al archivo de configuracion.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Muestra los modelos disponibles y termina.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Verifica el cutoff, la configuracion y la carga del checkpoint, "
            "muestra la lista de audios detectados y termina sin procesar."
        ),
    )
    return parser


# =========================================================
# AUDIO Y PROCESAMIENTO
# =========================================================
def get_config(config_path):
    with open(config_path) as f:
        return ConfigDict(yaml.load(f, Loader=yaml.FullLoader))


@dataclass
class LoadedModel:
    """El modelo ya cargado junto con los datos que necesita el procesamiento.

    Agrupar esto evita mezclar dos cosas distintas que antes se llamaban igual
    ("model"): el objeto de red y la clave del modelo elegido.
    """

    network: torch.nn.Module
    device: str
    sample_rate: int
    cutoff_hz: int


def load_audio(file_path, loaded):
    """Carga el audio en la frecuencia del modelo y aplica el cutoff."""
    audio, samplerate = librosa.load(
        file_path, mono=False, sr=int(loaded.sample_rate)
    )
    # El cutoff ya fue validado contra esta frecuencia de muestreo al arrancar.
    audio = apply_input_cutoff(audio, samplerate, loaded.cutoff_hz)
    cutoff_label = f"{loaded.cutoff_hz} Hz" if loaded.cutoff_hz else "sin cutoff"
    print(
        f"INPUT audio.shape = {audio.shape} | samplerate = {samplerate} "
        f"| cutoff = {cutoff_label}"
    )
    return torch.from_numpy(np.ascontiguousarray(audio)), samplerate


def save_audio(file_path, audio, samplerate):
    sf.write(file_path, audio.T, samplerate, subtype="PCM_16")


def process_chunk(chunk, loaded):
    chunk = chunk.unsqueeze(0).to(loaded.device)

    with torch.no_grad():
        return loaded.network(chunk).squeeze(0).squeeze(0).cpu()


def _getWindowingArray(window_size, fade_size):
    fadein = torch.linspace(1, 1, fade_size)
    fadeout = torch.linspace(0, 0, fade_size)

    window = torch.ones(window_size)
    window[-fade_size:] *= fadeout
    window[:fade_size] *= fadein

    return window


def process_file(input_wav, output_wav, loaded):
    """Procesa un audio por ventanas solapadas y guarda el WAV restaurado."""
    test_data, samplerate = load_audio(input_wav, loaded)

    C = CHUNK_SIZE * samplerate
    step = C // OVERLAP
    fade_size = 3 * samplerate
    border = C - step

    if len(test_data.shape) == 1:
        test_data = test_data.unsqueeze(0)

    if test_data.shape[1] > 2 * border and border > 0:
        test_data = torch.nn.functional.pad(
            test_data,
            (border, border),
            mode="reflect"
        )

    windowingArray = _getWindowingArray(C, fade_size)

    result = torch.zeros((1,) + tuple(test_data.shape))
    counter = torch.zeros_like(result)

    i = 0

    pbar = tqdm(
        total=test_data.shape[1],
        desc="Processing chunks",
        leave=False
    )

    while i < test_data.shape[1]:

        part = test_data[:, i:i + C]
        length = part.shape[-1]

        if length < C:
            part = torch.nn.functional.pad(
                part,
                (0, C - length)
            )

        out = process_chunk(part, loaded)

        window = windowingArray.clone()

        if i == 0:
            window[:fade_size] = 1
        elif i + C >= test_data.shape[1]:
            window[-fade_size:] = 1

        result[..., i:i + length] += (
            out[..., :length] * window[:length]
        )

        counter[..., i:i + length] += (
            window[:length]
        )

        i += step
        pbar.update(step)

    pbar.close()

    final_output = (result / counter).squeeze(0).numpy()

    np.nan_to_num(final_output, copy=False)

    if test_data.shape[1] > 2 * border and border > 0:
        final_output = final_output[..., border:-border]

    save_audio(output_wav, final_output, samplerate)

    print(f"Guardado: {output_wav}")


def load_model(ckpt_path, config_path, device, cutoff_hz):
    """Carga el checkpoint de Apollo y devuelve el modelo listo para procesar."""
    config = get_config(config_path)
    network = look2hear.models.BaseModel.from_pretrain(
        ckpt_path,
        sr=config.model.sr,
        win=config.model.win,
        feature_dim=config.model.feature_dim,
        layer=config.model.layer,
    ).to(device)
    network.eval()
    return LoadedModel(
        network=network,
        device=device,
        sample_rate=int(config.model.sr),
        cutoff_hz=cutoff_hz,
    )


# =========================================================
# PROGRAMA PRINCIPAL
# =========================================================
def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_models:
        print_models()
        return 0

    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else input_dir
    label, default_ckpt, default_config, model_suffix = MODEL_CONFIGS[args.model]
    ckpt_path = (
        Path(args.ckpt).expanduser() if args.ckpt
        else resolve_project_path(default_ckpt)
    )
    config_path = (
        Path(args.config).expanduser() if args.config
        else resolve_project_path(default_config)
    )
    output_suffix = normalize_suffix(
        args.suffix if args.suffix is not None else model_suffix
    )

    try:
        cutoff_hz = normalize_cutoff(args.cutoff)
    except CutoffError as e:
        parser.error(str(e))

    if not ckpt_path.is_file():
        parser.error(f"No encontre el checkpoint del modelo: {ckpt_path}")
    if not config_path.is_file():
        parser.error(f"No encontre la configuracion del modelo: {config_path}")
    if not input_dir.is_dir():
        parser.error(f"La carpeta de entrada no existe: {input_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Modelo: {label} ({args.model})")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Configuracion: {config_path}")
    print(f"Dispositivo: {device}")
    if device == "cpu":
        print(
            "AVISO: no hay GPU disponible, se procesa en CPU y puede tardar mucho.",
            flush=True,
        )

    loaded = load_model(ckpt_path, config_path, device, cutoff_hz)

    try:
        loaded.cutoff_hz = validate_cutoff(loaded.cutoff_hz, loaded.sample_rate)
    except CutoffError as e:
        parser.error(str(e))

    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.resolve() == input_dir.resolve():
        print(f"Salida: {output_dir} (junto a los originales)")
    else:
        print(f"Salida: {output_dir}")

    audio_files = find_audio_files(input_dir)

    if not audio_files:
        print(
            f"\nNo hay audios compatibles ({', '.join(AUDIO_EXTS)}) en {input_dir}"
        )

    if args.check:
        print(f"\nAudios detectados: {len(audio_files)}")
        for path in audio_files[:20]:
            print(f"  - {path.relative_to(input_dir)}")
        if len(audio_files) > 20:
            print(f"  ... y {len(audio_files) - 20} mas")
        print(
            f"\nCutoff validado: {loaded.cutoff_hz or 'sin cutoff'} "
            f"(sr {loaded.sample_rate} Hz)"
        )
        print(f"Sufijo de salida: {output_suffix}")
        print("VERIFICACION OK (no se proceso ningun audio)")
        return 0

    for index, input_path in enumerate(audio_files, start=1):
        relative_path = input_path.relative_to(input_dir)
        output_path = (
            output_dir
            / relative_path.parent
            / f"{relative_path.stem}{output_suffix}.wav"
        )

        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"[{index}/{len(audio_files)}] Ya existe, se omite: {relative_path}")
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n[{index}/{len(audio_files)}] Procesando: {relative_path}", flush=True)

        try:
            process_file(str(input_path), str(output_path), loaded)
        except Exception as e:
            print(f"ERROR en {relative_path}: {e}")

    loaded.network.cpu()
    if device == "cuda":
        torch.cuda.empty_cache()

    print("\nTODO TERMINADO")
    return 0


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES se fija antes de tocar la GPU.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    sys.exit(main())
