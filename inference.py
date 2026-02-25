import os
import torch
import librosa
import look2hear.models
import soundfile as sf
from tqdm.auto import tqdm
import numpy as np
import yaml
from ml_collections import ConfigDict
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# RUTAS
# =========================================================
INPUT_DIR = "/content/drive/MyDrive/Apollo"
CKPT_PATH = "/content/Apollo/model/apollo_model_uni.ckpt"
CONFIG_PATH = "/content/Apollo/configs/config_apollo_uni.yaml"

# Carpeta raíz de salida
EDITED_ROOT = os.path.join(INPUT_DIR, "Editado IA")
os.makedirs(EDITED_ROOT, exist_ok=True)

# =========================================================
# UTILIDADES
# =========================================================
def get_config(config_path):
    with open(config_path) as f:
        return ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

def load_audio(file_path):
    audio, samplerate = librosa.load(file_path, mono=False, sr=44100)
    print(f'INPUT audio.shape = {audio.shape} | samplerate = {samplerate}')
    return torch.from_numpy(audio), samplerate

def save_audio(file_path, audio, samplerate=44100):
    sf.write(file_path, audio.T, samplerate, subtype="PCM_16")

def process_chunk(chunk):
    chunk = chunk.unsqueeze(0).cuda()
    with torch.no_grad():
        return model(chunk).squeeze(0).squeeze(0).cpu()

def _getWindowingArray(window_size, fade_size):
    fadein = torch.linspace(1, 1, fade_size)
    fadeout = torch.linspace(0, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] *= fadeout
    window[:fade_size] *= fadein
    return window

# =========================================================
# PROCESAMIENTO POR ARCHIVO
# =========================================================
def main(input_wav, output_wav):
    test_data, samplerate = load_audio(input_wav)

    C = chunk_size * samplerate
    step = C // overlap
    fade_size = 3 * 44100
    border = C - step

    if len(test_data.shape) == 1:
        test_data = test_data.unsqueeze(0)

    if test_data.shape[1] > 2 * border and border > 0:
        test_data = torch.nn.functional.pad(test_data, (border, border), mode='reflect')

    windowingArray = _getWindowingArray(C, fade_size)

    result = torch.zeros((1,) + tuple(test_data.shape))
    counter = torch.zeros_like(result)

    i = 0
    pbar = tqdm(total=test_data.shape[1], desc="Processing chunks", leave=False)

    while i < test_data.shape[1]:
        part = test_data[:, i:i + C]
        length = part.shape[-1]

        if length < C:
            part = torch.nn.functional.pad(part, (0, C - length))

        out = process_chunk(part)

        window = windowingArray.clone()

        if i == 0:
            window[:fade_size] = 1
        elif i + C >= test_data.shape[1]:
            window[-fade_size:] = 1

        result[..., i:i+length] += out[..., :length] * window[:length]
        counter[..., i:i+length] += window[:length]

        i += step
        pbar.update(step)

    pbar.close()

    final_output = (result / counter).squeeze(0).numpy()
    np.nan_to_num(final_output, copy=False)

    if test_data.shape[1] > 2 * border and border > 0:
        final_output = final_output[..., border:-border]

    save_audio(output_wav, final_output, samplerate)
    print(f"Guardado: {output_wav}")

# =========================================================
# INICIALIZACIÓN MODELO
# =========================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = get_config(CONFIG_PATH)
chunk_size = 10
overlap = 2

model = look2hear.models.BaseModel.from_pretrain(
    CKPT_PATH,
    sr=config.model.sr,
    win=config.model.win,
    feature_dim=config.model.feature_dim,
    layer=config.model.layer
).cuda()

model.eval()

# =========================================================
# PROCESAMIENTO POR SUBCARPETAS
# =========================================================
audio_exts = (".wav", ".mp3", ".flac", ".aiff", ".m4a")

subfolders = [
    f for f in os.listdir(INPUT_DIR)
    if os.path.isdir(os.path.join(INPUT_DIR, f)) and f != "Editado IA"
]

print(f"Carpetas encontradas: {len(subfolders)}")

for folder in subfolders:

    input_folder = os.path.join(INPUT_DIR, folder)
    output_folder = os.path.join(EDITED_ROOT, f"{folder}_IA")

    os.makedirs(output_folder, exist_ok=True)

    files = sorted(
        f for f in os.listdir(input_folder)
        if f.lower().endswith(audio_exts)
    )

    print(f"\nProcesando carpeta: {folder} ({len(files)} archivos)")

    for idx, fname in enumerate(files, 1):

        in_path = os.path.join(input_folder, fname)
        out_name = os.path.splitext(fname)[0] + ".wav"
        out_path = os.path.join(output_folder, out_name)

        if os.path.exists(out_path):
            print(f"[{idx}/{len(files)}] {fname} — ya procesado")
            continue

        print(f"[{idx}/{len(files)}] Procesando {fname}")
        main(in_path, out_path)

# =========================================================
# LIMPIEZA
# =========================================================
model.cpu()
torch.cuda.empty_cache()
print("\nTODO TERMINADO")