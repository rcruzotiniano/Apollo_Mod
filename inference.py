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
CKPT_PATH = "/content/Apollo_Mod/model/pytorch_model.bin"
CONFIG_PATH = "/content/Apollo_Mod/configs/apollo.yaml"

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

        out = process_chunk(part)

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
# EXTENSIONES DE AUDIO
# =========================================================
audio_exts = (
    ".wav",
    ".mp3",
    ".flac",
    ".aiff",
    ".m4a"
)

# =========================================================
# PROCESAR ARCHIVOS SUELTOS EN APOLLO
# =========================================================
root_files = sorted(
    f for f in os.listdir(INPUT_DIR)
    if os.path.isfile(os.path.join(INPUT_DIR, f))
    and f.lower().endswith(audio_exts)
    and not os.path.splitext(f)[0].endswith("_IA")
)

for fname in root_files:

    in_path = os.path.join(INPUT_DIR, fname)

    out_name = os.path.splitext(fname)[0] + "_IA.wav"
    out_path = os.path.join(INPUT_DIR, out_name)

    if os.path.exists(out_path):
        continue

    print(f"Procesando: {fname}")

    try:
        main(in_path, out_path)

    except Exception as e:
        print(f"ERROR: {fname}")
        print(e)

# =========================================================
# PROCESAR SUBCARPETAS
# =========================================================
subfolders = sorted(
    f for f in os.listdir(INPUT_DIR)
    if os.path.isdir(os.path.join(INPUT_DIR, f))
    and not f.endswith("_IA")
)

for folder in subfolders:

    input_folder = os.path.join(INPUT_DIR, folder)

    output_folder = os.path.join(
        INPUT_DIR,
        folder + "_IA"
    )

    os.makedirs(output_folder, exist_ok=True)

    files = sorted(
        f for f in os.listdir(input_folder)
        if os.path.isfile(os.path.join(input_folder, f))
        and f.lower().endswith(audio_exts)
        and not os.path.splitext(f)[0].endswith("_IA")
    )

    for fname in files:

        in_path = os.path.join(input_folder, fname)

        out_name = os.path.splitext(fname)[0] + "_IA.wav"
        out_path = os.path.join(output_folder, out_name)

        if os.path.exists(out_path):
            continue

        print(f"Procesando: {folder}/{fname}")

        try:
            main(in_path, out_path)

        except Exception as e:
            print(f"ERROR: {folder}/{fname}")
            print(e)

# =========================================================
# LIMPIEZA
# =========================================================
model.cpu()
torch.cuda.empty_cache()

print("\nTODO TERMINADO")