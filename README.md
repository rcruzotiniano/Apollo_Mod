# Apollo Mod

Herramienta para **restaurar audio comprimido** (MP3 de baja tasa, voces con
artefactos de códec). Usa los modelos **Apollo** de *Kai Li y Yi Luo*
(Tsinghua University + Tencent AI Lab, ICASSP 2025) con tres formas de
ejecutarlo: cuaderno de Colab, menú de consola e interfaz web local.

> Este repositorio es un fork de trabajo de [JusperLee/Apollo](https://github.com/JusperLee/Apollo)
> con modelos afinados propios, filtro de cutoff configurable y utilidades para
> Windows. El proyecto original se documenta en
> [arXiv:2409.08514](https://arxiv.org/abs/2409.08514).

---

## Requisitos

- Windows 10/11 de 64 bits (probado con Python 3.14).
- GPU NVIDIA con CUDA para uso real. **Sin GPU funciona en CPU, pero es muy
  lento**: procesar 3-4 minutos de audio puede tardar horas en lugar de
  minutos. La app avisa cuando detecta que no hay GPU.
- Los pesos de los modelos (unos 314 MB) se bajan con **Git LFS**.

---

## Instalación

```powershell
git clone https://github.com/rcruzotiniano/Apollo_Mod.git
cd Apollo_Mod
git lfs pull          # baja los pesos de model/ (obligatorio)

python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

El `torch` que instala `requirements.txt` es la version de CPU. Para usar la
GPU, instalá primero la version CUDA:

```powershell
.\venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Verificá que la GPU se vea:

```powershell
.\venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Modelos disponibles

| `--model` | Nombre | Checkpoint | Config | Sufijo |
|---|---|---|---|---|
| `normal` | Normal (MP3 enhancer) | `model/pytorch_model.bin` | `configs/apollo.yaml` | `_N` |
| `vocal` | Lew Vocal Enhancer | `model/apollo_model.ckpt` | `configs/apollo.yaml` | `_V` |
| `vocal_v2` | Lew Vocal Enhancer v2 | `model/apollo_model_v2.ckpt` | `configs/config_apollo_vocal.yaml` | `_V2` |
| `universal` | Universal | `model/apollo_model_uni.ckpt` | `configs/config_apollo_uni.yaml` | `_U` |

Listalos desde la consola con:

```powershell
.\venv\Scripts\python.exe inference.py --list-models
```

---

## Uso

### 1. Colab (recomendado: GPU gratis)

Abrí `Apollo_Audio_Restoration_Richard.ipynb` en Colab, activá la GPU
(**Entorno de ejecución → Cambiar tipo de entorno → T4 GPU**) y ejecutá las
celdas. En la última elegís modelo, cutoff y carpetas.

### 2. Menú de consola (Windows)

```powershell
apollo.bat
```

Pregunta modelo, carpeta, cutoff y carpeta de salida. También acepta todo por
argumentos, sin preguntar:

```powershell
.\venv\Scripts\python.exe app.py --input_dir "D:\Musica\Apollo" --model universal --cutoff 16000 --yes
```

### 3. Interfaz web local

```powershell
.\venv\Scripts\python.exe web_app.py --open
```

Queda en `http://127.0.0.1:5000`. El procesamiento corre en segundo plano y el
detalle se ve en el panel **Registro** de la propia página y en `logs\`.

### 4. Línea de comandos directa

```powershell
.\venv\Scripts\python.exe inference.py --input_dir "D:\Musica\Apollo" --model normal --cutoff 16000
```

---

## El cutoff

El cutoff es un **filtro pasabajos de fase cero** (Butterworth orden 8) que se
aplica al audio **antes** de pasarlo por el modelo. Sirve para recortar el ruido
de alta frecuencia que deja el códec y suele mejorar el resultado en MP3
castigados.

- **Es editable**: cualquier valor de **1000 a 24000 Hz**, no solo múltiplos de
  1000. Valores típicos útiles: `11025`, `16000`, `16500`.
- **`0` (o vacío) = sin cutoff**: el audio entra tal cual.
- El valor debe quedar **por debajo de Nyquist** (22050 Hz). El programa rechaza
  el resto con un mensaje claro en lugar de fallar a mitad del proceso.

Antes de lanzar un trabajo largo, podés verificar que todo esté en orden sin
procesar nada:

```powershell
.\venv\Scripts\python.exe inference.py --input_dir "D:\Musica" --model normal --cutoff 16500 --check
```

---

## Cómo se nombran las salidas

Cada modelo deja su propio sufijo, así **cambiar de modelo no reutiliza las
salidas del otro**:

```
cancion.mp3            ->  cancion_N.wav    (modelo normal)
                       ->  cancion_V.wav    (modelo vocal)
                       ->  cancion_V2.wav   (modelo vocal_v2)
                       ->  cancion_U.wav    (modelo universal)
```

Reglas de procesamiento:

- Las **subcarpetas se recorren** y la estructura se conserva.
- Un audio que ya tiene su WAV del modelo elegido **no se reprocesa** (salvo que
  el archivo esté vacío).
- Los archivos que ya son salidas de Apollo (`_N`, `_V`, `_V2`, `_U`, `_IA`) y
  las carpetas `*_IA` **se ignoran como entrada**, así que podés reprocesar una
  carpeta sin miedo a encadenar salidas sobre salidas.
- Formatos de entrada: `.wav`, `.mp3`, `.flac`, `.aiff`, `.m4a`.
- Con `--output_dir` los resultados van a otra carpeta; sin él, quedan junto a
  los originales.

---

## Estructura del proyecto

```
inference.py    Motor único de inferencia (CLI). Lo usan los tres accesos.
audio_filter.py Filtro de cutoff y su validación.
app.py          Menú de consola.
web_app.py      Interfaz web local (Flask) + templates/index.html.
apollo.bat      Lanzador del menú de consola.
look2hear/      Arquitectura Apollo (modelo, STFT, sistema de entrenamiento).
configs/        Configuración por modelo (sr, win, feature_dim, layer).
model/          Pesos en Git LFS.
asserts/        Material de la demo (imágenes, audio de ejemplo).
train.py        Entrenamiento (requiere el dataset MUSDB18-HQ + MoisesDB).
```

---

## Limitaciones conocidas

- **CPU es muy lento.** Para uso práctico hace falta GPU.
- El procesamiento es **por ventanas solapadas** (`chunk_size` de 10 s con 50%
  de solapamiento, definido en `inference.py`), así que la memoria se mantiene
  acotada, pero la velocidad depende por completo de la GPU.
- `train.py` necesita el dataset preprocesado en HDF5 y no está adaptado a este
  fork; está incluido como referencia del proyecto original.
- Los cuadernos y utilidades asumen **44100 Hz**, la frecuencia de todos los
  modelos Apollo.

---

## Créditos y licencia

- Modelo Apollo: **Kai Li, Yi Luo** — *Apollo: Band-sequence Modeling for
  High-Quality Music Restoration in Compressed Audio* (ICASSP 2025).
- Licencia: **CC BY-SA 4.0** (ver [LICENSE](LICENSE)). Los pesos de los modelos
  heredan la licencia del proyecto original.

```bibtex
@inproceedings{li2025apollo,
  title={Apollo: Band-sequence Modeling for High-Quality Music Restoration in Compressed Audio},
  author={Li, Kai and Luo, Yi},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2025},
  organization={IEEE}
}
```
