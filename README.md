# Apollo Mod

Herramienta para **restaurar audio comprimido** (MP3 de baja tasa, voces con
artefactos de códec). Usa los modelos **Apollo** de *Kai Li y Yi Luo*
(Tsinghua University + Tencent AI Lab, ICASSP 2025) con modelos afinados propios
y filtro de cutoff configurable.

Se usa desde **Google Colab**, donde hay GPU gratis. El repositorio aporta el
motor de inferencia, los pesos y el cuaderno que lo maneja.

> Este repositorio es un fork de trabajo de [JusperLee/Apollo](https://github.com/JusperLee/Apollo).
> El proyecto original se documenta en
> [arXiv:2409.08514](https://arxiv.org/abs/2409.08514).

---

## Requisitos

- Una cuenta de Google y acceso a Colab con GPU (**T4**).
- Nada que instalar en tu equipo: Colab trae Python, torch con CUDA y las
  librerías de audio. El cuaderno clona el repo y baja los pesos solo.
- Los pesos de los modelos (unos 314 MB) vienen por **Git LFS**; el cuaderno lo
  resuelve automáticamente.

Opcional, para probar cambios sin gastar una sesión de Colab, se puede correr en
Windows con el venv del proyecto (ver más abajo). Sin GPU funciona, pero es muy
lento.

---

## Instalación

### En Colab

No hay que instalar nada a mano: el cuaderno
`Apollo_Audio_Restoration_Richard.ipynb` clona el repo, hace `git lfs pull` y
monta tu Google Drive. Solo elegí **Entorno de ejecución → Cambiar tipo de
entorno → T4 GPU** y ejecutá las celdas.

### En Windows (opcional, para pruebas)

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

### En Colab (el flujo normal)

Abrí `Apollo_Audio_Restoration_Richard.ipynb` en Colab, activá la GPU
(**Entorno de ejecución → Cambiar tipo de entorno → T4 GPU**) y ejecutá las
celdas. La primera clona el repo y baja los pesos por Git LFS; la última te
deja elegir **modelo, cutoff, carpeta de entrada y carpeta de salida**, y lanza
la inferencia.

### Desde la línea de comandos

Sirve para probar cambios sin gastar una sesión de Colab:

```powershell
.\venv\Scripts\python.exe inference.py --input_dir "D:\Musica\Apollo" --model normal --cutoff 16000
```

Sin GPU esto funciona, pero es muy lento. Antes de lanzar un trabajo largo podés
verificar la configuración sin procesar nada:

```powershell
.\venv\Scripts\python.exe inference.py --input_dir "D:\Musica" --model normal --cutoff 16500 --check
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
inference.py    Motor de inferencia (CLI). Es todo lo que corre Colab.
audio_filter.py Filtro de cutoff y su validación.
look2hear/      Arquitectura Apollo (modelo, STFT y utilidades).
configs/        Configuración por modelo (sr, win, feature_dim, layer).
model/          Pesos en Git LFS.
asserts/        Muestras de audio e imágenes de la demo del paper.
train.py        Entrenamiento (código del proyecto original, no se usa acá).
```

---

## Limitaciones conocidas

- **CPU es muy lento.** Para uso práctico hace falta GPU; en Colab elegí la T4.
- El procesamiento es **por ventanas solapadas** (`chunk_size` de 10 s con 50%
  de solapamiento, definido en `inference.py`), así que la memoria se mantiene
  acotada, pero la velocidad depende por completo de la GPU.
- La frecuencia de muestreo es **44100 Hz** en todos los modelos Apollo.
- `train.py` y los módulos de entrenamiento de `look2hear/` son código del
  proyecto original, incluidos como referencia; no están adaptados a este fork.

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
