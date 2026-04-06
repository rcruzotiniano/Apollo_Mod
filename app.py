import subprocess
import sys

def get_model_config(model):
    if model == '1':
        return 'model/pytorch_model.bin', 'configs/apollo.yaml'
    elif model == '2':
        return 'model/apollo_model.ckpt', 'configs/apollo.yaml'
    elif model == '3':
        return 'model/apollo_model_v2.ckpt', 'configs/config_apollo_vocal.yaml'
    elif model == '4':
        return 'model/apollo_model_uni.ckpt', 'configs/config_apollo_uni.yaml'
    else:
        return None, None

print("Selecciona modelo:")
print("1 - MP3 Enhancer")
print("2 - Lew Vocal Enhancer")
print("3 - Lew Vocal Enhancer v2")
print("4 - Universal")

op = input(">> ")

ckpt, config = get_model_config(op)

if not ckpt:
    print("Opción inválida")
    exit()

print("\nArrastra la CARPETA raíz (donde están tus subcarpetas):")
input_dir = input(">> ").strip().strip('"')

cmd = [
    sys.executable,   # 🔥 ESTE es el cambio clave
    "inference_dskt.py",
    "--input_dir", input_dir,
    "--ckpt", ckpt,
    "--config", config
]

subprocess.run(cmd)

input("\nENTER para salir...")