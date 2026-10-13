"""Menu de consola de Apollo.

Elige modelo (y opcionalmente cutoff y carpeta de salida) y ejecuta el motor de
inferencia de inference.py, sin duplicar rutas ni nombres de modelo.

Uso interactivo:
    venv\\Scripts\\python.exe app.py

Uso directo (por ejemplo desde apollo.bat):
    python app.py --input_dir "D:\\Audios" --model universal --cutoff 16000
    python app.py --inference_args "--input_dir D:\\Audios --model normal"
"""
import argparse
import sys
from pathlib import Path

import inference

# --help y --list-models del motor deben responder sin pedir datos al usuario.
DRY_RUN_FLAGS = ("-h", "--help", "--list-models")

MODEL_CHOICES = sorted(inference.MODEL_CONFIGS)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Menu de consola de Apollo.",
        epilog=(
            "Sin argumentos abre el menu interactivo. Ejemplo directo:\n"
            "  python app.py --input_dir \"D:\\\\Audios\" --model universal "
            "--cutoff 16000"
        ),
    )
    parser.add_argument("--input_dir", default=None, help="Carpeta a procesar.")
    parser.add_argument(
        "--model",
        default=None,
        choices=MODEL_CHOICES,
        help="Modelo de Apollo (si no se indica, se pregunta).",
    )
    parser.add_argument(
        "--cutoff",
        type=int,
        default=None,
        help="Cutoff en Hz (1000 a 24000). 0 o vacio = sin cutoff.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Carpeta de salida (si no se indica, se pregunta).",
    )
    parser.add_argument(
        "--suffix",
        default=None,
        help="Sufijo de los WAV de salida (por defecto, el del modelo).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="No pregunta nada: usa los valores por defecto que falten.",
    )
    parser.add_argument(
        "--inference_args",
        default=None,
        help="Argumentos extra para inference.py, tal cual (uso avanzado).",
    )
    return parser.parse_args(argv)


def show_menu():
    print("=" * 40)
    print("            APOLLO IA")
    print("=" * 40)
    print("Selecciona modelo:")
    for index, key in enumerate(MODEL_CHOICES, start=1):
        label = inference.MODEL_CONFIGS[key][0]
        suffix = inference.MODEL_CONFIGS[key][3]
        print(f"  {index} - {label} ({suffix})")
    print()


def choose_model():
    while True:
        raw = input(">> ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(MODEL_CHOICES):
            return MODEL_CHOICES[int(raw) - 1]
        if raw in inference.MODEL_CONFIGS:
            return raw
        print(f"Opcion invalida. Elegi 1-{len(MODEL_CHOICES)} o el nombre del modelo.")


def ask_path(prompt):
    """Pide una carpeta hasta que exista. Devuelve None si el usuario cancela."""
    while True:
        raw = input(prompt).strip().strip('"').strip("'")
        if not raw:
            print("Hace falta una ruta. (Ctrl+C para salir)")
            continue
        path = Path(raw).expanduser()
        if path.is_dir():
            return path
        print(f"No existe esa carpeta: {path}")


def ask_cutoff():
    """Pide el cutoff; el campo es libre y vacio significa sin cutoff."""
    while True:
        raw = input("Cutoff en Hz (1000-24000, vacio = sin cutoff): ").strip()
        if not raw:
            return 0
        try:
            cutoff = int(raw)
        except ValueError:
            print("Tiene que ser un numero entero (por ejemplo 16000).")
            continue
        try:
            return inference.validate_cutoff(
                cutoff, inference.ASSUMED_SAMPLE_RATE
            )
        except inference.CutoffError as e:
            print(e)


def prompt_for_settings(args):
    """Completa lo que falte preguntando, y devuelve (modelo, entrada, cutoff, salida)."""
    show_menu()
    model = args.model or choose_model()
    print(f"\nModelo elegido: {inference.MODEL_CONFIGS[model][0]}\n")

    input_dir = Path(args.input_dir).expanduser() if args.input_dir else None
    while input_dir is None or not input_dir.is_dir():
        if input_dir is not None:
            print(f"No existe esa carpeta: {input_dir}")
        input_dir = ask_path("Carpeta a procesar (arrastra la carpeta): ")

    cutoff = args.cutoff if args.cutoff is not None else ask_cutoff()

    # Si ya vino por argumento, se respeta y no se pregunta nada.
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else None
    if output_dir is None and not args.yes:
        raw = input(
            "\nCarpeta de salida (Enter = guardar junto a los originales): "
        ).strip().strip('"').strip("'")
        if raw:
            output_dir = Path(raw).expanduser()
    # La carpeta la crea el motor de inferencia, que es el dueño de la salida.
    return model, input_dir, cutoff, output_dir


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if args.inference_args:
        # Uso avanzado (o desde la GUI): los argumentos van tal cual al motor.
        tokens = args.inference_args.split()
        if any(flag in tokens for flag in DRY_RUN_FLAGS):
            return inference.main(tokens)
        print(f"Apollo: ejecutando con {args.inference_args}")
        return inference.main(tokens)

    model, input_dir, cutoff, output_dir = prompt_for_settings(args)
    engine_args = [
        "--input_dir", str(input_dir),
        "--model", model,
        "--cutoff", str(cutoff),
    ]
    if output_dir:
        engine_args += ["--output_dir", str(output_dir)]
    if args.suffix:
        engine_args += ["--suffix", args.suffix]

    print("=" * 40)
    print("Ejecutando Apollo...")
    print("=" * 40)
    return inference.main(engine_args)


if __name__ == "__main__":
    sys.exit(main())
