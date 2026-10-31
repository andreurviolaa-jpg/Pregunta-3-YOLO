#!/usr/bin/env python3
"""Un solo comando a partir del repositorio y los TAR públicos de CCT-20.

python reproducir.py --base ./trabajo --gpu

Se instalan dependencias y se ejecuta run_pipeline.py de este repositorio.
La GPU es obligatoria solo para entrenamiento desde cero; ver README.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description='Reproducción CCT-20 desde el Drive público y GitHub')
    parser.add_argument('--base', default='./trabajo', help='Destino de dataset, checkpoints y resultados')
    parser.add_argument('--gpu', action='store_true', help='Autoriza entrenamiento CUDA, muy costoso')
    parser.add_argument('--etapa', choices=['todo', 'descargar', 'preparar', 'entrenar', 'evaluar', 'resumen', 'diagnosticos', 'ecologia', 'figuras', 'pruebas'], default='todo')
    parser.add_argument('--pesos-publicos', action='store_true', help='OMITE entrenar: descarga checkpoints finales públicos y evalúa (no replica entrenamiento)')
    parser.add_argument('--sin-instalar', action='store_true', help='Omitir pip install -r requirements.txt')
    parser.add_argument('--cpu', action='store_true', help='Inferencia sobre CPU (no permite entrenamiento)')
    parser.add_argument('--sin-extra', action='store_true', help='Omitir diagnóstico adicional y demostración ecológica del modo todo')
    parser.add_argument('--limite-diagnostico', type=int, default=None,
                        help='Solo depuración: máximo de imágenes TEST por partición; resultados NO comparables')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    if not args.sin_instalar:
        print('[BOOTSTRAP] Instalando requirements.txt...', flush=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(root / 'requirements.txt')], check=True)
    if args.etapa == 'pruebas':
        cmd = [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v']
    else:
        cmd = [sys.executable, '-u', str(root / 'run_pipeline.py'), '--etapa', args.etapa, '--base', args.base]
        if args.pesos_publicos:
            cmd.append('--pesos-publicos')
        if args.gpu:
            cmd.append('--permitir-entrenamiento-gpu')
        if args.cpu:
            cmd.append('--cpu')
        if args.sin_extra:
            cmd.append('--sin-extra')
        if args.limite_diagnostico is not None:
            cmd.extend(['--limite-diagnostico', str(args.limite_diagnostico)])
    print('[BOOTSTRAP] Ejecutando:', ' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(root), check=True)


if __name__ == '__main__':
    main()
