"""CLI para preparar CCT-20, entrenar opcionalmente y evaluar mAP TEST.

Ejemplo (ejecución costosa e intencional):
  python run_pipeline.py --etapa todo --base ./trabajo --permitir-entrenamiento-gpu

Por defecto no se usa GPU ni se entrena. El módulo ecológico del repositorio
se ejecuta por separado con modulo_ecologico_yolov8s.py.
"""
import argparse
from pathlib import Path
from src.config import rutas


def main(argv=None):
    p = argparse.ArgumentParser(description='Pipeline CCT-20 YOLOv8 (Grupo 2).')
    p.add_argument('--etapa', required=True,
                   choices=['preparar', 'entrenar', 'evaluar', 'resumen', 'todo'],
                   help='todo: preparar + entrenar + evaluar + resumen (costoso)')
    p.add_argument('--base', type=Path, default=Path('./trabajo'),
                   help='Carpeta con raw/*.tar.gz, dataset/, runs/, resultados_analisis/')
    p.add_argument('--permitir-entrenamiento-gpu', action='store_true',
                   help='Obligatorio para entrenar (puede consumir muchas unidades GPU).')
    p.add_argument('--cpu', action='store_true',
                   help='Reevaluación TEST en CPU; entrenamiento sigue exigiendo CUDA.')
    args = p.parse_args(argv)
    cfg = rutas(args.base)
    if args.etapa in ('entrenar', 'todo') and not args.permitir_entrenamiento_gpu:
        p.error('Entrenar exige --permitir-entrenamiento-gpu; no inicia GPU por accidente.')
    print('Directorio de trabajo:', cfg['base'])
    print('Etapa:', args.etapa)
    if args.etapa in ('preparar', 'todo'):
        from src.prepare import preparar
        preparar(cfg['annotations'], cfg['images'], cfg['dataset'], strict=True)
    if args.etapa in ('entrenar', 'todo'):
        from src.train import entrenar
        entrenar(cfg['dataset'], cfg['runs'], args.permitir_entrenamiento_gpu)
    if args.etapa in ('evaluar', 'todo'):
        from src.evaluate import evaluar
        evaluar(cfg['dataset'], cfg['runs'], cfg['results'],
                dispositivo='cpu' if args.cpu else 'auto', strict=True)
    if args.etapa in ('resumen', 'todo'):
        from src.evaluate import resumir
        resumir(cfg['results'])
    if args.etapa == 'evaluar':
        from src.evaluate import resumir
        resumir(cfg['results'])

if __name__ == '__main__':
    main()
