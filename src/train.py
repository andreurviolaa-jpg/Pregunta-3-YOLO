"""Entrenamiento controlado: requiere autorización explícita y GPU CUDA."""
from pathlib import Path
from .config import ARQUITECTURAS, PARAMETROS_ENTRENAMIENTO


def entrenar(dataset: Path, runs: Path, permitir_gpu: bool):
    if not permitir_gpu:
        raise PermissionError('No se entrena por defecto. Usa --permitir-entrenamiento-gpu.')
    import torch
    from ultralytics import YOLO
    if not torch.cuda.is_available():
        raise RuntimeError('Entrenamiento bloqueado: no hay GPU CUDA.')
    yaml = Path(dataset) / 'data_cis.yaml'
    if not yaml.is_file():
        raise FileNotFoundError('Primero prepara el dataset (falta data_cis.yaml).')
    runs = Path(runs)
    for etiqueta, (arquitectura, nombre_run) in ARQUITECTURAS.items():
        destino = runs / nombre_run
        mejor = destino / 'weights' / 'best.pt'
        if mejor.is_file():
            print(f'{etiqueta}: se reutiliza {mejor}')
            continue
        if destino.exists():
            raise RuntimeError(f'Run incompleto: {destino}. Resolver manualmente antes de proseguir.')
        runs.mkdir(parents=True, exist_ok=True)
        print(f'Entrenando {etiqueta}: 20 épocas máximas (consume GPU)')
        modelo = YOLO(f'{arquitectura}.pt')
        modelo.train(data=str(yaml), device=0,
                     project=str(runs), name=nombre_run, exist_ok=False,
                     **PARAMETROS_ENTRENAMIENTO)
        del modelo
        torch.cuda.empty_cache()
