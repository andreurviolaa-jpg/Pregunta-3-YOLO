"""Entrena YOLOv8n/s y reanuda ejecuciones interrumpidas cuando hay last.pt."""
from __future__ import annotations

import json
from pathlib import Path
from .config import ARQUITECTURAS, PARAMETROS_ENTRENAMIENTO


def entrenar(dataset: Path, runs: Path, permitir_gpu: bool):
    if not permitir_gpu:
        raise PermissionError('Para entrenar se requiere --permitir-entrenamiento-gpu (consumo de GPU).')
    import torch
    from ultralytics import YOLO
    if not torch.cuda.is_available():
        raise RuntimeError('No hay CUDA. Necesitas una GPU compatible para el entrenamiento. '
                           'Puedes usar etapas descargar/preparar/resumen en CPU.')
    yaml = Path(dataset) / 'data_cis.yaml'
    if not yaml.is_file():
        raise FileNotFoundError(f'Falta {yaml}; ejecuta primero preparar.')
    runs = Path(runs)
    runs.mkdir(parents=True, exist_ok=True)
    for etiqueta, (arquitectura, nombre_run) in ARQUITECTURAS.items():
        destino = runs / nombre_run
        pesos = destino / 'weights' / 'best.pt'
        ultimo = destino / 'weights' / 'last.pt'
        marcador = destino / 'entrenamiento_terminado.json'
        if marcador.is_file() and pesos.is_file():
            print(f'[TRAIN] {etiqueta}: entrenamiento anterior marcado como completo; reutilizando pesos.', flush=True)
            continue
        # Conservamos los entrenamientos históricos originales: existía best.pt
        # sin nuestro marcador. No afirmamos que hayan pasado por este script.
        if pesos.is_file() and not marcador.is_file():
            # Caso frecuente al reutilizar un entrenamiento histórico completado:
            # 20 filas de results.csv indican que finalizó las 20 épocas.
            historial=destino / 'results.csv'
            if historial.is_file():
                import pandas as pd
                if len(pd.read_csv(historial)) >= int(PARAMETROS_ENTRENAMIENTO['epochs']):
                    print(f'[TRAIN] {etiqueta}: 20 épocas registradas, reutilizando pesos existentes.', flush=True)
                    continue
            if ultimo.is_file():
                # Importante: en una sesión interrumpida best.pt suele existir.
                # Reanudar desde last.pt antes de declarar el entrenamiento completo.
                print(f'[TRAIN] {etiqueta}: best.pt sin marcador; intentaremos continuar desde last.pt.', flush=True)
            else:
                print(f'[TRAIN] {etiqueta}: best.pt preexistente sin last.pt; reutilización sin verificar finalización.', flush=True)
                continue
        if ultimo.is_file() and not marcador.is_file():
            print(f'[TRAIN] {etiqueta}: reanudando desde last.pt...', flush=True)
            modelo = YOLO(str(ultimo))
            # Ultralytics recupera hiperparámetros y estado del optimizador.
            modelo.train(resume=True, device=0)
        elif destino.exists():
            raise RuntimeError(f'{destino} existe sin best.pt/last.pt válidos. '
                               'No se sobrescribe automáticamente; inspecciona esa ejecución.')
        else:
            print(f'[TRAIN] Entrenando {etiqueta} (máximo 20 épocas)...', flush=True)
            modelo = YOLO(f'{arquitectura}.pt')
            modelo.train(data=str(yaml), device=0, project=str(runs), name=nombre_run,
                         exist_ok=False, **PARAMETROS_ENTRENAMIENTO)
        if not pesos.is_file():
            raise FileNotFoundError(f'Ultralytics terminó sin generar {pesos}')
        marcador.write_text(json.dumps({'modelo': etiqueta, 'checkpoint': str(pesos),
                                        'politica': 'finalizado tras retorno exitoso de YOLO.train'},
                                       ensure_ascii=False, indent=2), encoding='utf-8')
        del modelo
        torch.cuda.empty_cache()
