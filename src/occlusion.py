"""Reproducción de auditoría de 120 fotografías; las etiquetas son humanas.

El CSV NO se deduce del dataset: se descarga del Drive público del grupo,
se mantiene el archivo sin cambios y se excluyen casos no evaluables.
Se miden recalls por matching 1:1, no mAP.
"""
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import numpy as np

from .config import ARQUITECTURAS
from .diagnostics import (leer_cajas_gt, ejecutar_prediccion,
                          matching_uno_a_uno, _guardar)

AUDIT_CSV_ID = '1sTEN66x-OIzJG8Ai8HZZkbG25pXt1rem'
AUDIT_NAME = 'muestra_oclusion_test.csv'


def recuperar_auditoria(results, download_fn=None):
    ruta = Path(results) / 'oclusion_test' / AUDIT_NAME
    ruta.parent.mkdir(parents=True, exist_ok=True)
    if not ruta.is_file():
        if download_fn is None:
            import gdown
            download_fn = gdown.download
        print('[OCLUSION] Descargando CSV de auditoría manual, no imágenes...', flush=True)
        try:
            got = download_fn(id=AUDIT_CSV_ID, output=str(ruta), quiet=False, resume=True, retries=3)
        except Exception as exc:
            raise RuntimeError(f'No se pudo descargar el CSV de auditoría manual: '
                               f'https://drive.google.com/file/d/{AUDIT_CSV_ID}/view. '
                               'Verifica permisos o copia el CSV a resultados_analisis/oclusion_test/.') from exc
        if not got or not ruta.exists():
            raise FileNotFoundError('No se pudo descargar el CSV de auditoría.')
    df = pd.read_csv(ruta, keep_default_na=False)
    necesarias = {'ID','Particion','Imagen','Indice_objeto','Clase','Oclusion_final','Validez_objeto'}
    faltan = necesarias - set(df.columns)
    if faltan:
        raise ValueError(f'CSV de auditoría con columnas faltantes: {faltan}')
    if df['ID'].duplicated().any() or len(df) != 120:
        raise ValueError('Se esperaban 120 casos manuales con ID único.')
    return df


def _evaluables(df):
    return df[(df['Validez_objeto'] == 'Confirmado') &
              df['Oclusion_final'].isin(['Baja','Media','Alta'])].copy()


def _result_for_image(modelo, fila, dataset, device):
    split = fila['Particion']; name = fila['Imagen']
    path = dataset / 'images' / split / name
    gt = leer_cajas_gt(dataset / 'labels' / split / (Path(name).stem + '.txt'))
    idx = int(fila['Indice_objeto'])
    if idx < 0 or idx >= len(gt):
        raise ValueError(f'Indice_objeto incorrecto en {fila["ID"]}')
    if gt[idx]['clase'] != int(fila['Clase']):
        raise ValueError(f'Clase incompatible en {fila["ID"]}')
    pred_b, pred_c, pred_conf = ejecutar_prediccion(modelo, str(path), device)
    detectados = matching_uno_a_uno(gt, pred_b, pred_c, pred_conf)
    return int(idx in detectados), gt, idx, path


def ejecutar_oclusion(dataset, runs, results, device='auto'):
    from ultralytics import YOLO
    import torch
    dataset, runs, results = map(Path, (dataset, runs, results))
    audit = recuperar_auditoria(results)
    casos = _evaluables(audit)
    if not len(casos):
        raise RuntimeError('No se encontraron casos auditados evaluables.')
    print(f'[OCLUSION] Auditados {len(audit)}; evaluables {len(casos)}.', flush=True)
    device = 'cpu' if device == 'cpu' else (0 if torch.cuda.is_available() else 'cpu')
    salida = results / 'diagnosticos_reproducibles'
    salida.mkdir(parents=True, exist_ok=True)
    for nombre, (_, run) in ARQUITECTURAS.items():
        pesos = runs / run / 'weights' / 'best.pt'
        if not pesos.is_file():
            raise FileNotFoundError(pesos)
        destino = salida / f'oclusion_{nombre}.csv'
        guardados = pd.read_csv(destino) if destino.exists() else pd.DataFrame()
        hechos = set(guardados['ID']) if len(guardados) else set()
        faltantes = casos[~casos['ID'].isin(hechos)]
        print(f'[OCLUSION] {nombre}: faltan {len(faltantes)} casos', flush=True)
        if len(faltantes):
            modelo = YOLO(str(pesos))
            buffer = []
            for i, (_, fila) in enumerate(faltantes.iterrows(), 1):
                detectado, _, _, _ = _result_for_image(modelo, fila, dataset, device)
                buffer.append(dict(Modelo=nombre, ID=fila['ID'], Particion=fila['Particion'],
                                   Oclusion=fila['Oclusion_final'], Tamano=fila.get('Tamano',''),
                                   Detectado=detectado))
                if i % 10 == 0 or i == len(faltantes):
                    guardados = pd.concat([guardados, pd.DataFrame(buffer)], ignore_index=True)
                    _guardar(guardados, destino)
                    buffer=[]
            del modelo
        if guardados['ID'].duplicated().any():
            raise ValueError('IDs de auditoría duplicados en resultados.')
    todo = pd.concat([pd.read_csv(salida / f'oclusion_{x}.csv') for x in ARQUITECTURAS], ignore_index=True)
    resumen = todo.groupby(['Modelo','Particion','Oclusion'], as_index=False).agg(
        Casos=('Detectado','size'), Detectados=('Detectado','sum'), Recall=('Detectado','mean'))
    resumen['Recall_pct'] = (100 * resumen['Recall']).round(2)
    _guardar(resumen, salida / 'resumen_oclusion_modelos.csv')
    print('[OCLUSION]\n', resumen.to_string(index=False), flush=True)
    return casos, resumen


def ejecutar_fondo(dataset, runs, results, device='auto'):
    """Compara original vs. fondo desenfocado, preservando caja GT auditada.

    Usa el MISMO matching 1:1 de oclusión (los resultados originales antiguos
    en `experimento_fondo_yolov8s.csv` usaban existencia de una predicción).
    """
    import cv2
    import torch
    from ultralytics import YOLO
    dataset, runs, results = map(Path,(dataset,runs,results))
    casos = _evaluables(recuperar_auditoria(results))
    pesos = runs / ARQUITECTURAS['YOLOv8s'][1] / 'weights' / 'best.pt'
    destino = results / 'diagnosticos_reproducibles' / 'fondo_yolov8s_matching_1a1.csv'
    anterior = pd.read_csv(destino) if destino.exists() else pd.DataFrame()
    hechos = set(anterior['ID']) if len(anterior) else set()
    pendientes = casos[~casos['ID'].isin(hechos)]
    device = 'cpu' if device == 'cpu' else (0 if torch.cuda.is_available() else 'cpu')
    if len(pendientes):
        modelo = YOLO(str(pesos))
        buffer = []
        for i, (_, fila) in enumerate(pendientes.iterrows(), 1):
            split, fname = fila['Particion'], fila['Imagen']
            img = cv2.imread(str(dataset / 'images' / split / fname))
            if img is None:
                raise FileNotFoundError(fname)
            gt = leer_cajas_gt(dataset / 'labels' / split / (Path(fname).stem + '.txt'))
            idx = int(fila['Indice_objeto'])
            if gt[idx]['clase'] != int(fila['Clase']):
                raise ValueError(f'Clase incompatible en {fila["ID"]}')
            altura, ancho = img.shape[:2]
            x1,y1,x2,y2 = gt[idx]['caja']
            x1,y1 = max(0,int(x1*ancho)), max(0,int(y1*altura))
            x2,y2 = min(ancho,int(x2*ancho)), min(altura,int(y2*altura))
            fondo = cv2.GaussianBlur(img, (0,0), sigmaX=35)
            mask = np.zeros((altura,ancho), dtype=np.uint8)
            cv2.rectangle(mask,(x1,y1),(x2,y2),255,-1)
            alpha = cv2.GaussianBlur(mask,(0,0),sigmaX=8).astype(np.float32)[...,None]/255
            alterada=(img.astype(np.float32)*alpha+fondo.astype(np.float32)*(1-alpha)).clip(0,255).astype(np.uint8)
            # Resultados manteniendo el mismo protocolo y misma caja objetivo
            vals=[]
            for entrada in (img,alterada):
                pb,pc,conf=ejecutar_prediccion(modelo, entrada, device)
                vals.append(int(idx in matching_uno_a_uno(gt,pb,pc,conf)))
            buffer.append(dict(ID=fila['ID'], Particion=split, Oclusion=fila['Oclusion_final'],
                               Detectado_original=vals[0], Detectado_fondo_desenfocado=vals[1]))
            if i % 10 == 0 or i == len(pendientes):
                anterior=pd.concat([anterior,pd.DataFrame(buffer)],ignore_index=True)
                _guardar(anterior,destino)
                buffer=[]
        del modelo
    resumen=anterior.groupby('Particion',as_index=False).agg(
        Casos=('ID','size'), Aciertos_original=('Detectado_original','sum'),
        Aciertos_alterado=('Detectado_fondo_desenfocado','sum'))
    resumen['Recall_original_pct']=(100*resumen['Aciertos_original']/resumen['Casos']).round(2)
    resumen['Recall_alterado_pct']=(100*resumen['Aciertos_alterado']/resumen['Casos']).round(2)
    resumen['Cambio_pp']=(resumen['Recall_alterado_pct']-resumen['Recall_original_pct']).round(2)
    _guardar(resumen,destino.with_name('resumen_fondo_yolov8s_matching_1a1.csv'))
    print('[FONDO]\n',resumen.to_string(index=False),flush=True)
    return resumen
