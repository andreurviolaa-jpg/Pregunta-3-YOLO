"""Diagnósticos en TEST: tamaño, falsas alarmas y oclusión auditada.

Reglas: conf>=0.25, IoU>=0.50, clase igual, matching 1:1 por
confianza. Los recalls por tamaño/oclusión NO son mAP de Ultralytics.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ARQUITECTURAS


def iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0., x2-x1) * max(0., y2-y1)
    area_a = max(0., a[2]-a[0]) * max(0., a[3]-a[1])
    area_b = max(0., b[2]-b[0]) * max(0., b[3]-b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.


def leer_cajas_gt(txt: Path):
    reales = []
    if not txt.exists():
        raise FileNotFoundError(txt)
    for j, linea in enumerate(txt.read_text().splitlines()):
        if not linea.strip():
            continue
        c, xc, yc, w, h = map(float, linea.split()[:5])
        reales.append(dict(objeto=j, clase=int(c), caja=(xc-w/2, yc-h/2, xc+w/2, yc+h/2), area=w*h,
                            sospechosa=(w>=.95 and h>=.95)))
    return reales


def matching_uno_a_uno(reales, cajas_pred, clases_pred, conf_pred, umbral=.5):
    """Devuelve IDs de objetos GT cubiertos, ordenando predicciones por confianza."""
    usados = set()
    if len(conf_pred) == 0:
        return usados
    for j in np.argsort(-np.asarray(conf_pred)):
        mejor, mayor = None, umbral
        for k, gt in enumerate(reales):
            if k in usados or gt['clase'] != int(clases_pred[j]):
                continue
            solap = iou_xyxy(gt['caja'], cajas_pred[j])
            if solap >= mayor:
                mejor, mayor = k, solap
        if mejor is not None:
            usados.add(mejor)
    return usados


def ejecutar_prediccion(modelo, imagen, device, conf=.25):
    pred = modelo.predict(source=imagen, imgsz=640, conf=conf, iou=.70,
                          device=device, verbose=False, save=False)[0]
    return (pred.boxes.xyxyn.cpu().numpy(),
            pred.boxes.cls.cpu().numpy().astype(int),
            pred.boxes.conf.cpu().numpy())


def _guardar(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def tamano(area):
    if area < .01:
        return 'Pequeno (<1%)'
    if area < .10:
        return 'Mediano (1-<10%)'
    return 'Grande (>=10%)'


def evaluar_tamano_y_vacios(dataset: Path, runs: Path, results: Path, *,
                             modelo_nombre='YOLOv8s', device='auto', limite=None):
    """Evalúa imágenes una vez para tamaño y tasa de falsas alarmas en vacíos.

    El parámetro limite es SOLO diagnóstico de código; no comparar métricas
    de una ejecución limitada con el benchmark original.
    """
    from ultralytics import YOLO
    import torch
    _, run = ARQUITECTURAS[modelo_nombre]
    pesos = Path(runs) / run / 'weights' / 'best.pt'
    if not pesos.exists():
        raise FileNotFoundError(pesos)
    dataset, results = Path(dataset), Path(results)
    device = ('cpu' if device == 'cpu' else (0 if torch.cuda.is_available() else 'cpu'))
    salida = results / 'diagnosticos_reproducibles'
    salida.mkdir(parents=True, exist_ok=True)
    config = salida / f'config_{modelo_nombre}.json'
    cfg = dict(checkpoint_size=pesos.stat().st_size, checkpoint_mtime=pesos.stat().st_mtime_ns,
               conf=.25, iou=.5, imgsz=640, limite=limite, protocolo='matching-1a1-confidence')
    if config.exists() and json.loads(config.read_text()) != cfg:
        raise ValueError(f'Configuración anterior diferente en {config}. Usa un directorio limpio para otro experimento.')
    config.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    modelo = YOLO(str(pesos))
    for split in ('cis_test', 'trans_test'):
        archivos_img = salida / f'{modelo_nombre}_{split}_imagenes.csv'
        archivos_obj = salida / f'{modelo_nombre}_{split}_objetos.csv'
        if archivos_img.exists():
            registros = pd.read_csv(archivos_img)
            datos_obj = pd.read_csv(archivos_obj) if archivos_obj.exists() else pd.DataFrame()
            terminados = set(registros['file_name'])
            if len(datos_obj):
                datos_obj = datos_obj[datos_obj['file_name'].isin(terminados)]
        else:
            registros = pd.DataFrame()
            datos_obj = pd.DataFrame()
            terminados = set()
        fotos = sorted((dataset / 'images' / split).glob('*.jpg'))
        if limite:
            fotos = fotos[:limite]
        pendientes = [p for p in fotos if p.name not in terminados]
        print(f'[DIAG] {modelo_nombre} {split}: {len(pendientes)} pendientes de {len(fotos)}', flush=True)
        buffer_f, buffer_o = [], []
        for i, img in enumerate(pendientes, 1):
            gt = leer_cajas_gt(dataset / 'labels' / split / f'{img.stem}.txt')
            boxes, clses, confs = ejecutar_prediccion(modelo, str(img), device)
            aciertos = matching_uno_a_uno(gt, boxes, clses, confs)
            buffer_f.append(dict(file_name=img.name, Particion=split, total_gt=len(gt),
                                 cajas_pred=len(boxes), con_pred=int(len(boxes)>0),
                                 aciertos=len(aciertos)))
            for j, caja in enumerate(gt):
                buffer_o.append(dict(file_name=img.name, Particion=split,
                                     Indice_objeto=j, Clase=caja['clase'],
                                     Area_relativa=caja['area'], Tamano=tamano(caja['area']),
                                     Caja_sospechosa=caja['sospechosa'],
                                     Detectado=int(j in aciertos)))
            if i % 64 == 0 or i == len(pendientes):
                if buffer_o:
                    datos_obj = pd.concat([datos_obj, pd.DataFrame(buffer_o)], ignore_index=True)
                    _guardar(datos_obj, archivos_obj)
                    buffer_o = []
                elif not archivos_obj.exists():
                    _guardar(pd.DataFrame(columns=['file_name','Particion','Indice_objeto','Clase','Area_relativa','Tamano','Caja_sospechosa','Detectado']), archivos_obj)
                registros = pd.concat([registros, pd.DataFrame(buffer_f)], ignore_index=True)
                _guardar(registros, archivos_img)
                buffer_f = []
                print(f'[DIAG] {modelo_nombre} {split}: {len(registros)}/{len(fotos)}', flush=True)
        if not archivos_obj.exists():
            _guardar(pd.DataFrame(columns=['file_name','Particion','Indice_objeto','Clase','Area_relativa','Tamano','Caja_sospechosa','Detectado']), archivos_obj)
    del modelo
    resumen_diagnosticos(salida, modelo_nombre)
    return salida


def resumen_diagnosticos(salida, nombre='YOLOv8s'):
    tabla_o, tabla_v = [], []
    for split in ('cis_test', 'trans_test'):
        imagenes = pd.read_csv(salida / f'{nombre}_{split}_imagenes.csv')
        objetos = pd.read_csv(salida / f'{nombre}_{split}_objetos.csv')
        if len(objetos):
            tabla_o.append(objetos.groupby(['Particion', 'Tamano'], as_index=False).agg(
                Casos=('Detectado','size'), Detectados=('Detectado','sum'), Recall=('Detectado','mean')))
        vacias = imagenes[imagenes['total_gt'] == 0]
        tabla_v.append(dict(Modelo=nombre, Particion=split, Fotos_vacias=len(vacias),
                            Fotos_vacias_con_alarma=int((vacias['cajas_pred'] > 0).sum()),
                            Cajas_falsas=int(vacias['cajas_pred'].sum()),
                            Porcentaje_vacias_con_alarma=(100*(vacias['cajas_pred']>0).mean() if len(vacias) else float('nan')),
                            Cajas_falsas_por_100_fotos=(100*vacias['cajas_pred'].mean() if len(vacias) else float('nan'))))
    size = pd.concat(tabla_o, ignore_index=True) if tabla_o else pd.DataFrame()
    alarms = pd.DataFrame(tabla_v)
    if len(size):
        size['Recall_pct'] = (100*size['Recall']).round(2)
    _guardar(size, salida / f'resumen_tamano_{nombre}.csv')
    _guardar(alarms, salida / f'falsas_alarmas_vacios_{nombre}.csv')
    print('[DIAG] Falsas alarmas sobre negativos:\n', alarms.to_string(index=False), flush=True)
    print('[DIAG] Tamaño:\n', size.to_string(index=False), flush=True)
    return size, alarms
