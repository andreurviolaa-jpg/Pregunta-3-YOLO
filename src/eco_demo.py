"""Demostración ecológica en TRANS con *secuencias reales* y fechas originales."""
from __future__ import annotations

import shutil
from pathlib import Path
import pandas as pd


def preparar_demo(dataset: Path, destino: Path, max_fotos: int = 48):
    dataset, destino = Path(dataset), Path(destino)
    fuente = dataset / 'metadatos_imagenes.csv'
    if not fuente.is_file():
        raise FileNotFoundError(f'No hay metadatos oficiales: {fuente}. Vuelve a ejecutar preparar.')
    meta = pd.read_csv(fuente, dtype={'location':str, 'seq_id':str, 'file_name':str}).fillna('')
    fotos = meta[(meta['Particion'] == 'trans_test') & (meta['location'] != '')].copy()
    # Sólo secuencias oficialmente identificadas. Evitar secuencias por fotografía.
    fotos = fotos[fotos['seq_id'] != '']
    if fotos.empty:
        raise RuntimeError('No hay metadatos de secuencia en TRANS TEST.')
    tam = fotos.groupby(['location','seq_id'], as_index=False).agg(n=('file_name','size'))
    multi = tam[tam['n'] >= 2]
    if multi.empty:
        raise RuntimeError('No existe una secuencia TRANS con varios fotogramas.')
    # Seleccionar cámara con más secuencias multiframe; desempate determinista.
    candidatas = multi.groupby('location',as_index=False).agg(secuencias=('seq_id','nunique'),imagenes=('n','sum'))
    candidatas=candidatas.sort_values(['secuencias','imagenes','location'],ascending=[False,False,True])
    camara = str(candidatas.iloc[0]['location'])
    candidatos = fotos[fotos['location'] == camara].copy()
    grupos = (candidatos.groupby('seq_id').size().sort_values(ascending=False))
    multiseq = grupos[grupos >= 2].index.tolist()
    # Primero hasta 10 secuencias reales, con hasta 3 fotogramas cada una.
    elegido=[]
    for seq in multiseq[:10]:
        s=candidatos[candidatos['seq_id']==seq].sort_values(['date_captured','file_name'])
        elegido.extend(s.head(3)['file_name'].tolist())
    seleccionado=set(elegido)
    # Luego, rellenar con más fotos de la misma cámara (incluidos vacíos),
    # por periodo real cuando se conoce. Nunca utilizar otra cámara.
    for _, fila in candidatos.sort_values(['date_captured','file_name']).iterrows():
        if len(elegido)>=max_fotos:
            break
        if fila['file_name'] not in seleccionado:
            elegido.append(fila['file_name']); seleccionado.add(fila['file_name'])
    elegido=elegido[:max_fotos]
    if len(elegido)<2:
        raise RuntimeError('No fue posible seleccionar una demostración real.')
    meta_demo=candidatos.set_index('file_name').loc[elegido].reset_index()
    imgs=destino / 'imagenes'
    imgs.mkdir(parents=True, exist_ok=True)
    for nombre in elegido:
        original=dataset / 'images' / 'trans_test' / nombre
        if not original.is_file():
            raise FileNotFoundError(original)
        local=imgs/nombre
        if not local.is_file():
            shutil.copy2(original,local)
    csv=destino/'metadata.csv'
    meta_demo[['file_name','location','seq_id','date_captured']].to_csv(csv,index=False)
    print(f'[ECO] Cámara nueva: {camara}, {len(elegido)} fotografías, '
          f'{meta_demo.seq_id.nunique()} secuencias, '
          f'{(meta_demo.groupby("seq_id").size()>=2).sum()} secuencias de varios fotogramas.',flush=True)
    return imgs, csv, camara


def ejecutar_demo_ecologica(dataset, runs, results, *, cpu=True, max_fotos=48):
    from modulo_ecologico_yolov8s import ejecutar_monitoreo
    dest=Path(results)/'demo_ecologica_reproducible'
    imgs, csv, camara=preparar_demo(dataset,dest,max_fotos=max_fotos)
    pesos=Path(runs)/'comparacion_yolov8s_l4'/'weights'/'best.pt'
    if not pesos.is_file():
        raise FileNotFoundError(pesos)
    resultados=ejecutar_monitoreo(carpeta_imagenes=imgs,pesos=pesos,
                                  salida=dest/'reportes',metadata_csv=csv,
                                  camara=camara,conf=.25,imgsz=640,cpu=cpu,
                                  modelo_nombre='YOLOv8s_final')
    return resultados
