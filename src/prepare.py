"""Prepara CCT-20 en formato YOLO replicando los filtros del notebook.

ATENCIÓN: fotos con fauna sin cajas completas se EXCLUYEN; nunca se tratan
como imágenes negativas. Las imágenes solo-car en train (668 en el estudio)
se incluyen con archivos de etiquetas vacíos.
"""
import json
import os
import shutil
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import yaml

from .config import CANTIDADES_ESPERADAS, ESPECIES, PARTICIONES


def cargar_anotaciones(archivo_tar: Path):
    archivo_tar = Path(archivo_tar)
    if not archivo_tar.is_file():
        raise FileNotFoundError(f'Faltan anotaciones: {archivo_tar}')
    datos = {}
    with tarfile.open(archivo_tar, 'r:gz') as tar:
        indice = {
            Path(m.name).name: m for m in tar.getmembers()
            if m.isfile() and m.name.endswith('.json')
        }
        for split in PARTICIONES:
            nombre = f'{split}_annotations.json'
            if nombre not in indice:
                raise FileNotFoundError(f'No existe {nombre} en {archivo_tar}')
            with tar.extractfile(indice[nombre]) as archivo:
                datos[split] = json.load(archivo)
    return datos


def caja_yolo(bbox, ancho, alto):
    if bbox is None or ancho <= 0 or alto <= 0:
        return None
    x, y, w, h = map(float, bbox)
    if w <= 0 or h <= 0:
        return None
    x1 = max(0.0, min(x, ancho))
    y1 = max(0.0, min(y, alto))
    x2 = max(0.0, min(x + w, ancho))
    y2 = max(0.0, min(y + h, alto))
    if x2 <= x1 or y2 <= y1:
        return None
    return (
        (x1 + x2) / (2 * ancho), (y1 + y2) / (2 * alto),
        (x2 - x1) / ancho, (y2 - y1) / alto,
    )


def seleccionar_dataset(datos):
    """Devuelve: nombre_foto -> (partición, líneas YOLO); estadísticas."""
    seleccion = {}
    estadisticas = []
    for split, obj in datos.items():
        categorias = {c['id']: c['name'] for c in obj['categories']}
        por_imagen = defaultdict(list)
        for ann in obj['annotations']:
            por_imagen[ann['image_id']].append(ann)
        cont = Counter()
        for imagen in obj['images']:
            anns = por_imagen.get(imagen['id'], [])
            if not anns:
                cont['sin_anotaciones'] += 1
                continue
            nombres = {categorias[a['category_id']] for a in anns}
            if nombres == {'car'} and split == 'train':
                lineas = []
                cont['negativos_car'] += 1
            elif nombres == {'empty'}:
                lineas = []
                cont['negativos_empty'] += 1
            elif ('car' in nombres or 'empty' in nombres or
                  not nombres.issubset(ESPECIES)):
                cont['categoria_no_elegible'] += 1
                continue
            else:
                ancho = float(imagen['width'])
                alto = float(imagen['height'])
                lineas = []
                for ann in anns:
                    box = caja_yolo(ann.get('bbox'), ancho, alto)
                    if box is None:
                        lineas = []
                        break
                    clase = ESPECIES.index(categorias[ann['category_id']])
                    lineas.append(f'{clase} ' + ' '.join(f'{p:.8f}' for p in box))
                if not lineas:
                    cont['sin_cajas_completas'] += 1
                    continue
            nombre = Path(imagen['file_name']).name
            if nombre in seleccion:
                raise ValueError(f'Nombre de imagen duplicado: {nombre}')
            seleccion[nombre] = (split, lineas)
            cont['incluidas'] += 1
            cont['cajas'] += len(lineas)
        estadisticas.append({'Particion': split, **dict(cont)})
    return seleccion, pd.DataFrame(estadisticas).fillna(0)


def guardar_yamls(dataset: Path):
    base = {
        'path': str(dataset.resolve()),
        'train': 'images/train',
        'val': 'images/cis_val',
        'test': 'images/cis_test',
        'names': {i: nombre for i, nombre in enumerate(ESPECIES)},
    }
    for nombre, cambios in {
        'data_cis.yaml': {},
        'data_trans.yaml': {'val': 'images/trans_val'},
        'data_cis_test.yaml': {'test': 'images/cis_test'},
        'data_trans_test.yaml': {'test': 'images/trans_test'},
    }.items():
        with (dataset / nombre).open('w', encoding='utf-8') as salida:
            yaml.safe_dump({**base, **cambios}, salida, sort_keys=False, allow_unicode=True)


def extraer_imagenes(archivo_tar: Path, seleccion: dict, dataset: Path):
    pendientes = {
        nombre: split for nombre, (split, _) in seleccion.items()
        if not (dataset / 'images' / split / nombre).is_file()
    }
    if not pendientes:
        print('Las fotografías seleccionadas ya están extraídas.')
        return
    if not Path(archivo_tar).is_file():
        raise FileNotFoundError(f'Faltan las imágenes originales: {archivo_tar}')
    print(f'Extrayendo {len(pendientes)} fotografías seleccionadas...')
    with tarfile.open(archivo_tar, 'r|gz') as tar:
        for miembro in tar:
            if not miembro.isfile():
                continue
            nombre = Path(miembro.name).name
            if nombre not in pendientes:
                continue
            destino = dataset / 'images' / pendientes.pop(nombre) / nombre
            destino.parent.mkdir(parents=True, exist_ok=True)
            tmp = destino.with_suffix(destino.suffix + '.part')
            with tar.extractfile(miembro) as origen, tmp.open('wb') as salida:
                shutil.copyfileobj(origen, salida)
            if tmp.stat().st_size != miembro.size:
                raise IOError(f'Extracción incompleta: {nombre}')
            os.replace(tmp, destino)
            if not pendientes:
                break
    if pendientes:
        raise FileNotFoundError(f'No se encontraron {len(pendientes)} fotos en el TAR de imágenes')


def guardar_metadata(datos: dict, seleccion: dict, dataset: Path):
    """Metadatos oficiales por fotografía elegible, sin inventar fechas o secuencias."""
    filas = []
    for split, obj in datos.items():
        for img in obj['images']:
            nombre = Path(img['file_name']).name
            if nombre not in seleccion or seleccion[nombre][0] != split:
                continue
            filas.append(dict(
                file_name=nombre, Particion=split,
                location=str(img.get('location', '')),
                seq_id=str(img.get('seq_id', '')),
                date_captured=img.get('date_captured') or img.get('datetime'),
                image_id=str(img.get('id', '')),
                ancho=int(img['width']), alto=int(img['height']),
                n_cajas=len(seleccion[nombre][1]),
            ))
    df = pd.DataFrame(filas)
    df.to_csv(dataset / 'metadatos_imagenes.csv', index=False)
    # Auditar geografía y secuencias. Las secuencias de TRAIN y CIS VAL
    # pueden coincidir (fotogramas distintos); no afirmar ausencia general.
    desglose=[]
    for split in PARTICIONES:
        parte=df[df['Particion']==split]
        desglose.append(dict(Particion=split, Imagenes=len(parte),
                             Camaras=parte.loc[parte['location'].ne(''), 'location'].nunique(),
                             Secuencias=parte.loc[parte['seq_id'].ne(''), 'seq_id'].nunique(),
                             Negativas=int((parte['n_cajas']==0).sum())))
    pd.DataFrame(desglose).to_csv(dataset/'auditoria_camaras.csv',index=False)
    pares=[]
    for i,a in enumerate(PARTICIONES):
        for b in PARTICIONES[i+1:]:
            suba=df[df.Particion==a]; subb=df[df.Particion==b]
            loc_a=set(suba.loc[suba.location.ne(''),'location']); loc_b=set(subb.loc[subb.location.ne(''),'location'])
            seq_a=set(suba.loc[suba.seq_id.ne(''),'seq_id']); seq_b=set(subb.loc[subb.seq_id.ne(''),'seq_id'])
            pares.append(dict(Particion_A=a,Particion_B=b,
                              Camaras_compartidas=len(loc_a&loc_b),
                              Secuencias_compartidas=len(seq_a&seq_b),
                              Fotografias_compartidas=len(set(suba.file_name)&set(subb.file_name))))
    pd.DataFrame(pares).to_csv(dataset/'auditoria_solapamientos.csv',index=False)
    # La separación geográfica del benchmark es un requisito; parar si se pierde.
    vistas=set(df[df.Particion.isin(['train','cis_val','cis_test'])].location)
    nuevas=set(df[df.Particion.isin(['trans_val','trans_test'])].location)
    vistas.discard(''); nuevas.discard('')
    if vistas & nuevas:
        raise ValueError(f'Cámaras repetidas CIS-TRANS: {sorted(vistas&nuevas)}')
    return df


def preparar(annotations_tar: Path, images_tar: Path, dataset: Path, strict=True):
    dataset = Path(dataset).resolve()
    marker = dataset / 'preparacion_terminada.json'
    if marker.is_file() and strict:
        anterior = json.loads(marker.read_text(encoding='utf-8'))
        archivos = (Path(annotations_tar), Path(images_tar))
        actual = {'dataset': str(dataset), 'raw_bytes': [p.stat().st_size for p in archivos if p.is_file()]}
        if actual == anterior and all(
            len(list((dataset / 'images' / split).glob('*.jpg'))) == CANTIDADES_ESPERADAS[split]
            and len(list((dataset / 'labels' / split).glob('*.txt'))) == CANTIDADES_ESPERADAS[split]
            for split in PARTICIONES
        ) and (dataset / 'metadatos_imagenes.csv').is_file():
            print('Dataset ya preparado y verificado; se omite la extracción.', flush=True)
            return pd.read_csv(dataset / 'auditoria_preparacion.csv')
    datos = cargar_anotaciones(annotations_tar)
    seleccion, auditoria = seleccionar_dataset(datos)
    for split in PARTICIONES:
        cantidad = sum(s == split for s, _ in seleccion.values())
        print(f'{split}: {cantidad} imágenes')
        if strict and cantidad != CANTIDADES_ESPERADAS[split]:
            raise ValueError(f'{split}: {cantidad} imágenes; se esperaban {CANTIDADES_ESPERADAS[split]}')
    for split in PARTICIONES:
        (dataset / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dataset / 'labels' / split).mkdir(parents=True, exist_ok=True)
    for nombre, (split, lineas) in seleccion.items():
        destino = dataset / 'labels' / split / (Path(nombre).stem + '.txt')
        destino.write_text('\n'.join(lineas) + ('\n' if lineas else ''), encoding='utf-8')
    guardar_yamls(dataset)
    extraer_imagenes(images_tar, seleccion, dataset)
    auditoria.to_csv(dataset / 'auditoria_preparacion.csv', index=False)
    guardar_metadata(datos, seleccion, dataset)
    marker.write_text(json.dumps({'dataset': str(dataset), 'raw_bytes': [Path(p).stat().st_size for p in (annotations_tar, images_tar)]}, indent=2), encoding='utf-8')
    print(f'Dataset listo en {dataset}')
    return auditoria
