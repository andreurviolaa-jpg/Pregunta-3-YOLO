#!/usr/bin/env python3
"""Auditoría humana CCT-20: muestra aleatoria ciega + hojas de contacto + recall.

No infiere oclusión de las predicciones. No modifica el benchmark ni sus etiquetas.
"""
from __future__ import annotations
import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

SPLITS = ('cis_test', 'trans_test')
LABELS = ('Baja', 'Media', 'Alta', 'No_evaluable')
FIELDS = ['ID', 'Particion', 'Imagen', 'Indice_objeto', 'Clase', 'Area_relativa',
          'Oclusion_final', 'Validez_objeto', 'Observaciones']


def leer_labels(dataset):
    for split in SPLITS:
        folder = dataset/'labels'/split
        if not folder.is_dir():
            raise FileNotFoundError(f'Falta {folder}: ejecuta preparar primero')
        for label_file in sorted(folder.glob('*.txt')):
            if not label_file.stat().st_size:
                continue
            foto = dataset/'images'/split/(label_file.stem+'.jpg')
            if not foto.is_file():
                continue
            for index, line in enumerate(label_file.read_text(encoding='utf-8').splitlines()):
                pieces = line.split()
                if len(pieces)<5:
                    continue
                try:
                    cls = int(float(pieces[0]))
                    xc,yc,w,h = map(float,pieces[1:5])
                except ValueError:
                    continue
                if w<=0 or h<=0:
                    continue
                yield {'ID': f'{split}_{label_file.stem}_{index}', 'Particion':split,
                       'Imagen':foto.name,'Indice_objeto':index, 'Clase':cls,
                       'Area_relativa':round(w*h, 7), 'coords':(xc,yc,w,h),
                       'Oclusion_final':'','Validez_objeto':'','Observaciones':''}


def render_tile(record, dataset, width=440, height=300):
    split, filename = record['Particion'], record['Imagen']
    im = Image.open(dataset/'images'/split/filename).convert('RGB')
    W,H = im.size
    xc,yc,w,h = record['coords']
    x1,y1,x2,y2 = (int((xc-w/2)*W),int((yc-h/2)*H),int((xc+w/2)*W),int((yc+h/2)*H))
    im_full = im.copy()
    df = ImageDraw.Draw(im_full)
    df.rectangle([x1,y1,x2,y2],outline='#ff3333',width=max(3,W//180))
    tile = Image.new('RGB',(width,height),'#f4f4f4')
    header = f"{record['ID'][:57]} | clase={record['Clase']}"
    ImageDraw.Draw(tile).text((8,7),header,fill='#111111')
    big = ImageOps.contain(im_full,(278,252))
    tile.paste(big,(8,31+(252-big.height)//2))
    # Zoom con contexto de 1.3x el tamaño de la caja del objeto.
    bw=max(40,x2-x1); bh=max(40,y2-y1)
    side=int(max(bw,bh)*1.6)
    cx=(x1+x2)//2;cy=(y1+y2)//2
    box=(max(0,cx-side//2),max(0,cy-side//2),min(W,cx+side//2),min(H,cy+side//2))
    if box[2]>box[0] and box[3]>box[1]:
        crop=ImageOps.contain(im.crop(box),(136,223))
        tile.paste(crop,(298+(136-crop.width)//2,36+(223-crop.height)//2))
    ImageDraw.Draw(tile).text((297,266),f"bbox area={record['Area_relativa']:.3%}",fill='#333333')
    return tile


def crear(args):
    dataset=Path(args.dataset)
    out=Path(args.salida)
    out.mkdir(parents=True,exist_ok=True)
    path=out/'revision_oclusion.csv'
    if path.exists() and not args.sobrescribir:
        raise FileExistsError(f'{path} ya existe: no se sobrescribe una auditoría humana. Usa --sobrescribir para reiniciar.')
    candidates=list(leer_labels(dataset))
    if not candidates:
        raise RuntimeError('No hay objetos GT en TEST.')
    by_split=defaultdict(list)
    for record in candidates:
        by_split[record['Particion']].append(record)
    rng=random.Random(args.semilla)
    chosen=[]
    for split in SPLITS:
        subset=by_split[split].copy()
        rng.shuffle(subset)
        chosen.extend(subset[:min(len(subset),args.n//2)])
    rng.shuffle(chosen)
    chosen=chosen[:args.n]
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f, fieldnames=FIELDS,delimiter=';')
        writer.writeheader()
        writer.writerows({key:r[key] for key in FIELDS} for r in chosen)
    sheets=out/'laminas'
    sheets.mkdir(exist_ok=True)
    cols, rows, tw, th=3,4,440,300
    for start in range(0,len(chosen),cols*rows):
        sheet=Image.new('RGB',(cols*tw,rows*th),'white')
        for j,record in enumerate(chosen[start:start+cols*rows]):
            tile=render_tile(record,dataset,tw,th)
            sheet.paste(tile,((j%cols)*tw,(j//cols)*th))
        dest=sheets/f'lamina_{start//(cols*rows)+1:02d}.jpg'
        sheet.save(dest,quality=85,optimize=True)
    print(f'[OK] {len(chosen)} objetos elegidos aleatoriamente, mitad por partición')
    print(f'[OK] CSV de anotación: {path}')
    print(f'[OK] {len(list(sheets.glob("lamina_*.jpg")))} láminas en {sheets}')
    print('Anota todas las filas ANTES de revisar las predicciones.')
    print('Oclusion_final: Baja, Media, Alta o No_evaluable.')
    print('Validez_objeto: Confirmado o No_evaluable.')


def resumen(args):
    import pandas as pd
    revision=Path(args.csv)
    df=pd.read_csv(revision,sep=';',encoding='utf-8-sig',keep_default_na=False)
    if not set(FIELDS).issubset(df.columns):
        raise ValueError('Faltan columnas del CSV de revisión')
    if df['ID'].duplicated().any():
        raise ValueError('Hay IDs duplicados')
    if (df['Oclusion_final'].eq('') | df['Validez_objeto'].eq('')).any():
        pendientes=df[df['Oclusion_final'].eq('') | df['Validez_objeto'].eq('')]
        raise ValueError(f'Faltan {len(pendientes)} casos por etiquetar; completar auditoría antes de evaluar')
    evaluables=df[(df['Validez_objeto']=='Confirmado') &
                  df['Oclusion_final'].isin(['Baja','Media','Alta'])].copy()
    if not len(evaluables):
        raise ValueError('No hay casos confirmados evaluables')
    pieces=[]
    diagnosticos=Path(args.diagnosticos)
    for split in SPLITS:
        ruta=diagnosticos/f'YOLOv8s_{split}_objetos.csv'
        if not ruta.is_file():
            raise FileNotFoundError(f'Falta {ruta}; ejecuta etapa diagnosticos')
        obj=pd.read_csv(ruta)
        obj=obj[obj['Particion']==split].copy()
        obj['Indice_objeto']=obj['Indice_objeto'].astype(int)
        pieces.append(obj)
    gt=pd.concat(pieces,ignore_index=True)
    gt=gt.drop_duplicates(['Particion','file_name','Indice_objeto'])
    joined=evaluables.merge(gt,how='left',left_on=['Particion','Imagen','Indice_objeto'],
                           right_on=['Particion','file_name','Indice_objeto'],validate='one_to_one',indicator=True)
    if (joined['_merge']!='both').any():
        raise ValueError(f"No hay diagnóstico para {(joined['_merge']!='both').sum()} objetos auditados")
    if (joined['Clase_x'].astype(int)!=joined['Clase_y'].astype(int)).any():
        raise ValueError('Hay clases que no coinciden con las anotaciones TEST')
    out=revision.parent
    recall=joined.groupby(['Particion','Oclusion_final'],as_index=False).agg(
        Casos=('Detectado','size'),Detectados=('Detectado','sum'))
    recall['Recall_pct']=(100*recall['Detectados']/recall['Casos']).round(2)
    recall.to_csv(out/'resumen_oclusion_adicional.csv',index=False,encoding='utf-8-sig')
    joined.to_csv(out/'detalle_oclusion_adicional.csv',index=False,encoding='utf-8-sig')
    print('\n[RESULTADOS] Auditoría nueva e independiente (NO fusionar sin análisis con muestra anterior)')
    print(recall.to_string(index=False))
    print(f'\nEvaluables: {len(evaluables)}; alta: {(evaluables.Oclusion_final=="Alta").sum()}')
    print(f'CSV: {out / "resumen_oclusion_adicional.csv"}')
    if not (evaluables.Oclusion_final=='Alta').any():
        print('[AVISO] Aún no hay casos altos confirmados; no inventar resultados.')
    print('Muestra aleatoria estratificada por split, NO necesariamente representativa por especie.')


def main():
    p=argparse.ArgumentParser(description='Auditoría manual reproducible de oclusión en CCT-20.')
    cmd=p.add_subparsers(dest='accion',required=True)
    c=cmd.add_parser('crear',help='Crear muestra ciega y láminas visuales')
    c.add_argument('--dataset',default=r'C:\cct20_trabajo\dataset')
    c.add_argument('--salida',default=r'C:\cct20_trabajo\auditoria_oclusion_nueva')
    c.add_argument('--n',type=int,default=240)
    c.add_argument('--semilla',type=int,default=2026)
    c.add_argument('--sobrescribir',action='store_true')
    r=cmd.add_parser('resumen',help='Calcular recall desde diagnósticos ya completados')
    r.add_argument('--csv',default=r'C:\cct20_trabajo\auditoria_oclusion_nueva\revision_oclusion.csv')
    r.add_argument('--diagnosticos',default=r'C:\cct20_trabajo\resultados_analisis\diagnosticos_reproducibles')
    args=p.parse_args()
    if args.accion=='crear':crear(args)
    else:resumen(args)

if __name__=='__main__':main()
