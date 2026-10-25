"""mAP TEST por especie + brechas CIS/TRANS (13 especies en común)."""
from pathlib import Path
import pandas as pd
from .config import ARQUITECTURAS, CANTIDADES_ESPERADAS


def _csv_paths(results: Path, modelo: str, split: str):
    if modelo == 'YOLOv8n':
        return results / f'metricas_final_{split}.csv', results / f'metricas_especies_{split}.csv'
    return results / f'metricas_yolov8s_{split}.csv', results / f'metricas_yolov8s_especies_{split}.csv'


def evaluar(dataset: Path, runs: Path, results: Path, dispositivo: str = 'auto', strict=True):
    from ultralytics import YOLO
    import torch
    dataset, runs, results = Path(dataset), Path(runs), Path(results)
    results.mkdir(parents=True, exist_ok=True)
    device = 'cpu' if dispositivo == 'cpu' else (0 if torch.cuda.is_available() else 'cpu')
    for nombre, (_, nombre_run) in ARQUITECTURAS.items():
        pesos = runs / nombre_run / 'weights' / 'best.pt'
        if not pesos.is_file():
            raise FileNotFoundError(f'No existe checkpoint de {nombre}: {pesos}')
        for split in ('cis_test', 'trans_test'):
            general_csv, especies_csv = _csv_paths(results, nombre, split)
            if general_csv.is_file() and especies_csv.is_file():
                print(f'{nombre} {split}: métricas existentes; se omite reevaluación.')
                continue
            yaml = dataset / f'data_{split}.yaml'
            if not yaml.is_file():
                raise FileNotFoundError(f'Falta el dataset preparado: {yaml}')
            n = len(list((dataset / 'images' / split).glob('*.jpg')))
            if strict and n != CANTIDADES_ESPERADAS[split]:
                raise ValueError(f'{split}: se esperaban {CANTIDADES_ESPERADAS[split]} fotos, hay {n}')
            modelo = YOLO(str(pesos))
            met = modelo.val(
                data=str(yaml), split='test', imgsz=640, batch=16,
                device=device, plots=True, project=str(runs),
                name=f'reevaluacion_{nombre}_{split}', exist_ok=True,
            )
            general = pd.DataFrame([dict(
                Modelo=nombre, Particion=split,
                Precision=float(met.box.mp), Recall=float(met.box.mr),
                mAP50=float(met.box.map50), mAP50_95=float(met.box.map),
            )])
            por_especie = pd.DataFrame([
                dict(Especie=modelo.names[int(cid)], mAP50=float(ap50), mAP50_95=float(ap95))
                for cid, ap50, ap95 in zip(met.box.ap_class_index, met.box.ap50, met.box.ap)
            ])
            general.to_csv(general_csv, index=False)
            por_especie.to_csv(especies_csv, index=False)
            print(f'{nombre} {split}: mAP50={met.box.map50:.4f}; mAP50-95={met.box.map:.4f}')


def resumir(results: Path):
    results = Path(results)
    tablas, filas, brechas = [], [], []
    for nombre in ARQUITECTURAS:
        info = {}
        for split in ('cis_test', 'trans_test'):
            g_path, e_path = _csv_paths(results, nombre, split)
            if not g_path.is_file() or not e_path.is_file():
                raise FileNotFoundError(f'Faltan resultados de {nombre}/{split}; primero evaluar')
            g = pd.read_csv(g_path)
            e = pd.read_csv(e_path)
            assert len(g) == 1 and e['Especie'].is_unique
            filas.append(g.iloc[0].to_dict())
            info[split] = e
        unidos = info['cis_test'].merge(
            info['trans_test'], on='Especie',
            suffixes=('_CIS', '_TRANS'), validate='one_to_one'
        )
        unidos['Modelo'] = nombre
        unidos['Brecha_mAP50_pp'] = 100 * (unidos['mAP50_TRANS'] - unidos['mAP50_CIS'])
        tablas.append(unidos)
        brechas.append(dict(
            Modelo=nombre, Especies_comunes=len(unidos),
            mAP50_CIS_comparable=unidos['mAP50_CIS'].mean(),
            mAP50_TRANS_comparable=unidos['mAP50_TRANS'].mean(),
            Brecha_pp=unidos['Brecha_mAP50_pp'].mean(),
        ))
    general = pd.DataFrame(filas)
    detalle = pd.concat(tablas, ignore_index=True)
    resumen = pd.DataFrame(brechas)
    general.to_csv(results / 'comparacion_yolov8n_vs_yolov8s_test.csv', index=False)
    detalle.to_csv(results / 'generalizacion_por_especie.csv', index=False)
    resumen.to_csv(results / 'generalizacion_comparable_13_especies.csv', index=False)
    print('\nRESULTADOS TEST'); print(general.to_string(index=False))
    print('\nGENERALIZACIÓN POR ESPECIES COMPARTIDAS'); print(resumen.to_string(index=False))
    return general, detalle, resumen
