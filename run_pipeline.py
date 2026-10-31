"""CLI CCT-20 reproducible desde TAR públicos y código del repositorio.

Desde cero con GPU:
    python reproducir.py --base ./trabajo --gpu
Evaluar con pesos públicos preexistentes (sin reentrenar):
    python reproducir.py --base ./trabajo --pesos-publicos --cpu

La etapa todo incluye descargas, preparación, pesos o entrenamiento,
validación TEST, diagnósticos de tamaño/negativos, oclusión/fondo (auditados),
demostración ecológica multiframe y figuras.
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from src.config import rutas


def main(argv=None):
    p=argparse.ArgumentParser(description='Pipeline CCT-20 YOLOv8 (Grupo 2)')
    p.add_argument('--etapa',required=True,choices=[
        'descargar','preparar','entrenar','evaluar','resumen','diagnosticos','ecologia','figuras','todo'])
    p.add_argument('--base',type=Path,default=Path('./trabajo'))
    p.add_argument('--permitir-entrenamiento-gpu',action='store_true',
                   help='Permite entrenar YOLOv8n y YOLOv8s (requiere CUDA, es costoso)')
    p.add_argument('--pesos-publicos',action='store_true',
                   help='Solo en todo/evaluar: descargar best.pt publicados por el grupo; NO entrena')
    p.add_argument('--cpu',action='store_true',help='Inferencia CPU (mucho más lenta que GPU)')
    p.add_argument('--sin-extra',action='store_true',help='En todo, omitir tamaño/vacíos, oclusión, fondo, demo y figuras')
    p.add_argument('--limite-diagnostico',type=int,default=None,
                   help='Prueba reducida, NO comparable con resultados completos')
    args=p.parse_args(argv)

    if args.etapa in ('todo','entrenar') and not args.pesos_publicos and not args.permitir_entrenamiento_gpu:
        p.error('Entrenamiento desde cero exige --permitir-entrenamiento-gpu. '
                'Alternativa para CPU: --pesos-publicos --cpu (NO replica entrenamiento).')
    if args.pesos_publicos and args.etapa == 'entrenar':
        p.error('--pesos-publicos es incompatible con --etapa entrenar.')
    if args.cpu and args.etapa in ('entrenar','todo') and not args.pesos_publicos:
        p.error('Entrenamiento desde cero no admite --cpu; activa GPU CUDA.')

    cfg=rutas(args.base)
    print(f'\n[PROYECTO] Directorio: {cfg["base"]}\n[PROYECTO] Etapa: {args.etapa}',flush=True)
    if args.limite_diagnostico is not None:
        print('⚠ Ejecución parcial: sus indicadores NO son los de TEST completo.',flush=True)

    if args.etapa in ('descargar','preparar','todo'):
        from src.download import descargar_raw
        descargar_raw(cfg['base']/'raw',min_free_gb=0)
        if args.etapa == 'descargar':
            return
    if args.etapa in ('preparar','todo'):
        from src.prepare import preparar
        preparar(cfg['annotations'],cfg['images'],cfg['dataset'],strict=True)
    if args.pesos_publicos and args.etapa in ('todo','evaluar'):
        from src.download import descargar_pesos_publicos
        descargar_pesos_publicos(cfg['runs'])
    elif args.etapa in ('entrenar','todo'):
        from src.train import entrenar
        entrenar(cfg['dataset'],cfg['runs'],args.permitir_entrenamiento_gpu)
    if args.etapa in ('evaluar','todo'):
        from src.evaluate import evaluar
        evaluar(cfg['dataset'],cfg['runs'],cfg['results'],
                dispositivo='cpu' if args.cpu else 'auto',strict=True)
    if args.etapa in ('resumen','evaluar','todo'):
        from src.evaluate import resumir
        resumir(cfg['results'])
    if args.etapa == 'diagnosticos' or (args.etapa=='todo' and not args.sin_extra):
        from src.diagnostics import evaluar_tamano_y_vacios
        from src.occlusion import ejecutar_oclusion, ejecutar_fondo
        evaluar_tamano_y_vacios(cfg['dataset'],cfg['runs'],cfg['results'],
                                modelo_nombre='YOLOv8s',
                                device='cpu' if args.cpu else 'auto',
                                limite=args.limite_diagnostico)
        ejecutar_oclusion(cfg['dataset'],cfg['runs'],cfg['results'],device='cpu' if args.cpu else 'auto')
        ejecutar_fondo(cfg['dataset'],cfg['runs'],cfg['results'],device='cpu' if args.cpu else 'auto')
    if args.etapa == 'ecologia' or (args.etapa=='todo' and not args.sin_extra):
        from src.eco_demo import ejecutar_demo_ecologica
        ejecutar_demo_ecologica(cfg['dataset'],cfg['runs'],cfg['results'],cpu=args.cpu)
    if args.etapa == 'figuras' or (args.etapa=='todo' and not args.sin_extra):
        from src.figures import generar_figuras
        generar_figuras(cfg['results'],cfg['base']/'articulo_final'/'figuras_reproducidas')

    # Manifest documental; nunca afirmar que una ejecución parcial se completó.
    if args.etapa=='todo':
        cfg['results'].mkdir(parents=True,exist_ok=True)
        from src.download import SOURCE_FOLDER, RAW_FILES
        archivo=cfg['results']/'manifiesto_reproduccion.json'
        registro=dict(finalizado=datetime.now(timezone.utc).isoformat(),
                      fuente_raw_publica=SOURCE_FOLDER,archivos_raw={k:v['id'] for k,v in RAW_FILES.items()},
                      python=platform.python_version(),plataforma=platform.platform(),
                      modo='pesos_publicos_sin_reentrenar' if args.pesos_publicos else 'entrenamiento_desde_cero',
                      extras_completados=not args.sin_extra,
                      muestra_diagnostico_limite=args.limite_diagnostico,
                      alcance='TEST filtrado elegible (no colección completa CCT)')
        archivo.write_text(json.dumps(registro,indent=2,ensure_ascii=False),encoding='utf-8')
        print(f'\n[OK] Pipeline finalizado. Manifiesto: {archivo}',flush=True)


if __name__=='__main__':
    main()
