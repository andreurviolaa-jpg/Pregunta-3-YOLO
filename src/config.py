"""Configuración original del examen (Grupo 2)."""
from pathlib import Path

ESPECIES = (
    'badger', 'bird', 'bobcat', 'cat', 'coyote', 'deer', 'dog',
    'fox', 'opossum', 'rabbit', 'raccoon', 'rodent', 'skunk', 'squirrel',
)
PARTICIONES = ('train', 'cis_val', 'trans_val', 'cis_test', 'trans_test')
CANTIDADES_ESPERADAS = dict(
    train=12099, cis_val=2619, trans_val=1594, cis_test=13085, trans_test=18698
)
ARQUITECTURAS = {
    'YOLOv8n': ('yolov8n', 'final_l4_v1'),
    'YOLOv8s': ('yolov8s', 'comparacion_yolov8s_l4'),
}
PARAMETROS_ENTRENAMIENTO = dict(
    epochs=20, patience=6, imgsz=640, batch=16,
    optimizer='AdamW', lr0=0.0006, lrf=0.10, cos_lr=True,
    seed=0, deterministic=True,
    hsv_h=0.015, hsv_s=0.50, hsv_v=0.30,
    fliplr=0.50, flipud=0.0, mosaic=0.50,
    mixup=0.0, close_mosaic=5,
    amp=True, cache=False, workers=4,
    save=True, save_period=5, plots=True,
)

def rutas(base: Path):
    base = Path(base).expanduser().resolve()
    return {
        'base': base,
        'annotations': base / 'raw' / 'eccv_18_annotations.tar.gz',
        'images': base / 'raw' / 'eccv_18_all_images_sm.tar.gz',
        'dataset': base / 'dataset',
        'runs': base / 'runs',
        'results': base / 'resultados_analisis',
    }
