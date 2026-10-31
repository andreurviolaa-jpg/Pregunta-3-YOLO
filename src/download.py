"""Obtención reanudable de los dos TAR del benchmark desde el Drive público.

Se fijan los ID de los *archivos*, no se descarga recursivamente la carpeta
examen_yolo (que contiene datos ajenos a la reproducción de entrenamiento).
Los ID fueron comprobados como contenido de examen_yolo/raw el 2026-10-29.
El acceso sin cuenta / límites de cuota deben verificarse en el entorno destino.
"""
from __future__ import annotations

import shutil
import tarfile
import hashlib
from pathlib import Path

SOURCE_FOLDER = 'https://drive.google.com/drive/folders/1Mq286eDir-lCFQ4MQ9gbvCUERlpVJ6tq?usp=drive_link'
RAW_FOLDER = 'https://drive.google.com/drive/folders/1hpeqFqK0WNzu6RA5y53Y5suHtmhrIMFs'

RAW_FILES = {
    'eccv_18_annotations.tar.gz': {'id': '1_JA450STBag6cPOEkrQTGUdsr3v7I_ho', 'bytes': 2997071, 'md5': '66a1f481b44aa1edadf75c9cfbd27aba'},
    'eccv_18_all_images_sm.tar.gz': {'id': '1WeKsZR892uU8YjHKkGa3z2DSj1ax2cxE', 'bytes': 6492615601, 'md5': '8143c17aa2a12872b66f284ff211531f'},
}


def _valid_size_header(path: Path, expected: int) -> bool:
    if not path.is_file() or path.stat().st_size != expected:
        return False
    with path.open('rb') as f:
        return f.read(2) == b'\x1f\x8b'


def descargar_raw(raw: Path, *, download_fn=None, min_free_gb: float = 0.0) -> dict:
    """Descarga solo si faltan los TAR; reanuda *.part por medio de gdown.

    `download_fn` inyectable para pruebas; firma igual a gdown.download.
    No se extrae el TAR aquí: se encarga `src.prepare`.
    """
    raw = Path(raw).expanduser().resolve()
    raw.mkdir(parents=True, exist_ok=True)
    estado = {}
    pendientes = [nombre for nombre, m in RAW_FILES.items()
                  if not _valid_size_header(raw / nombre, m['bytes'])]
    if pendientes and min_free_gb:
        espacio = shutil.disk_usage(raw).free / 1024**3
        if espacio < min_free_gb:
            raise OSError(f'Espacio insuficiente: {espacio:.1f} GiB libres; se recomiendan al menos {min_free_gb:.1f} GiB. '
                          'Usa una unidad mayor o una carpeta de trabajo diferente.')
    if pendientes and download_fn is None:
        try:
            import gdown
        except ImportError as e:
            raise RuntimeError('Instala las dependencias: python -m pip install -r requirements.txt') from e
        download_fn = gdown.download

    for nombre, meta in RAW_FILES.items():
        destino = raw / nombre
        if _valid_size_header(destino, meta['bytes']):
            print(f'[DRIVE] Ya existe un TAR íntegro por tamaño: {nombre}', flush=True)
            estado[nombre] = destino
            continue
        if destino.exists():
            # Un fichero incompleto en la ruta final puede ser reutilizado por gdown
            # con resume=True, cuando la versión lo admite.
            print(f'[DRIVE] Archivo previo incompleto: {nombre} ({destino.stat().st_size} bytes)', flush=True)
        print(f'[DRIVE] Descargando {nombre} ({meta["bytes"] / 1024**3:.2f} GiB) desde archivo público...', flush=True)
        try:
            resultado = download_fn(
                id=meta['id'], output=str(destino), quiet=False,
                resume=True, retries=3,
            )
        except Exception as exc:
            raise RuntimeError(f'No pudo descargarse {nombre}. Comprueba que el archivo y la carpeta RAW '
                               'tengan acceso «Cualquier persona con el enlace → Lector»; '
                               'Google Drive puede imponer cuotas de descarga. '
                               f'Enlace: https://drive.google.com/file/d/{meta["id"]}/view') from exc
        if not resultado or not _valid_size_header(destino, meta['bytes']):
            size = destino.stat().st_size if destino.exists() else 0
            raise RuntimeError(f'Descarga incompleta de {nombre}: {size} bytes recibidos, '
                               f'se esperan {meta["bytes"]}. No se usará ese archivo para entrenamiento. '
                               'Puedes volver a ejecutar el comando para reanudarla.')
        estado[nombre] = destino
        print(f'[DRIVE] OK: {nombre}', flush=True)

    # MD5 de anotaciones publicadas por los autores (3 MB). El MD5 de imágenes
    # 6.5 GB puede verificarse aparte; evita segunda lectura costosa por defecto.
    anotaciones=estado['eccv_18_annotations.tar.gz']
    if RAW_FILES['eccv_18_annotations.tar.gz'].get('md5'):
        md5=hashlib.md5(anotaciones.read_bytes()).hexdigest()
        if md5 != RAW_FILES['eccv_18_annotations.tar.gz']['md5']:
            raise ValueError(f'Anotaciones corruptas/alteradas; MD5 obtenido: {md5}')
    # Verificar accesibilidad de las anotaciones TAR, sin extraer 6.5 GB de fotos.
    with tarfile.open(estado['eccv_18_annotations.tar.gz'], 'r:gz') as tar:
        nombres = {Path(m.name).name for m in tar.getmembers() if m.isfile()}
        esperados = {f'{s}_annotations.json' for s in
                     ('train', 'cis_val', 'trans_val', 'cis_test', 'trans_test')}
        if not esperados.issubset(nombres):
            raise ValueError(f'El TAR de anotaciones no incluye: {sorted(esperados - nombres)}')
    return estado

PUBLIC_WEIGHTS = {
    'YOLOv8n': {'path': 'final_l4_v1/weights/best.pt',
                'id': '10x6GD26z6l22pSvvjcQ57vz6gDl4wZij', 'bytes': 6495786},
    'YOLOv8s': {'path': 'comparacion_yolov8s_l4/weights/best.pt',
                'id': '1cmHUbZ1aue4uj4ku0Q8p9eANZprR9f9o', 'bytes': 22512746},
}


def descargar_pesos_publicos(runs: Path, *, download_fn=None):
    """Reutiliza checkpoints preentrenados: NO reproduce el entrenamiento."""
    if download_fn is None:
        import gdown
        download_fn = gdown.download
    runs=Path(runs)
    for nombre, meta in PUBLIC_WEIGHTS.items():
        path = runs / meta['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and path.stat().st_size == meta['bytes']:
            print(f'[PESOS] Checkpoint válido por tamaño: {nombre}',flush=True)
            continue
        if path.is_file():
            raise ValueError(f'{nombre}: hay un checkpoint distinto en {path}; no se sobrescribe.')
        print(f'[PESOS] Descargando checkpoint público de {nombre} (no reentrenado)...',flush=True)
        try:
            downloaded=download_fn(id=meta['id'],output=str(path),quiet=False,resume=True,retries=3)
        except Exception as exc:
            raise RuntimeError(f'No se pudo descargar {nombre}. Comprueba acceso público: '
                               f'https://drive.google.com/file/d/{meta["id"]}/view') from exc
        if not downloaded or not path.is_file() or path.stat().st_size != meta['bytes']:
            raise RuntimeError(f'Checkpoint {nombre} incompleto; repite para reanudar.')
    return {n:runs/meta['path'] for n,meta in PUBLIC_WEIGHTS.items()}
