"""Módulo reproducible de monitoreo ecológico basado en YOLOv8.

Uso:
    python modulo_ecologico_yolov8s.py --images CARPETA --weights best.pt \
        --output RESULTADOS --metadata metadata.csv --cpu

Metadatos opcionales: file_name, location, seq_id y date_captured.
Se admite fecha_hora en lugar de date_captured. Sin metadatos de secuencia,
se considera cada fotografía un evento distinto, marcado como estimación.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DETECTIONS_COLS = [
    "file_name", "location", "seq_id", "fecha_hora", "fecha", "mes", "hora",
    "especie", "confianza", "x1", "y1", "x2", "y2", "modelo",
]
PHOTOS_COLS = [
    "file_name", "location", "seq_id", "fecha_hora", "fecha", "mes", "hora",
    "n_detecciones", "con_fauna_predicha", "modelo", "secuencia_inferida",
]


def guardar_csv_atomico(df: pd.DataFrame, destino: Path) -> None:
    temporal = destino.with_suffix(destino.suffix + ".tmp")
    df.to_csv(temporal, index=False)
    os.replace(temporal, destino)


def imagenes_en_carpeta(carpeta: Path) -> list[Path]:
    if not carpeta.is_dir():
        raise FileNotFoundError(f"No existe la carpeta: {carpeta}")
    imagenes = sorted(p for p in carpeta.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not imagenes:
        raise ValueError(f"No se encontraron fotografías en {carpeta}")
    nombres = [p.name for p in imagenes]
    if len(nombres) != len(set(nombres)):
        raise ValueError("Hay nombres de imagen duplicados en subcarpetas. Renómbralos antes de continuar.")
    return imagenes


def preparar_metadata(imagenes: list[Path], metadata_csv: Path | None, camara: str) -> pd.DataFrame:
    nombres = [p.name for p in imagenes]
    if metadata_csv:
        meta = pd.read_csv(metadata_csv, dtype={"file_name": str, "seq_id": str, "location": str})
        if "file_name" not in meta.columns:
            raise ValueError("El archivo de metadatos necesita una columna 'file_name'.")
        meta["file_name"] = meta["file_name"].map(lambda s: Path(str(s)).name)
        if meta["file_name"].duplicated().any():
            raise ValueError("El CSV de metadatos contiene file_name duplicados.")
        faltantes = sorted(set(nombres) - set(meta["file_name"]))
        if faltantes:
            raise ValueError(f"Faltan metadatos para {len(faltantes)} imágenes, por ejemplo: {faltantes[:3]}")
        meta = meta.set_index("file_name").loc[nombres].reset_index()
    else:
        meta = pd.DataFrame({"file_name": nombres})

    if "location" not in meta.columns:
        meta["location"] = camara
    else:
        meta["location"] = meta["location"].fillna(camara).replace("", camara).astype(str)

    if "seq_id" not in meta.columns:
        meta["seq_id"] = ""
    meta["seq_id"] = meta["seq_id"].fillna("").astype(str)
    meta["secuencia_inferida"] = meta["seq_id"].eq("")
    # Cuando no hay secuencia anotada, no agrupar arbitrariamente fotografías.
    meta.loc[meta["secuencia_inferida"], "seq_id"] = meta.loc[
        meta["secuencia_inferida"], "file_name"
    ].map(lambda x: "individual_" + x)

    columna_fecha = next(
        (x for x in ["date_captured", "fecha_hora", "datetime"] if x in meta.columns), None
    )
    meta["fecha_hora"] = pd.to_datetime(meta[columna_fecha], errors="coerce") if columna_fecha else pd.NaT
    meta["fecha"] = meta["fecha_hora"].dt.strftime("%Y-%m-%d")
    meta["mes"] = meta["fecha_hora"].dt.strftime("%Y-%m")
    meta["hora"] = meta["fecha_hora"].dt.hour
    return meta[["file_name", "location", "seq_id", "fecha_hora", "fecha", "mes", "hora", "secuencia_inferida"]]


def crear_reportes(fotos: pd.DataFrame, detecciones: pd.DataFrame, carpeta: Path) -> dict[str, pd.DataFrame]:
    if fotos.empty:
        raise ValueError("No hay fotografías procesadas.")
    fotos = fotos.copy()
    detecciones = detecciones.copy()
    resumen_camaras = fotos.groupby("location", as_index=False).agg(
        Fotografias=("file_name", "nunique"),
        Fotos_con_deteccion=("con_fauna_predicha", "sum"),
        Cajas_detectadas=("n_detecciones", "sum"),
        Secuencias=("seq_id", "nunique"),
    )
    # Número de especies predichas; no equivale a riqueza real validada.
    n_especies = detecciones.groupby("location")["especie"].nunique().reset_index(name="Especies_predichas")
    resumen_camaras = resumen_camaras.merge(n_especies, on="location", how="left")
    resumen_camaras["Especies_predichas"] = resumen_camaras["Especies_predichas"].fillna(0).astype(int)
    resumen_camaras["Fotos_sin_deteccion"] = (
        resumen_camaras["Fotografias"] - resumen_camaras["Fotos_con_deteccion"]
    )
    resumen_camaras["Porcentaje_sin_deteccion"] = (
        100 * resumen_camaras["Fotos_sin_deteccion"] / resumen_camaras["Fotografias"]
    ).round(2)
    resumen_camaras["Porcentaje_con_deteccion"] = (
        100 * resumen_camaras["Fotos_con_deteccion"] / resumen_camaras["Fotografias"]
    ).round(2)

    eventos = detecciones.groupby(["location", "seq_id", "especie"], as_index=False).agg(
        Primera_deteccion=("fecha_hora", "min"),
        Ultima_deteccion=("fecha_hora", "max"),
        Fotografias_con_especie=("file_name", "nunique"),
        Cajas_detectadas=("especie", "size"),
        Confianza_maxima=("confianza", "max"),
    )
    especies = detecciones.groupby(["location", "especie"], as_index=False).agg(
        Fotografias_con_especie=("file_name", "nunique"),
        Cajas_detectadas=("especie", "size"),
        Confianza_media=("confianza", "mean"),
    )
    num_eventos = eventos.groupby(["location", "especie"]).size().reset_index(name="Eventos_secuencia")
    especies = especies.merge(num_eventos, on=["location", "especie"], how="left")
    diario = detecciones.dropna(subset=["fecha"]).groupby(
        ["location", "fecha", "especie"], as_index=False
    ).agg(Fotografias_con_especie=("file_name", "nunique"), Cajas_detectadas=("especie", "size"))
    mensual = detecciones.dropna(subset=["mes"]).groupby(
        ["location", "mes", "especie"], as_index=False
    ).agg(Fotografias_con_especie=("file_name", "nunique"), Cajas_detectadas=("especie", "size"))
    horario = detecciones.dropna(subset=["hora"]).groupby(
        ["location", "hora", "especie"], as_index=False
    ).agg(Fotografias_con_especie=("file_name", "nunique"))

    reportes = {
        "resumen_camaras.csv": resumen_camaras,
        "resumen_especies.csv": especies,
        "eventos_secuencia.csv": eventos,
        "reporte_diario.csv": diario,
        "reporte_mensual.csv": mensual,
        "reporte_horario.csv": horario,
    }
    for nombre, tabla in reportes.items():
        guardar_csv_atomico(tabla, carpeta / nombre)
    return reportes


def ejecutar_monitoreo(
    carpeta_imagenes: str | Path,
    pesos: str | Path,
    salida: str | Path,
    metadata_csv: str | Path | None = None,
    camara: str = "camara_nueva",
    conf: float = 0.25,
    imgsz: int = 640,
    cpu: bool = True,
    limite: int | None = None,
    modelo_nombre: str = "YOLOv8s_final",
) -> dict[str, pd.DataFrame]:
    """Genera y persiste fotos, detecciones y reportes con reanudación segura.

    No es un conteo fiable de individuos: los eventos son presencias por secuencia.
    Sin fechas, los reportes por periodos estarán vacíos en lugar de inventar fechas.
    """
    import torch
    from ultralytics import YOLO

    carpeta_imagenes = Path(carpeta_imagenes).expanduser().resolve()
    pesos = Path(pesos).expanduser().resolve()
    salida = Path(salida).expanduser().resolve()
    metadata_csv = Path(metadata_csv).expanduser().resolve() if metadata_csv else None
    if not pesos.is_file():
        raise FileNotFoundError(f"No existen los pesos: {pesos}")
    if metadata_csv and not metadata_csv.is_file():
        raise FileNotFoundError(f"No existe el CSV: {metadata_csv}")
    if limite is not None and limite < 1:
        raise ValueError("limite debe ser >= 1")
    imagenes = imagenes_en_carpeta(carpeta_imagenes)
    if limite:
        imagenes = imagenes[:limite]
    meta = preparar_metadata(imagenes, metadata_csv, camara)
    map_meta = meta.set_index("file_name").to_dict("index")
    salida.mkdir(parents=True, exist_ok=True)
    fotos_path = salida / "fotografias.csv"
    det_path = salida / "detecciones.csv"
    config_path = salida / "configuracion.json"
    configuracion = {
        "carpeta_imagenes": str(carpeta_imagenes),
        "pesos": str(pesos), "tamano_pesos_bytes": pesos.stat().st_size,
        "modificado_pesos_ns": pesos.stat().st_mtime_ns,
        "metadata": str(metadata_csv) if metadata_csv else None,
        "conf": conf, "imgsz": imgsz, "modelo": modelo_nombre,
    }
    if config_path.is_file():
        anterior = json.loads(config_path.read_text(encoding="utf-8"))
        if anterior != configuracion:
            raise ValueError("Esta carpeta contiene resultados de otra configuración. Use una carpeta de salida distinta.")
    else:
        if fotos_path.exists() or det_path.exists():
            raise ValueError("Ya existen CSV sin configuracion.json; use una carpeta limpia para evitar mezclas.")
        config_path.write_text(json.dumps(configuracion, ensure_ascii=False, indent=2), encoding="utf-8")

    fotos = pd.read_csv(fotos_path, dtype={"file_name": str, "seq_id": str, "location": str}) if fotos_path.exists() else pd.DataFrame(columns=PHOTOS_COLS)
    detecciones = pd.read_csv(det_path, dtype={"file_name": str, "seq_id": str, "location": str}) if det_path.exists() else pd.DataFrame(columns=DETECTIONS_COLS)
    hechos = set(fotos["file_name"])
    # Tras una interrupción entre las dos escrituras, solo reutilizar
    # detecciones de fotografías cuyo registro fue confirmado.
    detecciones = detecciones[detecciones["file_name"].isin(hechos)].copy().reset_index(drop=True)
    pendientes = [p for p in imagenes if p.name not in hechos]
    print(f"Imágenes: {len(imagenes)}; terminadas: {len(hechos)}; pendientes: {len(pendientes)}")

    if pendientes:
        device = "cpu" if cpu else (0 if torch.cuda.is_available() else "cpu")
        print("Inferencia en:", device)
        modelo = YOLO(str(pesos))
        for i, foto in enumerate(pendientes, 1):
            pred = modelo.predict(str(foto), imgsz=imgsz, conf=conf, iou=0.7,
                                  device=device, save=False, verbose=False)[0]
            info = map_meta[foto.name]
            n = len(pred.boxes)
            fila_foto = dict(file_name=foto.name, location=info["location"], seq_id=info["seq_id"],
                             fecha_hora=info["fecha_hora"], fecha=info["fecha"], mes=info["mes"],
                             hora=info["hora"], n_detecciones=n, con_fauna_predicha=int(n > 0),
                             modelo=modelo_nombre, secuencia_inferida=info["secuencia_inferida"])
            fotos.loc[len(fotos)] = fila_foto
            if n:
                cajas = pred.boxes.xyxy.cpu().numpy()
                clases = pred.boxes.cls.cpu().numpy().astype(int)
                confianzas = pred.boxes.conf.cpu().numpy()
                for caja, clase, confianza in zip(cajas, clases, confianzas):
                    fila = dict(file_name=foto.name, location=info["location"], seq_id=info["seq_id"],
                                fecha_hora=info["fecha_hora"], fecha=info["fecha"], mes=info["mes"],
                                hora=info["hora"], especie=modelo.names[int(clase)],
                                confianza=float(confianza), x1=float(caja[0]), y1=float(caja[1]),
                                x2=float(caja[2]), y2=float(caja[3]), modelo=modelo_nombre)
                    detecciones.loc[len(detecciones)] = fila
            if i % 5 == 0 or i == len(pendientes):
                # Primero las detecciones; la lista de fotos confirma qué casos están completos.
                guardar_csv_atomico(detecciones, det_path)
                guardar_csv_atomico(fotos, fotos_path)
                print(f"Procesadas {len(fotos)}/{len(imagenes)}", flush=True)
        del modelo

    reportes = crear_reportes(fotos, detecciones, salida)
    print("Reporte general:")
    print(reportes["resumen_camaras.csv"].to_string(index=False))
    if meta["fecha_hora"].isna().any():
        print("Advertencia: faltan fechas para algunas imágenes. No se infirieron fechas.")
    if meta["secuencia_inferida"].any():
        print("Advertencia: faltan IDs de secuencia para algunas imágenes. Eventos agrupados por foto en esos casos.")
    print("Reportes guardados en:", salida)
    return {"fotografias": fotos, "detecciones": detecciones, **reportes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="Directorio con imágenes de cámara trampa")
    parser.add_argument("--weights", required=True, help="Ruta a best.pt")
    parser.add_argument("--output", required=True, help="Directorio de CSV de salida")
    parser.add_argument("--metadata", default=None, help="CSV file_name, location, seq_id, date_captured")
    parser.add_argument("--camera", default="camara_nueva", help="Identificador por defecto")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--limit", type=int, default=None)
    dispositivos = parser.add_mutually_exclusive_group()
    dispositivos.add_argument("--gpu", action="store_true", help="Opcional: inferencia en GPU")
    dispositivos.add_argument("--cpu", action="store_true", help="Explícitamente CPU; también es el valor por defecto")
    args = parser.parse_args()
    ejecutar_monitoreo(
        carpeta_imagenes=args.images, pesos=args.weights, salida=args.output,
        metadata_csv=args.metadata, camara=args.camera,
        conf=args.conf, imgsz=args.imgsz, limite=args.limit, cpu=not args.gpu,
    )


if __name__ == "__main__":
    main()
