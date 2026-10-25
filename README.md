# Detección de fauna silvestre con YOLOv8 — CCT-20

**Universidad Nacional de Ingeniería (UNI) · Examen de Deep Learning, Pregunta 3 · Grupo 2 (2026-II)**

Proyecto experimental de detección de fauna en imágenes de cámaras trampa de **Caltech Camera Traps (CCT-20)**. Se comparan dos detectores, **YOLOv8n** y **YOLOv8s**, con énfasis en la generalización de cámaras conocidas (**CIS TEST**) a cámaras no vistas durante entrenamiento (**TRANS TEST**), robustez y monitoreo ecológico.

## Archivos principales

- [`Pregunta_3_YOLO_Grupo_2.ipynb`](Pregunta_3_YOLO_Grupo_2.ipynb): notebook experimental en **Google Colab**, con exploración, preparación opcional de CCT-20, entrenamiento opcional, lectura de métricas TEST, análisis de robustez, sensibilidad al fondo, reportes y figuras.
- [`modulo_ecologico_yolov8s.py`](modulo_ecologico_yolov8s.py): programa independiente para inferencia sobre fotografías de una cámara y generación de reportes CSV.
- [`requirements.txt`](requirements.txt): dependencias Python.

> **Estado de reproducibilidad:** el módulo de monitoreo es ejecutable desde la línea de comandos, siempre que se proporcionen sus fotografías y los pesos entrenados. El notebook experimental requiere los archivos originales de CCT-20 y los artefactos históricos en la estructura de Google Drive descrita abajo. Los pesos finales, imágenes originales y CSV de resultados históricos **no están incluidos** en este repositorio por ahora. Por ello, publicar este código **no equivale todavía** a una reproducción integral de todos los experimentos desde cero.

## Resultados principales observados

| Modelo | Partición | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---:|---:|
| YOLOv8n | CIS TEST | 0.7107 | 0.4565 |
| YOLOv8s | CIS TEST | **0.7247** | **0.4712** |
| YOLOv8n | TRANS TEST | 0.3759 | 0.2305 |
| YOLOv8s | TRANS TEST | **0.4396** | **0.2742** |

Para calcular una brecha geográfica comparable se toma la intersección de **13 especies** presentes en ambos TEST: la brecha macro de mAP@0.5 es aproximadamente **−35.03 puntos porcentuales para YOLOv8n** y **−29.98 para YOLOv8s**. Los mAP globales de las particiones y estas brechas por especies comunes corresponden a dos agregaciones diferentes.

El análisis manual de oclusión revisó **120 instancias**, de las cuales **94 fueron confirmadas y evaluables** (84 con oclusión baja y 10 con oclusión media). No se confirmaron instancias de oclusión alta; los porcentajes por oclusión describen **esta muestra auditada**, no toda la distribución TEST. Las figuras de robustez por tamaño proceden de YOLOv8n, salvo que se indique lo contrario.

## 1. Preparar el entorno

Se recomienda usar Google Colab para el notebook y Python 3.10 o superior para el módulo de monitoreo.

```bash
python -m pip install -r requirements.txt
```

Para inferencia con GPU se requiere una instalación compatible de PyTorch y los controladores correspondientes; **la CPU también funciona** para el módulo ecológico.

## 2. Ejecutar el monitoreo ecológico con un comando

Se necesitan:

- Una carpeta local `imagenes/` con fotografías JPG, PNG u otros formatos admitidos.
- El checkpoint entrenado **YOLOv8s** (`best.pt`), suministrado por el equipo.
- Opcionalmente, `metadata.csv` para agrupar secuencias y registrar periodos.

```bash
python modulo_ecologico_yolov8s.py --images ./imagenes --weights ./best.pt --output ./salida_ecologica --metadata ./metadata.csv --cpu
```

Si **no tienes `metadata.csv`**, omite completamente `--metadata ./metadata.csv`:

```bash
python modulo_ecologico_yolov8s.py --images ./imagenes --weights ./best.pt --output ./salida_ecologica --cpu
```

El programa procesa las imágenes, guarda progreso para reanudación y produce:

- `fotografias.csv`: detección por fotografía.
- `detecciones.csv`: cajas, especies y confianzas predichas.
- `resumen_camaras.csv`: porcentaje de fotografías con/sin detección.
- `resumen_especies.csv`: detecciones y eventos por secuencia.
- `eventos_secuencia.csv`, `reporte_diario.csv`, `reporte_mensual.csv`, `reporte_horario.csv`.
- `configuracion.json`: configuración de la ejecución.

**Formato opcional de metadatos** (`metadata.csv`):

```csv
file_name,location,seq_id,date_captured
foto_001.jpg,camara_46,secuencia_001,2018-05-01 06:30:00
foto_002.jpg,camara_46,secuencia_001,2018-05-01 06:30:05
```

Sin `date_captured` **no se inventan fechas**; sin `seq_id`, cada fotografía se considera provisionalmente una secuencia distinta. Una *detección* tampoco equivale necesariamente a un individuo diferente. Las fotografías sin detección **no deben interpretarse automáticamente como fotografías sin fauna real**.

**Demostración ya ejecutada en CPU:** cámara TRANS TEST 46, 12 fotografías, 10 con detecciones, 12 cajas, 8 clases predichas, 16.67 % sin detección. Es una prueba funcional, no una medición adicional de exactitud frente a etiquetas reales.

## 3. Reabrir el notebook experimental en Colab

1. Abre `Pregunta_3_YOLO_Grupo_2.ipynb` desde GitHub en Google Colab.
2. Coloca en `MyDrive/examen_yolo/raw/` los archivos originales autorizados: `eccv_18_annotations.tar.gz` y `eccv_18_all_images_sm.tar.gz`.
3. El notebook utiliza `MyDrive/examen_yolo/runs/` para checkpoints y `MyDrive/examen_yolo/resultados_analisis/` para CSV históricos, que deben aportarse por separado si solo se quieren reproducir las figuras.
4. Por defecto, mantén:

```python
PREPARAR_DATASET = False
ENTRENAR_MODELOS = False
RECALCULAR_TEST = False
RECALCULAR_OCLUSION = False
RECALCULAR_FONDO = False
GENERAR_FIGURAS = True
```

**Importante:** con estas opciones se regeneran figuras **solo cuando los CSV históricos y modelos que el notebook requiere ya existen**. Para reconstruir las particiones o reentrenar se deben habilitar las opciones correspondientes y disponer del hardware y los datos necesarios. Los entrenamientos históricos utilizaron Ultralytics `8.4.184`, 20 épocas máximas, imágenes de 640 px, batch 16 y optimizador AdamW.

## 4. Limitaciones y uso responsable

- La generalización a cámaras geográficamente nuevas presenta una caída considerable del mAP.
- El diagnóstico de sensibilidad al fondo revela dependencia del contexto visual en algunos ejemplos, **sin probar por sí mismo memorización del escenario**.
- No hay suficientes ejemplos confirmados de oclusión alta para estimar robustez en ese subgrupo.
- Los falsos negativos pueden ocultar registros de especies sensibles; los falsos positivos aumentan la carga de verificación por especialistas.
- Los resultados deben ser supervisados por expertos en monitoreo ecológico; no constituyen censos automáticos de individuos.

## 5. Datos, créditos y distribución

**Conjunto de datos:** Caltech Camera Traps / ECCV 2018. Respeta las condiciones y atribuciones originales al obtener y utilizar imágenes y anotaciones. Este repositorio no redistribuye el conjunto completo.

**Licencia del código:** todavía no se ha especificado. La disponibilidad pública del repositorio no concede automáticamente permisos de reutilización. El equipo debe escoger una licencia cuando corresponda.

**Pendiente para cumplir la reproducibilidad integral del examen:** publicar los scripts independientes de preparación, entrenamiento y evaluación, y documentar un único comando orquestador que reconstruya todos los resultados desde las fuentes y checkpoints definidos. Esto se incorporará sin modificar ni inventar las métricas históricas.

