# Detección de fauna silvestre con YOLOv8 — CCT-20

**UNI · Redes Neuronales y Aprendizaje Profundo · Pregunta 3 · Grupo 2 · 2026-II**

Autores: Christian André Urviola García; Alex Reynaldo Pomazón Solórzano; Alexander André Chávez Cabana; Andhersson Manuel Salazar Fernández. **El enunciado solicita cinco integrantes; el quinto aún no ha sido identificado.**

Se comparan YOLOv8n y YOLOv8s en el subconjunto ECCV 2018 de *Caltech Camera Traps*, considerando cámaras conocidas (CIS) y ubicaciones no observadas durante el entrenamiento (TRANS). Se analizan AP por especie, tamaño relativo, falsas alarmas, auditoría manual de oclusión, sensibilidad al fondo y un módulo de reporte ecológico.

## Reproducción desde un solo comando

**Requisitos:** Python 3.10 o superior, conexión a Internet, espacio libre suficiente para ~6.5 GB comprimidos más las imágenes extraídas (se recomienda al menos 18 GB) y una GPU **CUDA** para reproducir el entrenamiento completo. Es compatible con T4 o L4 con las versiones correctas de controladores/PyTorch; CPU solo para inferencia o diagnósticos sin entrenamiento. `reproducir.py` instala los paquetes de `requirements.txt` automáticamente.

1. Obtener el código (GitHub → Code → Download ZIP, o clonar el repositorio). Situarse dentro de `Pregunta-3-YOLO/`.
2. **Ejecutar este único comando en una terminal**:

```bash
python reproducir.py --base ./trabajo --gpu
```

**Hace, en orden**:

1. Instala dependencias de `requirements.txt`, incluido `gdown`.
2. Descarga automáticamente los dos TAR **y solo esos dos** desde la carpeta pública [examen_yolo/raw](https://drive.google.com/drive/folders/1hpeqFqK0WNzu6RA5y53Y5suHtmhrIMFs).
3. Verifica tamaño y cabecera GZIP; comprueba también el MD5 oficial del TAR de anotaciones y la presencia de cinco JSON de particiones.
4. Prepara y filtra CCT-20, guarda 48 095 imágenes y sus etiquetas YOLO; guarda también metadatos originales de cámara, fecha y secuencia.
5. Entrena YOLOv8n y YOLOv8s con 20 épocas máximas y la configuración documentada.
6. Evalúa CIS TEST y TRANS TEST, calcula AP por especie y brechas geográficas comparables.
7. Ejecuta diagnóstico de **YOLOv8s** para tamaño de caja y falsas alarmas sobre imágenes vacías.
8. Descarga el CSV de auditoría humana de 120 casos desde la carpeta pública, compara ambos modelos por oclusión y evalúa sensibilidad al fondo con el mismo matching 1:1.
9. Ejecuta monitoreo ecológico con **secuencias multiframe reales** de una cámara TRANS, sus fechas originales y reportes CSV.
10. Exporta figuras y un manifiesto que describe qué se ejecutó.

**Advertencia**: este comando entrena dos modelos y realiza inferencias adicionales sobre decenas de miles de imágenes. Puede consumir bastante tiempo y GPU; no se ha validado todavía una ejecución completa de extremo a extremo con este paquete actualizado. Sus resultados pueden variar según hardware y librerías.

### Alternativa: verificar con pesos finales públicos, SIN reentrenar

```bash
python reproducir.py --base ./trabajo --pesos-publicos --cpu
```

Descarga el dataset y los checkpoints **publicados previamente** (YOLOv8n 6.50 MB y YOLOv8s 22.51 MB), prepara datos, ejecuta TEST y análisis adicionales. **Esta alternativa no reproduce los 20+20 ciclos de entrenamiento**, solo la evaluación. En CPU procesar >31 mil fotografías por modelo es lento. Para omitir los análisis adicionales, agregar `--sin-extra`.

### Comprobar componentes o reanudar por etapas

```bash
python reproducir.py --etapa pruebas --sin-instalar
python reproducir.py --etapa descargar --base ./trabajo
python reproducir.py --etapa preparar --base ./trabajo
python reproducir.py --etapa entrenar --base ./trabajo --gpu
python reproducir.py --etapa evaluar --base ./trabajo --cpu
python reproducir.py --etapa resumen --base ./trabajo
python reproducir.py --etapa diagnosticos --base ./trabajo --cpu
python reproducir.py --etapa ecologia --base ./trabajo --cpu
python reproducir.py --etapa figuras --base ./trabajo --cpu
```

Las descargas incompletas se pueden **reanudar**, y las imágenes ya extraídas se reutilizan. Cuando Ultralytics genera `last.pt` pero no se ha registrado una finalización, el entrenamiento intenta reanudarse; la reanudación no es universal si cambia el entorno o hay un run corrupto. No se sobrescriben automáticamente archivos de ejecución dudosos.

**En Google Colab**, primero clona o abre el proyecto y habilita GPU. Una celda de código utiliza un signo `!`:

```python
!python reproducir.py --base /content/trabajo --gpu
```

Si quieres resultados persistentes entre reinicios de Colab, usa una `--base` alojada en Drive montado, aunque extraer miles de imágenes allí puede ser bastante más lento. En una terminal Bash/PowerShell normal **no** se escribe `!`.

### Descarga pública y permisos

Carpeta original: [examen_yolo](https://drive.google.com/drive/folders/1Mq286eDir-lCFQ4MQ9gbvCUERlpVJ6tq?usp=drive_link). El descargador usa **los ID exactos** de los dos archivos dentro de `raw/`; no descarga recursivamente toda la carpeta de resultados y notebooks del autor.

| Archivo | Tamaño exacto | Uso |
|---|---:|---|
| `eccv_18_annotations.tar.gz` | 2 997 071 bytes | Anotaciones oficiales ECCV 2018 |
| `eccv_18_all_images_sm.tar.gz` | 6 492 615 601 bytes | Fotografías originales |

Para que el docente pueda descargar **sin iniciar sesión**, el propietario debe comprobar en una ventana de incógnito que **ambos archivos** están accesibles con **«Cualquier persona con el enlace → Lector»**. El hecho de ver archivos como propietario no acredita que sean anónimamente descargables. Google Drive puede imponer cuotas, sobre todo al TAR de 6.49 GB. Si falla la descarga, usar la fuente original [LILA BC / Caltech Camera Traps](https://lila.science/datasets/caltech-camera-traps), respetando su licencia y condiciones de redistribución. No se necesitan contraseñas ni tokens.

## Protocolo del benchmark

El CCT-20 general contiene más imágenes; este trabajo utiliza los **57 864 registros** de la partición ECCV 2018 publicada por los autores. Se seleccionan las fotografías elegibles con cajas coherentes y las negativas explícitas; nunca se convierten animales sin bounding box en falsos negativos de entrenamiento.

| Partición | Original | Elegibles | Excluidas |
|---|---:|---:|---:|
| TRAIN | 13 553 | 12 099 | 1 454 |
| CIS VAL | 3 484 | 2 619 | 865 |
| TRANS VAL | 1 725 | 1 594 | 131 |
| CIS TEST | 15 827 | 13 085 | 2 742 |
| TRANS TEST | 23 275 | 18 698 | 4 577 |
| **Total** | **57 864** | **48 095** | **9 769** |

14 clases de fauna, resolución 640, batch 16, AdamW, lr inicial 0.0006, lrf 0.10, cosine schedule, 20 épocas máximas, semilla 0, hsv/mosaic moderados. TRAIN posee **668 fotos solamente con automóvil** como negativos etiquetados sin cajas; no hay vacíos explícitos en TRAIN. La evaluación sí incluye fotografías vacías reales del benchmark. Las métricas informadas abarcan **las imágenes elegibles**, no la colección general de 243 100 imágenes.

**Comprobación geográfica:** CIS = cámaras vistas durante entrenamiento; TRANS = cámaras nuevas. El script conserva IDs oficiales en `dataset/metadatos_imagenes.csv` y produce `dataset/auditoria_camaras.csv` y `dataset/auditoria_solapamientos.csv`. En los datos elegibles se observaron 70 secuencias compartidas TRAIN/CIS VAL, pero ninguna TRAIN/CIS TEST. Para las brechas de AP se usa el conjunto de especies **compartidas** por ambas particiones.

## Resultados de referencia (obtenidos en la ejecución original con L4)

| Modelo | CIS TEST mAP50 | TRANS TEST mAP50 | CIS TEST mAP50-95 | TRANS TEST mAP50-95 |
|---|---:|---:|---:|---:|
| YOLOv8n | 0.7107 | 0.3759 | 0.4565 | 0.2305 |
| YOLOv8s | 0.7247 | 0.4396 | 0.4712 | 0.2742 |

Brechas de 13 especies comparables: YOLOv8n −35.03 pp, YOLOv8s −29.98 pp. Las nuevas métricas por tamaño son **recall de detecciones a conf≥0.25, IoU≥0.50**, no mAP. Para falsas alarmas se informa porcentaje de fotos sin cajas con al menos una predicción y cantidad de predicciones por 100 fotos vacías. El protocolo oficial de Ultralytics no utiliza necesariamente el mismo umbral que esos análisis personalizados.

Auditoría de oclusión: 120 casos marcados manualmente; 94 evaluables (84 bajos, 10 medios, **0 altos**). La ausencia de oclusión alta **sigue siendo una limitación**; el código no inventa esas etiquetas. Los análisis de fondo intervienen artificialmente los píxeles fuera de una caja GT, por lo que muestran sensibilidad al contexto visual, no causalidad definitiva de memorización del fondo.

## Carpeta resultante

```text
trabajo/
├── raw/                 # los 2 TAR públicos
├── dataset/
│   ├── images/{train,cis_val,trans_val,cis_test,trans_test}/
│   ├── labels/{train,cis_val,trans_val,cis_test,trans_test}/
│   ├── data_cis.yaml, data_trans_test.yaml, ...
│   ├── auditoria_preparacion.csv
│   └── metadatos_imagenes.csv
├── runs/                # pesos y registros de entrenamiento
├── resultados_analisis/
│   ├── comparacion_yolov8n_vs_yolov8s_test.csv
│   ├── generalizacion_comparable_13_especies.csv
│   ├── diagnosticos_reproducibles/  # recall tamaño, vacíos, oclusión, fondo
│   ├── oclusion_test/muestra_oclusion_test.csv
│   ├── demo_ecologica_reproducible/reportes/
│   └── manifiesto_reproduccion.json
└── articulo_final/figuras_reproducidas/
```

## Módulo ecológico independiente

```bash
python modulo_ecologico_yolov8s.py --images ./imagenes --weights ./best.pt \
  --output ./salida_ecologica --metadata ./metadata.csv --cpu
```

Se generan reportes por cámara, especie, secuencia y periodo. Sin metadatos originales de fecha/secuencia **no se inventan** periodos ni eventos. Los eventos estiman presencia por secuencia; no identifican individuos. «Sin detección» NO significa «sin fauna real». La reproducción integral selecciona automáticamente una cámara no vista y secuencias multiframe oficiales, para mostrar la agregación ecológica con una pequeña muestra (≤48 fotos).

## Límites y transparencia

- El detector está implementado por Ultralytics (YOLOv8); contribuciones originales del equipo: preparación, filtrado, orquestación, comparaciones, auditoría, intervenciones del fondo y agregación ecológica. **No se afirma haber implementado el backbone YOLO desde cero**.
- El modelo YOLOv8s se eligió tras explorar TEST; la comparación es exploratoria, no una validación ciega estricta.
- El análisis de alta oclusión sigue sin muestra suficiente. No se reemplaza por mala visibilidad ni truncamiento.
- Las pruebas automáticas son **sintéticas** y no equivalen a una reproducción completa: se necesita ejecutar y conservar el manifiesto en un entorno limpio. El acceso anónimo y la descarga de 6.49 GB no se han podido verificar desde aquí.
- Los falsos negativos de especies amenazadas y los falsos positivos requieren revisión humana; el modelo no se ha validado para Amazonía ni Andes peruanos.
- El conjunto original y los pesos se alojan fuera de GitHub. Verificar siempre los derechos de redistribución y atribución. La visibilidad pública del repositorio **no concede automáticamente una licencia**.
