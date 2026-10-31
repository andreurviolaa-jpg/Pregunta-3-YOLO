"""Figuras finales solo desde CSV: ninguna figura requiere reentrenar."""
from __future__ import annotations
from pathlib import Path
import pandas as pd


def generar_figuras(results: Path, destino: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    results,destino=Path(results),Path(destino)
    destino.mkdir(parents=True,exist_ok=True)
    guardados=[]
    def salvar(fig,nombre):
        path=destino/nombre
        fig.savefig(path,dpi=220,bbox_inches='tight')
        plt.close(fig)
        guardados.append(path)
    data=results/'comparacion_yolov8n_vs_yolov8s_test.csv'
    if data.exists():
        df=pd.read_csv(data)
        fig,ax=plt.subplots(figsize=(7.4,4.3))
        x=np.arange(2)
        for j,modelo in enumerate(('YOLOv8n','YOLOv8s')):
            sub=df[df.Modelo==modelo].set_index('Particion').reindex(['cis_test','trans_test'])
            ax.bar(x+(j-.5)*.36,sub['mAP50']*100,.35,label=modelo)
        ax.set_xticks(x, ['CIS TEST','TRANS TEST']); ax.set_ylabel('mAP@0.5 (%)');ax.set_ylim(0,100)
        ax.set_title('Generalización: ubicaciones conocidas vs. nuevas');ax.legend()
        fig.tight_layout();salvar(fig,'01_map_cis_trans.png')
    data=results/'generalizacion_por_especie.csv'
    if data.exists():
        df=pd.read_csv(data)
        sub=df[df.Modelo=='YOLOv8s'].sort_values('Brecha_mAP50_pp')
        fig,ax=plt.subplots(figsize=(8.4,6.2))
        ax.barh(sub['Especie'],sub['Brecha_mAP50_pp'])
        ax.invert_yaxis();ax.axvline(0,color='gray',lw=.8)
        ax.set_xlabel('Brecha AP@0.5 (TRANS - CIS), puntos porcentuales')
        ax.set_title('YOLOv8s: cambio geográfico por especie')
        fig.tight_layout();salvar(fig,'02_brecha_especies.png')
    d=results/'diagnosticos_reproducibles'
    data=d/'resumen_tamano_YOLOv8s.csv'
    if data.exists():
        df=pd.read_csv(data)
        order=['Pequeno (<1%)','Mediano (1-<10%)','Grande (>=10%)']
        x=np.arange(len(order));fig,ax=plt.subplots(figsize=(8,4.3))
        for j,split in enumerate(('cis_test','trans_test')):
            sub=df[df.Particion==split].set_index('Tamano').reindex(order)
            bars=ax.bar(x+(j-.5)*.36,sub['Recall_pct'],.35,label=split)
            for bar,n in zip(bars,sub['Casos']):
                if pd.notna(n):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f'n={int(n)}',ha='center',fontsize=8)
        ax.set_xticks(x,['Pequeño','Mediano','Grande']);ax.set_ylim(0,110)
        ax.set_ylabel('Recall a conf=0.25, IoU>=0.50 (%)');ax.set_title('YOLOv8s: recall según tamaño relativo')
        ax.legend();fig.tight_layout();salvar(fig,'03_tamano_yolov8s.png')
    data=d/'resumen_oclusion_modelos.csv'
    if data.exists():
        df=pd.read_csv(data)
        sub=df.groupby(['Modelo','Oclusion'],as_index=False).agg(Casos=('Casos','sum'),Detectados=('Detectados','sum'))
        sub['Recall_pct']=100*sub.Detectados/sub.Casos
        niv=[n for n in ['Baja','Media','Alta'] if n in set(sub.Oclusion)]
        x=np.arange(len(niv));fig,ax=plt.subplots(figsize=(7.8,4.2))
        for j,mod in enumerate(('YOLOv8n','YOLOv8s')):
            filas=sub[sub.Modelo==mod].set_index('Oclusion').reindex(niv)
            bars=ax.bar(x+(j-.5)*.36,filas['Recall_pct'],.35,label=mod)
            for b,n in zip(bars,filas['Casos']):
                if pd.notna(n):ax.text(b.get_x()+b.get_width()/2,b.get_height()+1,f'n={int(n)}',ha='center',fontsize=8)
        ax.set_xticks(x,niv);ax.set_ylim(0,110);ax.set_ylabel('Recall (%)')
        ax.set_title('Oclusión: muestra auditada manualmente')
        ax.legend();fig.tight_layout();salvar(fig,'04_oclusion_auditada.png')
    data=d/'resumen_fondo_yolov8s_matching_1a1.csv'
    if data.exists():
        df=pd.read_csv(data)
        x=np.arange(len(df));fig,ax=plt.subplots(figsize=(7.5,4.3))
        ax.bar(x-.18,df.Recall_original_pct,.35,label='Original')
        ax.bar(x+.18,df.Recall_alterado_pct,.35,label='Fondo desenfocado')
        ax.set_xticks(x,df.Particion);ax.set_ylim(0,100)
        ax.set_ylabel('Recall (%)');ax.set_title('Sensibilidad contextual, GT enmascarada')
        ax.legend();fig.tight_layout();salvar(fig,'05_sensibilidad_fondo.png')
    data=d/'falsas_alarmas_vacios_YOLOv8s.csv'
    if data.exists():
        df=pd.read_csv(data)
        fig,ax=plt.subplots(figsize=(7,4))
        ax.bar(df.Particion,df.Porcentaje_vacias_con_alarma)
        ax.set_ylim(0,100);ax.set_ylabel('Fotografías negativas con alguna detección (%)')
        ax.set_title('YOLOv8s: falsas alarmas en imágenes vacías')
        fig.tight_layout();salvar(fig,'06_falsas_alarmas.png')
    eco=results/'demo_ecologica_reproducible'/'reportes'/'resumen_especies.csv'
    if eco.exists():
        df=pd.read_csv(eco)
        if len(df):
            df=df.sort_values('Eventos_secuencia',ascending=False).head(12)
            fig,ax=plt.subplots(figsize=(8.2,4.5))
            ax.bar(df.especie,df.Eventos_secuencia)
            ax.set_ylabel('Secuencias con especie predicha')
            ax.set_title('Cámara TRANS: actividad ecológica por especie')
            ax.tick_params(axis='x',rotation=35);fig.tight_layout();salvar(fig,'07_demo_ecologia.png')
    print(f'[FIGURAS] {len(guardados)} archivos en {destino}',flush=True)
    return guardados
