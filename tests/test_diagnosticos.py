import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.diagnostics import iou_xyxy, matching_uno_a_uno, tamano, resumen_diagnosticos

class TestDiagnosticos(unittest.TestCase):
    def test_matching_clase_iou_y_no_duplicar(self):
        gt=[{'clase':2,'caja':(.1,.1,.3,.3)},
            {'clase':2,'caja':(.5,.5,.8,.8)}]
        b=np.array([[.1,.1,.3,.3],[.1,.1,.3,.3],[.5,.5,.8,.8],[.5,.5,.8,.8]])
        c=np.array([2,2,2,4])
        conf=np.array([.8,.9,.7,.99])
        self.assertEqual(matching_uno_a_uno(gt,b,c,conf),{0,1})
        self.assertEqual(iou_xyxy(gt[0]['caja'], (.4,.4,.6,.6)),0.0)

    def test_umbral_areas(self):
        self.assertIn('Pequeno',tamano(.0099))
        self.assertIn('Mediano',tamano(.01))
        self.assertIn('Grande',tamano(.10))

    def test_falsas_alarmas_recapituladas(self):
        with tempfile.TemporaryDirectory() as tmp:
            dir=Path(tmp)
            for split in ('cis_test','trans_test'):
                pd.DataFrame([
                    dict(file_name='a.jpg',Particion=split,total_gt=0,cajas_pred=2,con_pred=1,aciertos=0),
                    dict(file_name='b.jpg',Particion=split,total_gt=0,cajas_pred=0,con_pred=0,aciertos=0),
                    dict(file_name='c.jpg',Particion=split,total_gt=1,cajas_pred=1,con_pred=1,aciertos=1),
                ]).to_csv(dir/f'YOLOv8s_{split}_imagenes.csv',index=False)
                pd.DataFrame([dict(file_name='c.jpg',Particion=split,Indice_objeto=0,Clase=0,Area_relativa=.05,Tamano='Mediano (1-<10%)',Caja_sospechosa=False,Detectado=1)]).to_csv(dir/f'YOLOv8s_{split}_objetos.csv',index=False)
            size,alarms=resumen_diagnosticos(dir)
            self.assertTrue((alarms['Fotos_vacias']==2).all())
            self.assertTrue((alarms['Fotos_vacias_con_alarma']==1).all())
            self.assertTrue((alarms['Cajas_falsas_por_100_fotos']==100).all())
            self.assertTrue((size['Recall_pct']==100).all())

if __name__=='__main__':unittest.main()
