"""Pruebas del resumen de mAP sin ejecutar inferencia."""
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from src.evaluate import resumir


class TestResumen(unittest.TestCase):
    def test_brecha_solo_especies_comunes(self):
        with tempfile.TemporaryDirectory() as carpeta:
            d = Path(carpeta)
            for modelo in ('YOLOv8n', 'YOLOv8s'):
                prefijo = 'metricas_final' if modelo == 'YOLOv8n' else 'metricas_yolov8s'
                eprefijo = 'metricas_especies' if modelo == 'YOLOv8n' else 'metricas_yolov8s_especies'
                for split, valor in [('cis_test', 0.7), ('trans_test', 0.4)]:
                    pd.DataFrame([dict(Modelo=modelo, Particion=split, mAP50=valor,
                                       mAP50_95=0.2, Precision=0.5, Recall=0.5)]).to_csv(
                        d / f'{prefijo}_{split}.csv', index=False)
                    especies = ['cat', 'dog', 'deer'] if split == 'cis_test' else ['cat', 'dog']
                    pd.DataFrame([dict(Especie=x, mAP50=valor, mAP50_95=0.2)
                                  for x in especies]).to_csv(d / f'{eprefijo}_{split}.csv', index=False)
            globales, detalle, resumen = resumir(d)
            self.assertEqual(len(globales), 4)
            self.assertEqual(len(detalle), 4)
            self.assertTrue((resumen['Especies_comunes'] == 2).all())
            self.assertTrue(((resumen['Brecha_pp'] + 30).abs() < 1e-8).all())
            self.assertTrue((d / 'generalizacion_comparable_13_especies.csv').is_file())

if __name__ == '__main__':
    unittest.main()
