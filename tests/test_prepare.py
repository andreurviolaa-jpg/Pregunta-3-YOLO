"""Pruebas sintéticas, sin usar datos privados ni consumir GPU."""
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from src.prepare import caja_yolo, seleccionar_dataset, preparar
from src.config import PARTICIONES


class TestCCT20(unittest.TestCase):
    def test_caja_recortada(self):
        self.assertEqual(caja_yolo([90, 20, 30, 40], 100, 100),
                         (0.95, 0.4, 0.1, 0.4))
        self.assertIsNone(caja_yolo(None, 100, 100))

    def test_no_confundir_fauna_sin_caja_con_negativo(self):
        imagenes = [
            dict(id=1, file_name='fauna.jpg', width=100, height=100),
            dict(id=2, file_name='sin_caja.jpg', width=100, height=100),
            dict(id=3, file_name='car.jpg', width=100, height=100),
        ]
        cats = [dict(id=1, name='cat'), dict(id=2, name='car')]
        anns = [
            dict(image_id=1, category_id=1, bbox=[10, 20, 30, 40]),
            dict(image_id=2, category_id=1),
            dict(image_id=3, category_id=2),
        ]
        datos = {'train': dict(images=imagenes, annotations=anns, categories=cats)}
        seleccion, stats = seleccionar_dataset(datos)
        self.assertEqual(set(seleccion), {'fauna.jpg', 'car.jpg'})
        self.assertEqual(seleccion['car.jpg'][1], [])
        self.assertEqual(len(seleccion['fauna.jpg'][1]), 1)
        self.assertEqual(int(stats['sin_cajas_completas'].iloc[0]), 1)

    def test_preparar_archivos_tar_sinteticos(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            a = tmp / 'eccv_18_annotations.tar.gz'
            b = tmp / 'eccv_18_all_images_sm.tar.gz'
            with tarfile.open(a, 'w:gz') as tar:
                for split in PARTICIONES:
                    data = {
                        'images': [dict(id=1, file_name=f'{split}.jpg', width=100, height=100)],
                        'categories': [dict(id=1, name='cat')],
                        'annotations': [dict(image_id=1, category_id=1, bbox=[20, 30, 40, 50])],
                    }
                    bs = json.dumps(data).encode('utf8')
                    mi = tarfile.TarInfo(f'eccv_18_annotations/{split}_annotations.json')
                    mi.size = len(bs)
                    tar.addfile(mi, io.BytesIO(bs))
            with tarfile.open(b, 'w:gz') as tar:
                for split in PARTICIONES:
                    bs = b'fake-image-content-for-io-test'
                    mi = tarfile.TarInfo(f'eccv_18_all_images_sm/{split}.jpg')
                    mi.size = len(bs)
                    tar.addfile(mi, io.BytesIO(bs))
            destino = tmp / 'data'
            preparar(a, b, destino, strict=False)
            self.assertTrue((destino / 'data_trans_test.yaml').is_file())
            self.assertEqual(len(list((destino / 'images' / 'train').glob('*.jpg'))), 1)
            self.assertTrue((destino / 'labels' / 'trans_test' / 'trans_test.txt').is_file())

if __name__ == '__main__':
    unittest.main()
