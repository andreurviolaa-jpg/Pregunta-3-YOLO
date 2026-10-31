"""Validación sin Internet del protocolo de descarga pública."""
import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.download import descargar_raw


class TestDrivePublico(unittest.TestCase):
    @staticmethod
    def tar_gzip(anotaciones):
        b = io.BytesIO()
        with tarfile.open(fileobj=b, mode='w:gz') as tar:
            nombres = ([f'{s}_annotations.json' for s in
                        ('train','cis_val','trans_val','cis_test','trans_test')]
                       if anotaciones else ['camaras.jpg'])
            for nombre in nombres:
                contenido = b'{}' if anotaciones else b'fake-jpg'
                info=tarfile.TarInfo('dir/'+nombre)
                info.size=len(contenido)
                tar.addfile(info,io.BytesIO(contenido))
        return b.getvalue()

    def test_descarga_y_reutilizacion_con_tamano_verificado(self):
        a,b=self.tar_gzip(True),self.tar_gzip(False)
        metadatos={'eccv_18_annotations.tar.gz':{'id':'fake-a','bytes':len(a)},
                   'eccv_18_all_images_sm.tar.gz':{'id':'fake-b','bytes':len(b)}}
        calls=[]
        def fake_downloader(*, id, output, **kwargs):
            calls.append(id)
            Path(output).write_bytes(a if id=='fake-a' else b)
            return output
        with tempfile.TemporaryDirectory() as temp:
            with patch('src.download.RAW_FILES',metadatos):
                descargar_raw(Path(temp),download_fn=fake_downloader)
                descargar_raw(Path(temp),download_fn=fake_downloader)
                self.assertEqual(calls,['fake-a','fake-b'])

    def test_no_acepta_html_en_lugar_del_tar(self):
        a=self.tar_gzip(True)
        metadatos={'eccv_18_annotations.tar.gz':{'id':'fake-a','bytes':len(a)},
                   'eccv_18_all_images_sm.tar.gz':{'id':'fake-b','bytes':2}}
        def falso(*,id,output,**kwargs):
            Path(output).write_bytes(a if id=='fake-a' else b'<!')
            return output
        with tempfile.TemporaryDirectory() as temp:
            with patch('src.download.RAW_FILES',metadatos):
                with self.assertRaises(RuntimeError):
                    descargar_raw(Path(temp),download_fn=falso)

if __name__=='__main__':
    unittest.main()
