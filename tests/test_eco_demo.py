import tempfile
import unittest
from pathlib import Path
import pandas as pd
from src.eco_demo import preparar_demo

class TestDemoEcologica(unittest.TestCase):
    def test_preferir_secuencias_reales_multiframe(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)/'data';(d/'images'/'trans_test').mkdir(parents=True)
            filas=[]
            for i in range(8):
                name=f'foto_{i}.jpg'
                (d/'images'/'trans_test'/name).write_bytes(b'fake')
                filas.append(dict(file_name=name,Particion='trans_test',location='46',
                                  seq_id='s1' if i<3 else ('s2' if i<6 else 's3'),
                                  date_captured=f'2012-03-24 02:54:{i:02d}'))
            pd.DataFrame(filas).to_csv(d/'metadatos_imagenes.csv',index=False)
            images,metadata,loc=preparar_demo(d,Path(tmp)/'out',max_fotos=8)
            m=pd.read_csv(metadata)
            self.assertEqual(loc,'46')
            self.assertEqual(len(m),8)
            self.assertEqual(m.seq_id.nunique(),3)
            self.assertEqual(len(list(images.glob('*.jpg'))),8)

if __name__=='__main__':unittest.main()
