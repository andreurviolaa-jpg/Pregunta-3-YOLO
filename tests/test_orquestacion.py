"""El comando 'todo' se puede orquestar sin red, CUDA ni dataset real."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from run_pipeline import main


class TestOrquestacion(unittest.TestCase):
    def test_todo_requiere_autorizacion_o_pesos(self):
        with self.assertRaises(SystemExit) as ctx:
            main(['--etapa','todo','--base','/tmp/no_ejecutar'])
        self.assertEqual(ctx.exception.code, 2)

    def test_todo_con_pesos_publicos_sin_gpu(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            import src.download, src.prepare, src.evaluate
            with patch.object(src.download,'descargar_raw',side_effect=lambda *a,**kw:calls.append('descargar')),\
                 patch.object(src.prepare,'preparar',side_effect=lambda *a,**kw:calls.append('preparar')),\
                 patch.object(src.download,'descargar_pesos_publicos',side_effect=lambda *a,**kw:calls.append('pesos')),\
                 patch.object(src.evaluate,'evaluar',side_effect=lambda *a,**kw:calls.append('evaluar')),\
                 patch.object(src.evaluate,'resumir',side_effect=lambda *a,**kw:calls.append('resumen')):
                main(['--etapa','todo','--base',tmp,'--pesos-publicos','--cpu','--sin-extra'])
            self.assertEqual(calls,['descargar','preparar','pesos','evaluar','resumen'])
            self.assertTrue((Path(tmp)/'resultados_analisis'/'manifiesto_reproduccion.json').is_file())

if __name__=='__main__':unittest.main()
