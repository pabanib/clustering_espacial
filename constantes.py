from pathlib import Path
import os

FILE_DATADIR_PYPROJ = 'C:\\Anaconda3\\envs\\covid\\share\\proj'
DIR_PRINCIPAL = "SDEC//datos"
EPSG = '5345'

BASE_DIR = Path(__file__).resolve().parent

DIR_DATOS = os.path.join(BASE_DIR, "datos")
DIR_RESULTADOS = os.path.join(BASE_DIR, "resultados")