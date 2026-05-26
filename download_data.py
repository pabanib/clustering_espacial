"""
Script para descargar los datos del paper desde Zenodo.
Ejecutar desde la raíz del proyecto: python download_data.py

Requiere: pip install rarfile
Windows: conda install -c conda-forge unrar
Linux:   sudo apt install unrar
Mac:     brew install rar
"""

import os
import urllib.request
import rarfile

ZENODO_ID = "20381447"
BASE_URL = f"https://zenodo.org/records/{ZENODO_ID}/files"

ARCHIVOS = {
    "covid.rar":        "datos/procesados/",
    "salarios.rar":     "datos/procesados/",
    "simulaciones.rar": "datos/procesados/",
    "resultados.rar":   ".",
}


def descargar(nombre, destino):
    url = f"{BASE_URL}/{nombre}?download=1"
    path_rar = os.path.join(destino, nombre)
    os.makedirs(destino, exist_ok=True)

    print(f"⬇️  Descargando {nombre}...")
    urllib.request.urlretrieve(url, path_rar)
    print(f"   ✅ Descargado")
    return path_rar


def descomprimir(path_rar, destino):
    print(f"📦 Descomprimiendo {os.path.basename(path_rar)}...")
    with rarfile.RarFile(path_rar) as rf:
        rf.extractall(destino)
    os.remove(path_rar)
    print(f"   ✅ Listo en {destino}")


if __name__ == "__main__":
    print("=" * 50)
    print("Descarga de datos - Spatial Deep Embedding Clustering")
    print("Zenodo DOI: 10.5281/zenodo.20381447")
    print("=" * 50)

    for archivo, carpeta in ARCHIVOS.items():
        if os.path.exists(os.path.join(carpeta, archivo)):
            print(f"⏭️  Ya existe: {carpeta} — saltando")
            continue

        path_rar = descargar(archivo, carpeta)
        descomprimir(path_rar, carpeta)

    print("\n✅ Todos los datos están listos.")
    print("   Podés ejecutar los notebooks.")