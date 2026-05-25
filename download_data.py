"""
Script para descargar los datos del paper desde Zenodo.
Ejecutar desde la raíz del proyecto: python download_data.py
"""

import os
import urllib.request
import rarfile  # pip install rarfile

ZENODO_ID = "20380822"
BASE_URL = f"https://zenodo.org/records/{ZENODO_ID}/files"

ARCHIVOS = {
    "covid.rar":        "datos/procesados/covid/",
    "salarios.rar":     "datos/procesados/salarios/",
    "simulaciones.rar": "datos/procesados/simulaciones/",
    "resultados.rar":   "resultados/",
}


def descargar(nombre, destino):
    url = f"{BASE_URL}/{nombre}"
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
    os.remove(path_rar)  # borra el .rar una vez descomprimido
    print(f"   ✅ Listo en {destino}")


if __name__ == "__main__":
    print("=" * 50)
    print("Descarga de datos - clustering_espacial")
    print("=" * 50)

    for archivo, carpeta in ARCHIVOS.items():
        # Verificar si ya existe contenido en la carpeta
        if os.path.exists(carpeta) and len(os.listdir(carpeta)) > 0:
            print(f"⏭️  Ya existe: {carpeta} — saltando")
            continue

        path_rar = descargar(archivo, carpeta)
        descomprimir(path_rar, carpeta)

    print("\n✅ Todos los datos están listos.")
    print("   Podés ejecutar los notebooks.")
