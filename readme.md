# Clustering Espacial de Salarios

Este proyecto implementa un pipeline para el procesamiento, análisis y clustering espacial de datos salariales en Argentina, utilizando modelos clásicos y deep clustering (SDEC). La estructura está pensada para separar el procesamiento, el entrenamiento y la exploración de resultados.

---

## Estructura del proyecto

```
clustering_espacial/
│
├── salarios/
│   ├── procesamiento_sal.py      # Clase para cargar y procesar datos salariales
│   ├── entrenador.py             # Script para entrenar modelos y guardar resultados
│   ├── casos.yaml                # Configuración de experimentos/casos a correr
│   └── resultados/
│       └── resultados_*.pkl      # Resultados de los experimentos
│
├── SDEC/
│   ├── regionalizacion/
│   ├── datos/
│   └── ...                      # Código base de clustering espacial y deep clustering
│
└── notebooks/
    └── exploracion.ipynb        # Notebooks para análisis y visualización
```

---

## Flujo de trabajo

### 1. **Definir experimentos**

Edita el archivo [`salarios/casos.yaml`](salarios/casos.yaml) para agregar o modificar los experimentos a correr.  
Ejemplo de entrada:

```yaml
casos:
  - nombre: masa_puestos
    variables: [masa_cer, puestos]
    poblacion: personas
    n_clusters_d: 30
    n_clusters_h: 31
    semilla: 42
    matriz: 4
    procesar: true
```

- `procesar: true` indica qué caso se ejecutará.

---

### 2. **Entrenamiento de modelos**

Corre el script de entrenamiento desde PowerShell (o terminal):

```powershell
conda activate geo
python salarios/entrenador.py
```

Esto:
- Lee los casos activos de `casos.yaml`.
- Procesa los datos.
- Entrena modelos SDEC para los rangos de clusters especificados.
- Guarda los resultados y parámetros en archivos `.pkl` dentro de `salarios/resultados/`.

---

### 3. **Exploración y análisis**

En los notebooks de `notebooks/`:
- Carga los resultados guardados.
- Visualiza mapas, clusters y métricas.
- Calcula y muestra estadísticas por grupo y año.

Ejemplo para cargar resultados:

```python
import pickle
with open('salarios/resultados/resultados_masa_puestos.pkl', 'rb') as f:
    resultados = pickle.load(f)
```

---

## Notas técnicas

- Los parámetros de cada experimento se guardan junto con los resultados para asegurar trazabilidad.
- El pipeline está pensado para ser reproducible y escalable a nuevos experimentos.
- Se recomienda usar entornos virtuales (ej: conda) para asegurar dependencias.

---

## Dependencias principales

- Python 3.8+
- pandas, numpy, geopandas, matplotlib, scikit-learn
- PyYAML
- SDEC (código propio)
- (Opcional) colorcet, cmocean para más escalas de colores

Instalación de dependencias extra:
```sh
pip install pyyaml colorcet cmocean
```

---

## Contacto

Para dudas o mejoras, contacta al autor del repositorio.