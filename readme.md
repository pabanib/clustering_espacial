# Spatial Clustering and SDEC Project

This repository contains the code for spatial clustering analysis using the SDEC algorithm, with three main case studies presented in the paper.

## Overview

The project includes:
- pre-processing scripts for salary and COVID spatial data
- the SDEC deep clustering implementation under `SDEC/`
- notebooks for analysis and visualization
- a training pipeline to run new SDEC experiments

The notebooks under `analisis/` are the key case studies:
- `analisis/cluster_salarios.ipynb` — salary clustering case
- `analisis/covid.ipynb` — COVID case
- `analisis/DEC.ipynb` — DEC / deep embedding clustering case

These three notebooks correspond to the cases discussed in the presented paper.

## Installation

### Recommended: conda environment

1. Install Miniconda or Anaconda.
2. From the repository root, create the environment:

```powershell
conda env create -f environment.yml
```

3. Activate the environment:

```powershell
conda activate sdec
```

### Alternative: pip

If you prefer pip, install the required packages and then the extra tools:

```powershell
python -m pip install -r requirements.txt
python -m pip install rarfile
```

### System requirement for data download

The data downloader uses RAR archives, so you also need a system RAR extractor:
- Windows: `conda install -c conda-forge unrar`
- Linux: `sudo apt install unrar`
- macOS: `brew install rar`

## Download the data first

Before running the notebooks or training scripts, download the processed data from Zenodo:

```powershell
python download_data.py
```

This script downloads and extracts:
- `covid.rar`
- `salarios.rar`
- `simulaciones.rar`
- `resultados.rar`

It extracts them into the local `datos/procesados/` and root folders, so the notebooks and scripts can run without needing to rebuild the full dataset from scratch.

## Quick Start

1. Activate the environment.
2. Run the data download script.
3. Open one of the notebooks in `analisis/`.

Example:

```powershell
conda activate sdec
python download_data.py
code .
```

Then open:
- `analisis/cluster_salarios.ipynb`
- `analisis/covid.ipynb`
- `analisis/DEC.ipynb`

## How the pipeline works

### Data preparation

Salary preprocessing is implemented in `datos/scripts_preproceso/procesamiento_sal.py` with the class `Salarios`.
The main methods are:
- `Salarios.preparar_datos(variables, poblacion)` — builds the feature matrix from salary data
- `Salarios.generar_matriz(knn)` — generates the spatial weight matrix used by SDEC

COVID preprocessing is available in `datos/scripts_preproceso/lectura_datos_covid.py`.

### Training with SDEC

The training pipeline is implemented in `scripts_entrenamiento/entrenador.py`.
It reads the experiment cases from `scripts_entrenamiento/casos.yaml`, then:
- prepares the data
- builds the spatial weight matrix
- creates the SDEC model
- trains the autoencoder and spatial clustering
- saves results in `salarios/resultados/`

The example case file `scripts_entrenamiento/casos.yaml` includes configurations like:
- `variables`
- `poblacion`
- `n_clusters_d` and `n_clusters_h`
- `semilla`
- `matriz`
- `procesar`

## Using SDEC for new cases

To apply SDEC to a new dataset or a different spatial clustering problem:

1. prepare your data in the same structure used by the preprocessing scripts
   (or adapt `datos/scripts_preproceso/` for the new source format)
2. add a new case entry in `scripts_entrenamiento/casos.yaml`
3. set `procesar: true` for the case you want to run
4. run the training pipeline:

```powershell
python scripts_entrenamiento/entrenador.py
```

If you want to build a custom experiment, you can also write a small script that:
- loads data with `Salarios()` or a similar preprocessing class
- calls `preparar_datos(...)`
- creates `W = self.datos.matriz_W(knn)`
- instantiates `SDEC.regionalizacion.deep_cluster.sdec(...)`
- trains the model and saves the result

## Project structure

Important folders:
- `analisis/` — notebooks for the paper cases and visual analysis
- `datos/` — raw and processed datasets, plus preprocessing scripts
- `scripts_entrenamiento/` — training pipeline and case definitions
- `SDEC/` — spatial deep clustering implementation
- `resultados/` — output files and saved results

## Notes

- The three notebooks under `analisis/` are the paper cases.
- Always download the dataset first with `python download_data.py`.
- Use the `scripts_entrenamiento/casos.yaml` file to define new experiments.
- The repository is organized for repeatability: data download, preprocessing, training, and analysis are separate steps.
