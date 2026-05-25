# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 11:48:02 2021

@author: pabanib

Este archivo trae los datos del COVID de la fuente oficial para Argentina.

"""

import os 
import requests
from zipfile import ZipFile

url =  'https://sisa.msal.gov.ar/datos/descargas/covid-19/files/Covid19Casos.zip'

archivo = requests.get(url)
open('datos/covid.zip', 'wb').write(archivo.content)

with ZipFile('datos/covid19Casos.zip', 'r') as zip:
    zip.extractall('datos')

#os.remove('datos/covid.csv')
