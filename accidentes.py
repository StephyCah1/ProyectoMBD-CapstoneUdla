import pandas as pd
import re


########### PRE-PROCESAMIENTO DE LOS DATOS ############
# Función para estandarizar cada archivo e incluir el año
def estandarizar_archivo(ruta_archivo):
    try:
        df = pd.read_csv(ruta_archivo, sep=';', encoding='latin1')
    except:
        df = pd.read_csv(ruta_archivo, sep=';', encoding='utf-8')

    # Extraer el año del nombre del archivo 
    match = re.search(r'(\d{4})', ruta_archivo)
    anio = int(match.group(1)) if match else None
    df["anio"] = anio

######## ANEXO 1!!!!!! #####
    # Mapeo de nombres estándar
    mapeo_columnas = {
        'PROVINCIA': 'provincia',
        'CANTÓN': 'canton',
        'CANTÃ\x93N': 'canton',
        'MES': 'mes',
        'DIA': 'dia',
        'HORA': 'hora',
        'CLASE': 'clase',
        'CAUSA': 'causa',
        'ZONA': 'zona',
        'NUM_FALLECIDO': 'fallecidos',
        'NUM_LESIONADO': 'lesionados',
        'TOTAL_VICTIMAS': 'total_victimas',
        'anio': 'anio'  # Año incluido en mapeo
    }

    # Limpiar nombres de columnas
    df.columns = [col.replace('ï»¿', '').replace('"', '').strip() for col in df.columns]
    df = df.rename(columns=mapeo_columnas)

    # Columnas de interes de estudio
    columnas_objetivo = ['provincia', 'canton', 'mes', 'dia', 'hora',
                         'clase', 'causa', 'zona', 'fallecidos',
                         'lesionados', 'total_victimas', 'anio']

    # Agregar columnas faltantes si alguna no está
    for col in columnas_objetivo:
        if col not in df.columns:
            df[col] = pd.NA

    return df[columnas_objetivo]

# Lista de archivos en la misma carpeta del script
archivos = [
    "I_2023.csv", "I_2024.csv",
    "II_2022.csv", "II_2023.csv", "II_2024.csv",
    "III_2022.csv", "III_2023.csv", "III_2024.csv",
    "IV_2022.csv", "IV_2023.csv", "IV_2024.csv"
]

# Unificar todos los archivos
df_unificado = pd.concat([estandarizar_archivo(f) for f in archivos], ignore_index=True)

# Mostrar las primeras filas para verificar
print(df_unificado.head())

#LIMPIEZA DATOS 2022 Y 2023
# Crear un DataFrame combinado de los años 2022 y 2023
df_2022_2023 = df_unificado[df_unificado["anio"].isin([2022, 2023])].copy()
#Revision dataframe 
print("� Revisión general del DataFrame:\n")
df_2022_2023.info()
#Revision de las categorias para revisar columnas
columnas_categoricas = ["provincia", "canton", "clase", "causa", "zona"]

for col in columnas_categoricas:
    print(f"\n� Valores únicos en '{col}':")
    print(df_2022_2023[col].dropna().unique())
#Revision columnas numericas 
columnas_numericas = ["fallecidos", "lesionados", "total_victimas"]
print("\n� Tipos de datos actuales en columnas numéricas:")
print(df_2022_2023[columnas_numericas].dtypes)
#Revision de dia, mes, hora y sus formatos
for col in columnas_numericas:
    print(f"\n� Valores únicos en '{col}' (primeros 10):")
    print(df_2022_2023[col].dropna().unique()[:10])

print("\n� Revisión de columnas de fecha:")

print("\n� Mes:")
print(df_2022_2023["mes"].dropna().unique())

print("\n� Día:")
print(df_2022_2023["dia"].dropna().unique())

print("\n� Hora:")
print(df_2022_2023["hora"].dropna().unique()[:10])  
#Estandarizacion de tildes y caracteres especiales año 2022 y 2023
import unicodedata

# Función para limpiar y corregir texto codificado mal
def limpiar_texto(texto):
    if pd.isnull(texto):
        return texto
    if not isinstance(texto, str):
        texto = str(texto)
    # Normalizar texto para quitar caracteres raros
    texto = unicodedata.normalize("NFKD", texto)
    texto = texto.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
    texto = texto.strip().upper()
    return texto

# Aplicar limpieza a todas las columnas categóricas
columnas_categoricas = ["provincia", "canton", "clase", "causa", "zona", "mes", "dia", "hora"]

for col in columnas_categoricas:
    df_2022_2023[col] = df_2022_2023[col].apply(limpiar_texto)

# Revisar un ejemplo después de limpiar
print(df_2022_2023[["clase", "causa"]].drop_duplicates().head(10))

# Diccionario de correcciones manuales conocidas
correcciones_texto = {
    "PARDIDA DE PISTA": "PERDIDA DE PISTA",  
    
}

# Aplicar la corrección a la columna 'clase'
df_2022_2023["clase"] = df_2022_2023["clase"].replace(correcciones_texto)
# Limpieza completa de las columnas
from difflib import get_close_matches
import unicodedata
import pandas as pd

# Creacion df unificado
df_2022_2023 = df_unificado[df_unificado["anio"].isin([2022, 2023])].copy()

# Función de limpieza
def limpiar_texto(texto):
    if pd.isnull(texto):
        return texto
    if not isinstance(texto, str):
        texto = str(texto)
    texto = unicodedata.normalize("NFKD", texto)
    texto = texto.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
    texto = texto.strip().upper()
    return texto

# Limpiar columnas categóricas
columnas_categoricas = ["provincia", "canton", "clase", "causa", "zona", "mes", "dia", "hora"]
for col in columnas_categoricas:
    df_2022_2023[col] = df_2022_2023[col].apply(limpiar_texto)

# Buscar errores ortográficos con comparación difusa
columnas_a_revisar = ["clase", "causa", "zona", "provincia", "canton", "dia"]
coincidencias_similares = {}

for col in columnas_a_revisar:
    valores = sorted(df_2022_2023[col].dropna().unique())
    revisados = set()
    similares_col = []

    for val in valores:
        if val not in revisados:
            coincidencias = get_close_matches(val, valores, n=5, cutoff=0.85)
            if len(coincidencias) > 1:
                similares_col.append(coincidencias)
                revisados.update(coincidencias)
    
    coincidencias_similares[col] = similares_col

# Mostrar resultados
for col, coincidencias in coincidencias_similares.items():
    print(f"\n� Posibles errores ortográficos en '{col}':")
    for grupo in coincidencias:
        print(" ↔ ".join(grupo))

###### ANEXO 2!!!!! ####
# Diccionarios de correccion por columna
correcciones = {
    "clase": {
        "PARDIDA DE PISTA": "PERDIDA DE PISTA",
        "ESTRELLAMIENTOS": "ESTRELLAMIENTOS",
        "CAADA DE PASAJEROS": "CAIDA PASAJEROS",
        "CHOQUES": "CHOQUES"
    },
    "causa": {
        "EXCESO VELOCIDA": "EXCESO DE VELOCIDAD",
        "NO RESPETA LAS SEAALES DE TRANSITO": "NO RESPETA LAS SEÑALES DE TRANSITO",
        "IMPRUDENCIA DEL PEATAN": "IMPRUDENCIA DEL PEATON",
        "MAL ESTADO DE LA VAA": "MAL ESTADO DE LA VIA",
        "MAL REBASAMIENTO INVADIR CARRIL": "MAL REBASAMIENTO O INVADIR VIA",
        "DAAOS MECANICOS": "DAÑOS MECANICOS",
        "EXCESO VELOCIDAD": "EXCESO DE VELOCIDAD",
        "IMPRUDENCIA  DEL PEATAN": "IMPRUDENCIA DEL PEATON"
    },
    "provincia": {
        "LOS RAOS": "LOS RIOS", 
        "MANABA": "MANABI",
        "BOLAVAR": "BOLIVAR", 
        "SUCUMBAOS": "SUCUMBIOS",
        "CAAAR": "CAÑAR",
        "SANTO DOMINGO DE LOS TSACHILAS": "SANTO DOMINGO"
    },
    "dia": {
        "MIARCOLES": "MIERCOLES"
    },
    "canton": {
        "PIAAS": "PIÑAS",
        "BAAOS DE AGUA SANTA": "BAÑOS DE AGUA SANTA",
        "RUMIAAHUI": "RUMIÑAHUI", 
        "CORONEL MARCELINO MARIDUEAA": "CORONEL MARCELINO MARIDUEÑA", 
        "OAA": "OÑA",
        "CAAAR": "CAÑAR"
    }
    
}

# Aplicar las correcciones definidas por columna
for col, reemplazos in correcciones.items():
    df_2022_2023[col] = df_2022_2023[col].replace(reemplazos)
#Guardar los diccionarios como archivo .json
import json
## Limpieza continuacion
archivo_diccionarios = "diccionario_codificacion_2024.xlsx"
hojas = pd.ExcelFile(archivo_diccionarios).sheet_names
print("📄 Hojas disponibles en el archivo:")
print(hojas)

# Crear df_2024 a partir de df_unificado
# Separar el año 2024 del DataFrame unificado
df_2024 = df_unificado[df_unificado["anio"] == 2024].copy()

# Aplicar limpieza básica al texto en columnas categóricas
columnas_categoricas = ["provincia", "canton", "clase", "causa", "zona", "mes", "dia", "hora"]
for col in columnas_categoricas:
    df_2024[col] = df_2024[col].apply(limpiar_texto)

##ESTANDARIZACIÓN 2024 MEDIANTE DICCIONARIO XLSX

archivo_diccionarios = "diccionario_codificacion_2024.xlsx"

# Crear diccionarios para todas las hojas disponibles
diccionarios = {}
for variable in ["provincia", "clase", "causa", "canton", "dia", "zona", "mes"]:
    df_dic = pd.read_excel(archivo_diccionarios, sheet_name=variable)
    diccionarios[variable] = dict(zip(df_dic["codigo"].astype(str), df_dic["descripcion"]))

# Aplicar mapeos a df_2024
for col, dic in diccionarios.items():
    df_2024[col] = df_2024[col].map(dic)

# Verificar resultados
print("\n✅ Mapeo aplicado a df_2024:")
print(df_2024.head())

# Función para convertir número de hora a formato de texto
def formatear_hora(hora):
    try:
        hora = int(hora)
        return f"{hora:02d}:00 a {hora:02d}:59"
    except:
        return hora  

# Aplicar al df_2024
df_2024["hora"] = df_2024["hora"].apply(formatear_hora)

# Verificar resultados
print("\n✅ Horas estandarizadas en df_2024:")
print(df_2024["hora"].unique()[:10])

# Cargar el archivo Excel con los diccionarios aplicado al año 2023

archivo_diccionarios = "diccionario_codificacion_2024.xlsx"
diccionarios = {}

with pd.ExcelFile(archivo_diccionarios) as xls:
    for hoja in xls.sheet_names:
        df_dic = pd.read_excel(xls, sheet_name=hoja)
        # Asumimos que el diccionario tiene dos columnas: código y descripción
        col_codigo = df_dic.columns[0]
        col_texto = df_dic.columns[1]
        # Convertir a string para evitar errores de tipo
        df_dic[col_codigo] = df_dic[col_codigo].astype(str)
        df_dic[col_texto] = df_dic[col_texto].astype(str)
        diccionarios[hoja] = dict(zip(df_dic[col_codigo], df_dic[col_texto]))

# Aplicar mapeos al df_2023
for col, dic in diccionarios.items():
    if col in df_2022_2023.columns:
        df_2022_2023[col] = df_2022_2023[col].map(dic).fillna(df_2022_2023[col])

# Revision de mapeo
print("\n✅ Revisión después del mapeo:")
for col in columnas_categoricas:
    print(f"🔍 {col}: {df_2022_2023[col].dropna().unique()[:10]}")  


### Unir los años 2022-2023 con el 2024
df_final = pd.concat([df_2022_2023, df_2024], ignore_index=True)

# Verificar el resultado
print("\n✅ Base final unificada:")
print(df_final.head())
print(f"\n🔍 Total de registros: {len(df_final)}")
print(f"🔍 Años en la base: {df_final['anio'].unique()}")

# Filtro para las provincias válidas
df_final['provincia'] = df_final['provincia'].astype(str)
df_final = df_final[df_final['provincia'].str.match(r'^[A-ZÑ\s]+$', na=False)]

# Agrupamiento y gráfico como antes:
df_provincia = df_final.groupby('provincia')[['fallecidos', 'lesionados']].sum().reset_index()

top_provincias = df_provincia.sort_values(by='fallecidos', ascending=False).head(10)

######## PRUEBAS DE NORMALIDAD DE DATOS #######

from scipy.stats import shapiro, normaltest

# Seleccionamos la columna de interés (puede ser 'fallecidos' o 'total_victimas')
columna_analizar = "fallecidos"

# Filtrar solo valores no nulos
datos = df_final[columna_analizar].dropna()

# Shapiro-Wilk 
stat_shapiro, p_shapiro = shapiro(datos.sample(n=5000, random_state=42) if len(datos) > 5000 else datos)
print(f"\n✅ Prueba de Shapiro-Wilk para '{columna_analizar}':")
print(f"Estadístico = {stat_shapiro:.4f}, p-valor = {p_shapiro:.4f}")

# D'Agostino y Pearson (normaltest)
stat_norm, p_norm = normaltest(datos)
print(f"\n✅ Prueba de D'Agostino-Pearson para '{columna_analizar}':")
print(f"Estadístico = {stat_norm:.4f}, p-valor = {p_norm:.4f}")

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Filtrar datos válidos para graficar
datos = df_final["fallecidos"].dropna()

# Histograma + Densidad
plt.figure(figsize=(10,5))
sns.histplot(datos, kde=True, bins=30, color="skyblue")
plt.title("Distribución de Fallecidos con Densidad KDE")
plt.xlabel("Número de Fallecidos")
plt.ylabel("Frecuencia")
plt.show()

# Q-Q Plot
plt.figure(figsize=(6,6))
stats.probplot(datos, dist="norm", plot=plt)
plt.title("Q-Q Plot de Fallecidos")
plt.show()

########### ANALISIS EXPLORATORIO DE DATOS ##########
import matplotlib.pyplot as plt
import seaborn as sns

# Resumen general de la base
print("\n🔍 Resumen general de df_final:")
print(df_final.describe(include='all'))

# Registros por año
print("\n✅ Registros por año:")
print(df_final["anio"].value_counts())

# Registros por provincia
print("\n✅ Registros por provincia:")
print(df_final["provincia"].value_counts().head(10))

# Registros por clase de accidente
print("\n✅ Registros por tipo de accidente (clase):")
print(df_final["clase"].value_counts().head(10))

# Distribución de fallecidos por año
plt.figure(figsize=(8,5))
df_final.groupby("anio")["fallecidos"].sum().plot(kind="bar", color="coral")
plt.title("Total de Fallecidos por Año")
plt.ylabel("Fallecidos")
plt.xlabel("Año")
plt.show()

# Evolución de fallecidos por trimestre mes_num
meses = ['ENERO', 'FEBRERO', 'MARZO', 'ABRIL', 'MAYO', 'JUNIO', 
         'JULIO', 'AGOSTO', 'SEPTIEMBRE', 'OCTUBRE', 'NOVIEMBRE', 'DICIEMBRE']
mes_a_num = {m: i+1 for i, m in enumerate(meses)}
df_final["mes_num"] = df_final["mes"].map(mes_a_num)

df_final["trimestre"] = ((df_final["mes_num"] - 1) // 3 + 1).astype("Int64")

# Gráfico de fallecidos por trimestre y año
df_final.groupby(["anio", "trimestre"])["fallecidos"].sum().unstack().T.plot(kind="bar", figsize=(10,6))
plt.title("Fallecidos por Trimestre y Año")
plt.ylabel("Fallecidos")
plt.xlabel("Trimestre")
plt.legend(title="Año")
plt.show()

# Fallecidos por provincia
plt.figure(figsize=(10,6))
df_final.groupby("provincia")["fallecidos"].sum().sort_values(ascending=False).head(10).plot(kind="bar", color="teal")
plt.title("Top 10 Provincias con más Fallecidos")
plt.ylabel("Fallecidos")
plt.xlabel("Provincia")
plt.show()

#Grafica de Fallecidos vs. Lesionados por año
import matplotlib.pyplot as plt
import seaborn as sns

# Agrupar datos por año
df_anio = df_final.groupby('anio')[['fallecidos', 'lesionados']].sum().reset_index()

# Gráfica de barras comparativa
df_anio.set_index('anio')[['fallecidos', 'lesionados']].plot(kind='bar', stacked=False, figsize=(8,5), color=['#FF5733', '#33C1FF'])

plt.title('Fallecidos vs Lesionados por Año')
plt.ylabel('Número de Personas')
plt.xlabel('Año')
plt.legend(title='Categoría')
plt.tight_layout()
plt.show()

#Grafica de Fallecidos vs. Lesionados por provincia
top_provincias = df_provincia.sort_values(by='fallecidos', ascending=False).head(10)

top_provincias.plot(
    x='provincia',
    y=['fallecidos', 'lesionados'],
    kind='bar',
    figsize=(12, 6),
    color=['#FF5733', '#33C1FF']
)

plt.title('Top 10 Provincias: Fallecidos vs Lesionados')
plt.ylabel('Número de Personas')
plt.xlabel('Provincia')
plt.legend(title='Categoría')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

## Análisis exploratorio clase y causa #####
import matplotlib.pyplot as plt

# Agrupar por 'clase' y calcular totales
df_clase = df_final.groupby('clase')[['fallecidos', 'lesionados']].sum().sort_values(by='fallecidos', ascending=False).reset_index()

# Agrupar por 'causa' y calcular totales
df_causa = df_final.groupby('causa')[['fallecidos', 'lesionados']].sum().sort_values(by='fallecidos', ascending=False).reset_index()

# Gráfica de barras para 'clase'
plt.figure(figsize=(12,6))
plt.bar(df_clase['clase'].head(10), df_clase['fallecidos'].head(10), color='purple', alpha=0.7, label='Fallecidos')
plt.bar(df_clase['clase'].head(10), df_clase['lesionados'].head(10), color='pink', alpha=0.5, label='Lesionados')
plt.xticks(rotation=45, ha='right')
plt.title('Top 10 Clases de Accidente con Mayor Número de Fallecidos y Lesionados')
plt.ylabel('Número de Personas')
plt.legend()
plt.tight_layout()
plt.show()

# Gráfica de barras para 'causa'
plt.figure(figsize=(12,6))
plt.bar(df_causa['causa'].head(10), df_causa['fallecidos'].head(10), color='purple', alpha=0.7, label='Fallecidos')
plt.bar(df_causa['causa'].head(10), df_causa['lesionados'].head(10), color='pink', alpha=0.5, label='Lesionados')
plt.xticks(rotation=45, ha='right')
plt.title('Top 10 Causas de Accidente con Mayor Número de Fallecidos y Lesionados')
plt.ylabel('Número de Personas')
plt.legend()
plt.tight_layout()
plt.show()

### Tabla promedio anual de fallecidos y lesionados ###
import pandas as pd

# Agrupar por año y clase
tabla_clase = df_final.groupby(['anio', 'clase'])[['fallecidos', 'lesionados']].sum().reset_index()
print(tabla_clase)

# Agrupar por año y causa
tabla_causa = df_final.groupby(['anio', 'causa'])[['fallecidos', 'lesionados']].sum().reset_index()
print(tabla_causa)

import matplotlib.pyplot as plt

# Agrupar por clase de accidente
tabla_clase = df_final.groupby('clase')['fallecidos'].sum().sort_values(ascending=False)

# Gráfico de barras
plt.figure(figsize=(10,6))
tabla_clase.plot(kind='bar', color='#FF5733', edgecolor='black')
plt.title('Fallecidos por Clase de Accidente')
plt.xlabel('Clase de Accidente')
plt.ylabel('Número de Fallecidos')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Agrupar por causa de accidente
tabla_causa = df_final.groupby('causa')['fallecidos'].sum().sort_values(ascending=False)

# Gráfico de barras
plt.figure(figsize=(10,6))
tabla_causa.plot(kind='bar', color='#33C1FF', edgecolor='black')
plt.title('Fallecidos por Causa de Accidente')
plt.xlabel('Causa del Accidente')
plt.ylabel('Número de Fallecidos')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

for index, value in enumerate(tabla_clase.values):
    plt.text(index, value, str(value), ha='center', va='bottom')

for index, value in enumerate(tabla_causa.values):
    plt.text(index, value, str(value), ha='center', va='bottom')

## Gráfico para lesionados por causa y año
import matplotlib.pyplot as plt

# Agrupamos por año y causa, obteniendo un DataFrame
tabla_causa = df_final.groupby(['anio', 'causa'])[['fallecidos', 'lesionados']].sum().reset_index()

# Pivot table para lesionados
pivot_lesionados = tabla_causa.pivot_table(index='causa', columns='anio', values='lesionados', fill_value=0)

# Gráfica de barras
pivot_lesionados.plot(kind='bar', figsize=(12,6))
plt.title('Lesionados por Causa y Año')
plt.ylabel('Número de Lesionados')
plt.xlabel('Causa')
plt.show()


######### PREPARACION PREVIA A ARIMA #######
# Serie de tiempo: Fallecidos por año (total nacional)
fallecidos_por_anio = df_final.groupby('anio')['fallecidos'].sum().reset_index()
fallecidos_por_anio.set_index('anio', inplace=True)
print("\n📊 Fallecidos por año:")
print(fallecidos_por_anio)

# Serie de tiempo: Fallecidos por provincias clave
provincias_clave = ['GUAYAS', 'PICHINCHA', 'LOS RIOS']

fallecidos_por_provincia = df_final[df_final['provincia'].isin(provincias_clave)]
fallecidos_por_provincia = fallecidos_por_provincia.groupby(['anio', 'provincia'])['fallecidos'].sum().reset_index()

# Pivot para formato de series de tiempo
serie_provincias = fallecidos_por_provincia.pivot(index='anio', columns='provincia', values='fallecidos')
print("\n📊 Fallecidos por año y provincia:")
print(serie_provincias)

import matplotlib.pyplot as plt

# Fallecidos por año con el uso de datos a nivel nacional
plt.figure(figsize=(8,5))
plt.plot(fallecidos_por_anio.index, fallecidos_por_anio['fallecidos'], marker='o', linestyle='-')
plt.title('Fallecidos por Año (Total Nacional)')
plt.xlabel('Año')
plt.ylabel('Número de Fallecidos')
plt.grid(True)
plt.show()

# Fallecidos por año en provincias clave Guayas, Pichincha, Los Rios
plt.figure(figsize=(10,6))
for provincia in serie_provincias.columns:
    plt.plot(serie_provincias.index, serie_provincias[provincia], marker='o', linestyle='-', label=provincia)

plt.title('Fallecidos por Año en Provincias Clave')
plt.xlabel('Año')
plt.ylabel('Número de Fallecidos')
plt.legend()
plt.grid(True)
plt.show()

# Mapeo de nombres de meses a números
mes_mapeo = {
    'ENERO': 1, 'FEBRERO': 2, 'MARZO': 3, 'ABRIL': 4,
    'MAYO': 5, 'JUNIO': 6, 'JULIO': 7, 'AGOSTO': 8,
    'SEPTIEMBRE': 9, 'OCTUBRE': 10, 'NOVIEMBRE': 11, 'DICIEMBRE': 12
}

# Crear columna mes_num
df_final['mes_num'] = df_final['mes'].map(mes_mapeo)

print(df_final[['mes', 'mes_num']].head(20))

#Mapeo para gráfica Fallecidos pro trimestre por año
df_final['fallecidos'].value_counts()
df_final[['anio', 'mes', 'mes_num', 'trimestre', 'fallecidos']].sample(10)
# Mapear mes a número correctamente:
mes_dict = {
    'ENERO': 1, 'FEBRERO': 2, 'MARZO': 3, 'ABRIL': 4, 'MAYO': 5, 'JUNIO': 6,
    'JULIO': 7, 'AGOSTO': 8, 'SEPTIEMBRE': 9, 'OCTUBRE': 10, 'NOVIEMBRE': 11, 'DICIEMBRE': 12
}
df_final['mes_num'] = df_final['mes'].map(mes_dict)

# Crear columna de trimestre
df_final['trimestre'] = ((df_final['mes_num'] - 1) // 3 + 1).astype("Int64")
fallecidos_trimestre = df_final.groupby(['anio', 'trimestre'])['fallecidos'].sum().reset_index()
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,6))
sns.lineplot(data=fallecidos_trimestre, x='trimestre', y='fallecidos', hue='anio', marker='o')
plt.title('Fallecidos por Trimestre y Año')
plt.ylabel('Número de Fallecidos')
plt.xlabel('Trimestre')
plt.show()

#Tabla resumen de fallecidos y lesionados por año y trimestre
# Agrupamos por año y trimestre, sumando fallecidos y lesionados
resumen_trimestre = df_final.groupby(['anio', 'trimestre'])[['fallecidos', 'lesionados']].sum().reset_index()

# Mostramos la tabla en consola
print(resumen_trimestre)

# Guardar en un Excel o CSV:
resumen_trimestre.to_excel("resumen_fallecidos_lesionados.xlsx", index=False)

import matplotlib.pyplot as plt

# Obtenemos los años únicos
anios = resumen_trimestre['anio'].unique()

# Iteramos por cada año
for anio in anios:
    data = resumen_trimestre[resumen_trimestre['anio'] == anio]
    
    # Crear el gráfico
    plt.figure(figsize=(8,5))
    plt.bar(data['trimestre'] - 0.15, data['fallecidos'], width=0.3, label='Fallecidos', color='#C39BD3')  # Morado claro
    plt.bar(data['trimestre'] + 0.15, data['lesionados'], width=0.3, label='Lesionados', color='#F5B7B1')  # Rosa claro
    
    plt.title(f'Fallecidos y Lesionados por Trimestre en {anio}')
    plt.xlabel('Trimestre')
    plt.ylabel('Número de Personas')
    plt.xticks(data['trimestre'])
    plt.legend()
    plt.tight_layout()
    plt.show()

    #Creacion de df_trimestres para pruebas posteriores
    import pandas as pd

# Creamos el DataFrame a mano según la tabla 
data = {
    'anio': [2022, 2022, 2022, 2022, 2023, 2023, 2023, 2023, 2024, 2024, 2024, 2024],
    'trimestre': [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
    'fallecidos': [1554, 1614, 1178, 557, 2212, 1809, 1200, 617, 1960, 1644, 1178, 675],
    'lesionados': [12564, 14493, 10108, 4933, 18180, 13077, 9686, 4858, 16755, 14190, 9052, 4867]
}

df_trimestres = pd.DataFrame(data)

# Revisar el DataFrame
print(df_trimestres)


########### REVISION DE SUPUESTOS DE ARIMA ########
# 1. ESTACIONARIEDAD FALLECIDOS

from statsmodels.tsa.stattools import adfuller

# Tomamos la serie de fallecidos agregados por trimestre
serie_fallecidos = df_trimestres['fallecidos']

# Ejecutamos la prueba ADF
resultado_adf = adfuller(serie_fallecidos.dropna())

# Mostramos resultados
print("Resultado de la prueba ADF:")
print(f"Estadístico ADF: {resultado_adf[0]}")
print(f"Valor p: {resultado_adf[1]}")
print(f"Número de lags: {resultado_adf[2]}")
print(f"Número de observaciones: {resultado_adf[3]}")

# Valores críticos
for clave, valor in resultado_adf[4].items():
    print(f"Valor crítico {clave}: {valor}")

# Interpretación simple
if resultado_adf[1] < 0.05:
    print("\n✅ La serie es estacionaria (rechazamos la hipótesis nula de no estacionariedad).")
else:
    print("\n❌ La serie NO es estacionaria (no se rechaza la hipótesis nula).")

#Diferenciacion de la sere fallecidos para probar estacionariedad
# Diferenciar la serie de fallecidos
df_trimestres['fallecidos_diff'] = df_trimestres['fallecidos'].diff()

# Mostrar los primeros valores para revisar
print(df_trimestres[['anio', 'trimestre', 'fallecidos', 'fallecidos_diff']])


## 1.1 ESTACIONARIEDAD PRUEBA DICKEY-FULLER DIFERENCIADA
from statsmodels.tsa.stattools import adfuller

# Serie diferenciada (sin NaN)
fallecidos_diff = df_trimestres['fallecidos_diff'].dropna()

# Prueba ADF
resultado_adf = adfuller(fallecidos_diff)

# Resultados
print("\n✅ Resultado de la prueba ADF (Serie Diferenciada):")
print(f"Estadístico ADF: {resultado_adf[0]}")
print(f"Valor p: {resultado_adf[1]}")
print(f"Número de lags: {resultado_adf[2]}")
print(f"Número de observaciones: {resultado_adf[3]}")
for key, value in resultado_adf[4].items():
   print(f"Valor crítico {key}: {value}")

# Interpretación
if resultado_adf[1] <= 0.05:
   print("✅ La serie diferenciada es estacionaria (se rechaza la hipótesis nula).")
else:
   print("❌ La serie diferenciada NO es estacionaria (no se rechaza la hipótesis nula).")

###### Definicion de p y q con d=1 ########
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Gráfico ACF
plot_acf(df_trimestres['fallecidos_diff'].dropna(), lags=4, ax=axs[0])
axs[0].set_title('ACF - Correlación con Retardos')

# Gráfico PACF
plot_pacf(df_trimestres['fallecidos_diff'].dropna(), lags=4, ax=axs[1])
axs[1].set_title('PACF - Correlación Parcial con Retardos')

plt.tight_layout()
plt.show()

##### Elección de parámetros para ARIMA (1,1,1) y ARIMA (2,1,2) #######
import warnings
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

# Ignorar advertencias
warnings.filterwarnings("ignore")

# Definir la serie (ya diferenciada)
serie_fallecidos = df_trimestres['fallecidos']

# Probar ARIMA(1,1,1)
print("\n🔎 Resultados ARIMA(1,1,1):")
modelo_111 = ARIMA(serie_fallecidos, order=(1,1,1))
resultado_111 = modelo_111.fit()
print(resultado_111.summary())

# Probar ARIMA(2,1,2)
print("\n🔎 Resultados ARIMA(2,1,2):")
modelo_212 = ARIMA(serie_fallecidos, order=(2,1,2))
resultado_212 = modelo_212.fit()
print(resultado_212.summary())

##### Predicciones con ARIMA 1,1,1
# Ajustar el modelo ARIMA(1,1,1)
from statsmodels.tsa.statespace.sarimax import SARIMAX

modelo_arima = SARIMAX(df_trimestres['fallecidos'], order=(1,1,1))
resultado_arima = modelo_arima.fit(disp=False)

# Generar predicciones in-sample (ajustadas a los datos)
predicciones = resultado_arima.get_prediction(start=0, end=20, dynamic=False)
predicciones_media = predicciones.predicted_mean

# Calcular intervalo de confianza si deseas
conf_int = predicciones.conf_int()

#Gráfico 
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.plot(df_trimestres['fallecidos'].reset_index(drop=True), label='Datos Reales', marker='o')
plt.plot(predicciones_media.reset_index(drop=True), label='Predicciones ARIMA(1,1,1)', linestyle='--', marker='x')

plt.fill_between(predicciones_media.index,
                 conf_int.iloc[:, 0],
                 conf_int.iloc[:, 1],
                 color='pink', alpha=0.2, label='Intervalo de Confianza')

plt.title('Predicción ARIMA(1,1,1) vs Datos Reales')
plt.xlabel('Observaciones (Año-Trimestre)')
plt.ylabel('Número de Fallecidos')
plt.legend()
plt.tight_layout()
plt.show()

## Comparación grafica de ARIMA 2,1,2

from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Ajustar modelo ARIMA(2,1,2)
modelo_212 = SARIMAX(df_trimestres['fallecidos'], order=(2,1,2))
resultado_212 = modelo_212.fit(disp=False)

# Número de observaciones actuales
n_obs = len(df_trimestres)

# Predicciones extendidas 8 trimestres más
predicciones_212 = resultado_212.get_prediction(start=0, end=n_obs + 7)
pred_media_212 = predicciones_212.predicted_mean
conf_int_212 = predicciones_212.conf_int()

# Crear eje x extendido
index_extendido = list(df_trimestres.index) + [f'Pred {i+1}' for i in range(8)]

# Gráfico
plt.figure(figsize=(10,6))
plt.plot(index_extendido, pred_media_212, label='Predicción ARIMA(2,1,2)', marker='x', linestyle='--')
plt.plot(df_trimestres.index, df_trimestres['fallecidos'], label='Datos Reales', marker='o')
plt.fill_between(range(len(pred_media_212)), 
                 conf_int_212.iloc[:,0], 
                 conf_int_212.iloc[:,1], 
                 color='pink', alpha=0.3, label='Intervalo de Confianza')

plt.title('Predicción ARIMA(2,1,2) extendida vs Datos Reales')
plt.xlabel('Observaciones (Año-Trimestre)')
plt.ylabel('Número de Fallecidos')
plt.legend()
plt.tight_layout()
plt.show()


### PRUEBA SARIMA CON ESTACIONALIDAD DE TRIMESTRES (1,1,1,1)x(1,1,1,4) ###
# Parámetros para SARIMA (puedes ajustarlos luego)
p = 1  # AR
d = 1  # Diferenciación
q = 1  # MA
P = 1  # Estacional AR
D = 1  # Estacional diferencia
Q = 1  # Estacional MA
s = 4  # Periodicidad estacional (4 trimestres)

from statsmodels.tsa.statespace.sarimax import SARIMAX

# Definir modelo SARIMA
modelo_sarima = SARIMAX(df_trimestres['fallecidos'],
                        order=(p,d,q),
                        seasonal_order=(P,D,Q,s))

# Ajustar modelo
resultado_sarima = modelo_sarima.fit()

# Ver resumen
print(resultado_sarima.summary())

####### APLICACION DE HOLT-WINTERS ######
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

# Ajustar modelo Holt-Winters (si no lo tienes aún)
modelo_hw = ExponentialSmoothing(df_trimestres['fallecidos'], trend='add', seasonal='add', seasonal_periods=4)
resultado_hw = modelo_hw.fit()

# Predicción extendida
pred_hw = resultado_hw.predict(start=0, end=len(df_trimestres) + 7)

# Eje x extendido
index_ext = list(df_trimestres.index) + [f'Pred {i+1}' for i in range(8)]

# Gráfico
plt.figure(figsize=(10,6))
plt.plot(index_ext, pred_hw, label='Predicciones Holt-Winters', linestyle='--', marker='x')
plt.plot(df_trimestres.index, df_trimestres['fallecidos'], label='Datos Reales', marker='o')

plt.title('Predicción Holt-Winters extendida vs Datos Reales')
plt.xlabel('Observaciones (Año-Trimestre)')
plt.ylabel('Número de Fallecidos')
plt.legend()
plt.tight_layout()
plt.show()

# Mostrar resumen del modelo
print(resultado_hw.summary())

# Predicciones de Holt-Winters
predicciones_hw = resultado_hw.fittedvalues

##### MÉTRICAS DE MEDICIÓN COMPARATIVAS ENTRE ARIMA Y HOLT-WINTERS #####
# Para ARIMA(1,1,1)
predicciones_arima_111 = resultado_111.fittedvalues

# Para ARIMA(2,1,2)
predicciones_arima_212 = resultado_212.fittedvalues

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Datos reales
y_real = df_trimestres['fallecidos'].values

# Predicciones (ya generadas previamente)
y_pred_arima_111 = predicciones_arima_111  # variable con tus predicciones ARIMA(1,1,1)
y_pred_arima_212 = predicciones_arima_212  # variable con tus predicciones ARIMA(2,1,2)
y_pred_hw = predicciones_hw  # variable con tus predicciones Holt-Winters

# Función para calcular métricas
def calcular_metricas(y_real, y_pred):
    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mape = np.mean(np.abs((y_real - y_pred) / y_real)) * 100
    return mae, rmse, mape

# Calcular métricas
mae_arima_111, rmse_arima_111, mape_arima_111 = calcular_metricas(y_real, y_pred_arima_111)
mae_arima_212, rmse_arima_212, mape_arima_212 = calcular_metricas(y_real, y_pred_arima_212)
mae_hw, rmse_hw, mape_hw = calcular_metricas(y_real, y_pred_hw)

# Mostrar resultados
print(f"ARIMA(1,1,1): MAE={mae_arima_111:.2f}, RMSE={rmse_arima_111:.2f}, MAPE={mape_arima_111:.2f}%")
print(f"ARIMA(2,1,2): MAE={mae_arima_212:.2f}, RMSE={rmse_arima_212:.2f}, MAPE={mape_arima_212:.2f}%")
print(f"Holt-Winters: MAE={mae_hw:.2f}, RMSE={rmse_hw:.2f}, MAPE={mape_hw:.2f}%")


##### PREDICCIONES PARA PROVINCIAS IMPORTANTES GUAYAS-PICHINCHA-LOS RIOS####
#### GUAYAS #####
df_guayas = df_final[df_final['provincia'] == 'GUAYAS'].groupby(['anio', 'trimestre'])['fallecidos'].sum().reset_index()
from statsmodels.tsa.holtwinters import ExponentialSmoothing

modelo_hw_guayas = ExponentialSmoothing(
    df_guayas['fallecidos'],
    trend='add',
    seasonal='add',
    seasonal_periods=4
).fit()

predicciones_guayas = modelo_hw_guayas.fittedvalues
print(modelo_hw_guayas.summary())
predicciones_guayas = modelo_hw_guayas.fittedvalues

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.plot(df_guayas['fallecidos'].values, label='Datos Reales', marker='o')
plt.plot(predicciones_guayas.values, label='Predicciones Holt-Winters', linestyle='--', marker='x')
plt.title('Predicción Holt-Winters para Guayas')
plt.xlabel('Observaciones (Año-Trimestre)')
plt.ylabel('Número de Fallecidos')
plt.legend()
plt.grid(True)
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np

# Datos reales y predicciones para Guayas
y_real_guayas = df_guayas['fallecidos'].values
y_pred_guayas = predicciones_guayas # Asegúrate de que esta sea tu variable de predicciones

# Cálculo de métricas
mae_guayas = mean_absolute_error(y_real_guayas, y_pred_guayas)
rmse_guayas = np.sqrt(mean_squared_error(y_real_guayas, y_pred_guayas))
mape_guayas = mean_absolute_percentage_error(y_real_guayas, y_pred_guayas) * 100

print(f"MAE (Guayas) = {mae_guayas:.2f}")
print(f"RMSE (Guayas) = {rmse_guayas:.2f}")
print(f"MAPE (Guayas) = {mape_guayas:.2f}%")

#### PICHINCHA #####
df_pichincha = df_final[df_final['provincia'] == 'PICHINCHA'].groupby(['anio', 'trimestre'])['fallecidos'].sum().reset_index()
from statsmodels.tsa.holtwinters import ExponentialSmoothing

modelo_hw_pichincha = ExponentialSmoothing(
    df_pichincha['fallecidos'],
    trend='add',
    seasonal='add',
    seasonal_periods=4
).fit()

predicciones_pichincha = modelo_hw_pichincha.fittedvalues
print(modelo_hw_pichincha.summary())
predicciones_pichincha = modelo_hw_pichincha.fittedvalues

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.plot(df_pichincha['fallecidos'].values, label='Datos Reales', marker='o')
plt.plot(predicciones_pichincha.values, label='Predicciones Holt-Winters', linestyle='--', marker='x')
plt.title('Predicción Holt-Winters para Pichincha')
plt.xlabel('Observaciones (Año-Trimestre)')
plt.ylabel('Número de Fallecidos')
plt.legend()
plt.grid(True)
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np

# Datos reales y predicciones para Pichincha
y_real_pichincha = df_pichincha['fallecidos'].values
y_pred_pichincha = predicciones_pichincha 

# Cálculo de métricas
mae_pichincha = mean_absolute_error(y_real_pichincha, y_pred_pichincha)
rmse_pichincha = np.sqrt(mean_squared_error(y_real_pichincha, y_pred_pichincha))
mape_pichincha = mean_absolute_percentage_error(y_real_pichincha, y_pred_pichincha) * 100

print(f"MAE (Pichincha) = {mae_pichincha:.2f}")
print(f"RMSE (Pichincha) = {rmse_pichincha:.2f}")
print(f"MAPE (Pichincha) = {mape_pichincha:.2f}%")

#### LOS RIOS #####
df_losrios = df_final[df_final['provincia'] == 'LOS RIOS'].groupby(['anio', 'trimestre'])['fallecidos'].sum().reset_index()
from statsmodels.tsa.holtwinters import ExponentialSmoothing

modelo_hw_losrios = ExponentialSmoothing(
    df_losrios['fallecidos'],
    trend='add',
    seasonal='add',
    seasonal_periods=4
).fit()

predicciones_losrios = modelo_hw_losrios.fittedvalues
print(modelo_hw_losrios.summary())
predicciones_losrios = modelo_hw_losrios.fittedvalues

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.plot(df_losrios['fallecidos'].values, label='Datos Reales', marker='o')
plt.plot(predicciones_losrios.values, label='Predicciones Holt-Winters', linestyle='--', marker='x')
plt.title('Predicción Holt-Winters para Los Rios')
plt.xlabel('Observaciones (Año-Trimestre)')
plt.ylabel('Número de Fallecidos')
plt.legend()
plt.grid(True)
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np

# Datos reales y predicciones para Los Rios
y_real_losrios = df_losrios['fallecidos'].values
y_pred_losrios = predicciones_losrios 

# Cálculo de métricas
mae_losrios = mean_absolute_error(y_real_losrios, y_pred_losrios)
rmse_losrios = np.sqrt(mean_squared_error(y_real_losrios, y_pred_losrios))
mape_losrios = mean_absolute_percentage_error(y_real_losrios, y_pred_losrios) * 100

print(f"MAE (Los_Rios) = {mae_losrios:.2f}")
print(f"RMSE (Los_Rios) = {rmse_losrios:.2f}")
print(f"MAPE (Los_Rios) = {mape_losrios:.2f}%")