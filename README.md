# ProyectoMBD-CapstoneUdla
Este proyecto forma parte de una tesis de maestria en Negocios y Análisis de Datos que busca principalmente analizar, modelar y predecir el número de accidentes de tránsito en Ecuador, empleando datos trimestrales de los años 2022-2024
El proyecto tiene como objeto desarollar un modelo de predicción que permita anticipar la siniestralidad en Ecuador, apoyando a la toma de decisiones generenciales en temas de seguridad y planificación pública. De este modo, el codigo cuenta con las siguientes partes: 
# Carga de datos y limpieza
- Estandarización de provincias y variables categóricas
- Conversión de fechas y creación de diccionarios de referencias para estandarización
# Análisis exploratorio
- Gráficos de barras y líneas por año, provincia, clase y causa de accidente
# Modelado estadístico
- Aplicación de modelos ARIMA (1,1,1) y (2,1,2)
- Aplicación de Holt-Winters a nivel nacional y provincial
# Evaluación de modelos 
- Métricas evaluadas: MAE, RMSE, MAPE
- Comparación visual y numérica de predicciones vs. datos reales
# Resultados
- Predicción por trimestres hasta el año 2026
El proyecto fue desarrollado en Visual Code Studio mediante el lenguaje de programación de Phyton, empleando principalmente las siguientes librerias: pandas, numpy, matplotlib, seaborn, statsmodels, sklearn, openpyxl.
# Salida esperada
- Gráficos de predicción del número de fallecidos hasta el año 2026
- Métricas de desempeño para cada modelo
- Visualización de tendencias estacionarias
- Tabla final con predicciones trimestrales
# Autora
Stephanie Cahueñas
Maestría en Negocios y Análisis de Datos - Universidad de las Américas
Año: 2025
# Referencias Principales
- Castillo, D., Coral, C., Salazar Méndez, Y., Castillo, D., Coral, C., & Salazar Méndez, Y. (2020). Modelización Econométrica de los Accidentes de Tránsito en el Ecuador. Revista Politécnica, 46(2), 21–28. https://doi.org/10.33333/RP.VOL46N2.02
- INEC. (2025). Siniestros de tránsito trimestral |. https://www.ecuadorencifras.gob.ec/siniestros-transito-trimestral/
- Otros artículos incluídos en el documento de tesis
