#-----------------------------------------------------------------------------------
#Descripción:  
# Este script emplea el modelo “distilbert-base-multilingual-cased” de Hugging Face para realizar un análisis de texto genérico sobre una serie de comentarios en un archivo CSV. A continuación se describen sus pasos principales:

# 1. Carga de librerías y configuración del pipeline:
#    - Utiliza pandas para leer y manipular datos.
#    - Usa transformers para crear un pipeline de clasificación basado en “distilbert-base-multilingual-cased”.

# 2. Lectura del dataset:
#    - Carga el archivo “comentariosDataset.csv”

# 3. Clasificación de comentarios:
#    - Aplica el pipeline a cada comentario, generando una etiqueta genérica (“LABEL_0”, “LABEL_1”, etc.) y un score de confianza.
#    - Agrega los resultados en nuevas columnas del DataFrame.

# 4. Cálculo de estadísticas:
#    - Cuenta cuántos comentarios existen por cada etiqueta.
#    - Calcula la proporción de cada etiqueta con respecto al total de comentarios.

# 5. Filtrado de comentarios de interés:
#    - Selecciona aquellos comentarios clasificados como “LABEL_0” cuyo score de confianza sea mayor a 0.55.
#    - Imprime estos comentarios en consola para identificar posibles textos de baja calidad o negativos con alta certeza.

# 6. Salida esperada:
#    - Un resumen en consola con los conteos y proporciones de cada etiqueta.
#    - La lista de comentarios filtrados con “LABEL_0” y score > 0.55, para su revisión manual.

# Antes de ejecutar este script, asegúrate de haber instalado las dependencias necesarias:
# pip install pandas transformers

#-----------------------------------------------------------------------------------

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Cargar el modelo preentrenado DistilBERT multilingüe
model_name = "distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Crear el pipeline de análisis de sentimiento
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Leer el archivo CSV que contiene los comentarios
df = pd.read_csv('comentariosDataset.csv')

# Realizar el análisis de sentimientos en cada comentario
df['sentimiento'] = df['comentario'].apply(lambda x: sentiment_pipeline(x)[0])

# Extraer los labels (LABEL_0, LABEL_1, etc.) para cada comentario
df['label'] = df['sentimiento'].apply(lambda x: x['label'])

# Contar cuántos comentarios hay por cada label (LABEL_0, LABEL_1)
conteo_labels = df['label'].value_counts()
print("Conteo de comentarios por label:")
print(conteo_labels)

# Calcular la proporción de comentarios de cada tipo respecto al total
total_comentarios = len(df)
proporciones = conteo_labels / total_comentarios
print("\nProporción de comentarios por label:")
print(proporciones)

# Apartar los comentarios con LABEL_0 cuyo score sea mayor a 0.55
comentarios_label_0 = df[(df['label'] == 'LABEL_0') & (df['sentimiento'].apply(lambda x: x['score'] > 0.541))]

# Imprimir los comentarios filtrados
print("\nComentarios con LABEL_0 y score mayor a 0.50:")
print(comentarios_label_0[['comentario', 'sentimiento']])
