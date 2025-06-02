#-----------------------------------------------------------------------------------
#Descripción: Este script utiliza el modelo preentrenado BRT para analizar el sentimiento
# Se realiza el análisis de sentimientos en cada comentario y se extrae la cantidad de estrellas
# Se cuenta cuántos comentarios hay por cantidad de estrellas y se calcula la proporción de cada tipo de comentario
# Se filtran los comentarios de 1, 2 y 3 estrellas con un score mayor a 0.55
#-----------------------------------------------------------------------------------

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Cargar el modelo preentrenado BERT multilingüe
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Crear el pipeline de análisis de sentimiento
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Leer el archivo CSV que contiene los comentarios
df = pd.read_csv('comentariosDataset.csv')

# Realizar el análisis de sentimientos en cada comentario
df['sentimiento'] = df['comentario'].apply(lambda x: sentiment_pipeline(x)[0])

# Extraer la cantidad de estrellas (label) para cada comentario
df['estrellas'] = df['sentimiento'].apply(lambda x: x['label'])

# Contar cuántos comentarios hay por cantidad de estrellas
conteo_estrellas = df['estrellas'].value_counts().sort_index()
print("Conteo de comentarios por estrellas:")
print(conteo_estrellas)

# Calcular la proporción de cada tipo de comentario respecto al total
total_comentarios = len(df)
proporciones = conteo_estrellas / total_comentarios
print("\nProporción de comentarios por estrellas:")
print(proporciones)

# Apartar los comentarios de 1 y 2 estrellas con un score mayor a 0.60
comentarios_bajos = df[(df['estrellas'].isin(['1 star', '2 stars', '3 stars'])) & (df['sentimiento'].apply(lambda x: x['score'] > 0.55))]

# Imprimir los comentarios filtrados
print("\nComentarios de 1, 2 y 3 estrellas con score mayor a 0.55:")
print(comentarios_bajos[['comentario', 'sentimiento']])

