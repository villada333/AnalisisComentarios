#-----------------------------------------------------------------------------------
# Descripción:  Uso el modelo BETO para realizar análisis de sentimientos en comentarios
# Con el modelo se hace la clasificacion de cada comentario y se entrega si es positivo o negativo con su respectivo indice de confianza
# Adicionalmente se cuenta los comentarios positivos y negativos y se calcula el ratio entre ellos
# Se filtran los comentarios negativos con un score mayor a 0.70 ya que el indice de confianza es alto
#-----------------------------------------------------------------------------------

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Cargar el modelo preentrenado BETO
model_name = "finiteautomata/beto-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Crear el pipeline de análisis de sentimiento
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Leer el archivo CSV que contiene los comentarios
df = pd.read_csv('comentariosDataset.csv')

# Realizar el análisis de sentimientos en cada comentario
df['sentimiento'] = df['comentario'].apply(lambda x: sentiment_pipeline(x)[0])

# Contar cuántos comentarios positivos y negativos hay
positivos = df[df['sentimiento'].apply(lambda x: x['label']) == 'POS'].shape[0]
negativos = df[df['sentimiento'].apply(lambda x: x['label']) == 'NEG'].shape[0]

print(f"Comentarios positivos: {positivos}")
print(f"Comentarios negativos: {negativos}")

# Calcular el ratio entre comentarios positivos y negativos
if negativos != 0:
    ratio = positivos / negativos
else:
    ratio = 'No se puede calcular (división por cero)'

print(f"Ratio de comentarios positivos vs negativos: {ratio}")

# Apartar comentarios negativos con un score mayor a 0.70
comentarios_negativos = df[df['sentimiento'].apply(lambda x: x['label'] == 'NEG' and x['score'] > 0.70)]

# Imprimir los comentarios negativos filtrados
print("Comentarios negativos con score mayor a 0.70:")
print(comentarios_negativos[['comentario', 'sentimiento']])
