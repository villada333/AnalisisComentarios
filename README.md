# Análisis de Sentimientos con Modelos Preentrenados

Este repositorio contiene varios scripts en Python para realizar análisis de sentimientos sobre un conjunto de comentarios utilizando distintos modelos de Transformers. El objetivo principal es demostrar cómo cargar modelos preentrenados (BERT multilingüe, BETO y DistilBERT multilingüe), procesar un archivo CSV de comentarios, calcular estadísticas básicas (conteos y proporciones por etiqueta o cantidad de estrellas) y filtrar comentarios de interés según umbrales de confianza.

Cada script está diseñado para:

- Cargar un modelo preentrenado de Hugging Face.
- Procesar un CSV de comentarios.
- Calcular estadísticas de conteo y proporción por etiqueta o estrellas.
- Filtrar aquellos comentarios con alta probabilidad de ser negativos o de baja calificación.

---

## Estructura del Repositorio

```
/
├── bert.py
├── beto.py
├── distilbertmultilingue.py
└── comentariosDataset.csv
```

- **bert.py**  
  Script que utiliza el modelo `nlptown/bert-base-multilingual-uncased-sentiment` para clasificar cada comentario en un rango de 1 a 5 estrellas, calcular conteos y proporciones por cantidad de estrellas y filtrar comentarios con 1, 2 o 3 estrellas cuyo score supere 0.55.

- **beto.py**  
  Script que emplea el modelo `finiteautomata/beto-sentiment-analysis` (BETO) para categorizar cada comentario como POS (positivo) o NEG (negativo), contar cuántos comentarios entran en cada categoría, calcular el ratio positivo/negativo y filtrar aquellos negativos con score > 0.70.

- **distilbertmultilingue.py**  
  Script que utiliza `distilbert-base-multilingual-cased` para clasificar cada comentario con etiquetas genéricas (`LABEL_0`, `LABEL_1`, etc.), calcular conteos y proporciones por etiqueta, y filtrar aquellos comentarios con etiqueta `LABEL_0` cuyo score sea mayor a 0.55.

- **comentariosDataset.csv**  
  Archivo CSV de ejemplo que contiene, al menos, una columna llamada `comentario` con texto libre. Cada script leerá este archivo para procesarlo.

---

## Descripción General

Cada uno de los tres scripts sigue un flujo similar:

1. **Carga de librerías y modelo preentrenado**  
   - `pandas` para leer y manipular datos en formato CSV.  
   - `transformers` (de Hugging Face) para cargar el tokenizador y el modelo de clasificación de sentimiento.  
   - Se crea un pipeline de `sentiment-analysis` asociado al modelo seleccionado.

2. **Lectura del Dataset**  
   - Se abre el archivo `comentariosDataset.csv` (debe estar en la misma carpeta que los scripts).  
   - Se asume que existe una columna llamada `comentario` con texto en español.

3. **Análisis de Sentimiento**  
   - Se aplica el pipeline de sentimiento a cada fila, obteniendo un diccionario con `label` y `score`.  
   - En función del modelo:  
     - `bert.py` extrae etiquetas como `"1 star"`, `"2 stars"`, …, `"5 stars"`.  
     - `beto.py` clasifica en `"POS"` o `"NEG"`.  
     - `distilbertmultilingue.py` usa etiquetas genéricas como `"LABEL_0"`, `"LABEL_1"`, etc.

4. **Cálculo de Estadísticas**  
   - Se cuentan cuántos comentarios hay por cada etiqueta o cantidad de estrellas.  
   - Se calcula la proporción de cada categoría con respecto al total de comentarios.

5. **Filtrado de Comentarios Relevantes**  
   - Cada script aplica un umbral de score para seleccionar comentarios “negativos” o “bajos” con alta confianza:  
     - En `bert.py`: comentarios de 1, 2 o 3 estrellas cuyo score (según el pipeline) > 0.55.  
     - En `beto.py`: comentarios negativos (`label == "NEG"`) con score > 0.70.  
     - En `distilbertmultilingue.py`: comentarios con `LABEL_0` y score > 0.55.  
   - Se imprimen dichos comentarios para su revisión.

---

## Requisitos

- **Python 3.7+**  
- **pip** (o `conda` / `venv`)  
- **Librerías**  
  - `pandas`  
  - `transformers` (incluye `torch` o `tensorflow` como backend, según configuración)  

### Instalación Rápida

1. Crear un entorno virtual (opcional pero recomendado):
   ```bash
   python3 -m venv venv
   source venv/bin/activate       # Linux/macOS
   .\venv\Scripts\activate        # Windows
   ```

2. Instalar dependencias:
   ```bash
   pip install pandas transformers
   ```

3. Verificar que los scripts y el archivo CSV estén en la misma carpeta.

---

## Cómo Ejecutar Cada Script

A continuación se muestran ejemplos de uso para cada uno de los archivos. Asegúrate de que el archivo `comentariosDataset.csv` existe y contiene al menos la columna `comentario`.

### 1. `bert.py`

```bash
python bert.py
```

- **Qué hace**  
  - Carga `nlptown/bert-base-multilingual-uncased-sentiment`.  
  - Lee `comentariosDataset.csv`.  
  - Clasifica cada comentario en `"1 star"`, `"2 stars"`, …, `"5 stars"`.  
  - Imprime en consola:
    1. Conteo total de comentarios por cantidad de estrellas (ordenado de menor a mayor).
    2. Proporción de cada categoría respecto al total.
    3. Lista de comentarios de 1, 2 o 3 estrellas con score > 0.55 (etiqueta + score).

- **Salida esperada (ejemplo)**  
  ```
  Conteo de comentarios por estrellas:
  1 star      25
  2 stars     40
  3 stars     60
  4 stars     80
  5 stars    100
  Name: estrellas, dtype: int64

  Proporción de comentarios por estrellas:
  1 star     0.10
  2 stars    0.16
  3 stars    0.24
  4 stars    0.32
  5 stars    0.40
  Name: estrellas, dtype: float64

  Comentarios de 1, 2 y 3 estrellas con score mayor a 0.55:
                                  comentario                          sentimiento
  12     "No me gustó nada el producto..."    {'label': '1 star', 'score': 0.60}
  47     "Servicio lento, atención deficiente"    {'label': '2 stars', 'score': 0.57}
  ...
  ```
  *(Los valores de conteo y proporción varían según el dataset real.)*

### 2. `beto.py`

```bash
python beto.py
```

- **Qué hace**  
  - Carga `finiteautomata/beto-sentiment-analysis`.  
  - Lee `comentariosDataset.csv`.  
  - Clasifica cada comentario en `"POS"` o `"NEG"`, con su score de confianza.  
  - Imprime en consola:
    1. Cantidad de comentarios positivos (`POS`).
    2. Cantidad de comentarios negativos (`NEG`).
    3. Ratio de comentarios positivos vs negativos (si negativos ≠ 0; de lo contrario, indicará que no se puede calcular).
    4. Lista de comentarios negativos con score > 0.70.

- **Salida esperada (ejemplo)**  
  ```
  Comentarios positivos: 150
  Comentarios negativos: 75
  Ratio de comentarios positivos vs negativos: 2.0

  Comentarios negativos con score mayor a 0.70:
                                  comentario                          sentimiento
  23     "El producto llegó dañado..."    {'label': 'NEG', 'score': 0.75}
  89     "Muy mala atención al cliente"   {'label': 'NEG', 'score': 0.82}
  ...
  ```

### 3. `distilbertmultilingue.py`

```bash
python distilbertmultilingue.py
```

- **Qué hace**  
  - Carga `distilbert-base-multilingual-cased` (modelo general, sin afinación específica para sentimiento).  
  - Lee `comentariosDataset.csv`.  
  - Clasifica cada comentario en etiquetas genéricas (`LABEL_0`, `LABEL_1`, etc.), con su score.  
  - Imprime en consola:
    1. Conteo de comentarios por cada etiqueta.
    2. Proporción de cada etiqueta respecto al total.
    3. Lista de comentarios con `LABEL_0` y score > 0.55.

- **Salida esperada (ejemplo)**  
  ```
  Conteo de comentarios por label:
  LABEL_0    120
  LABEL_1    100
  LABEL_2     50
  Name: label, dtype: int64

  Proporción de comentarios por label:
  LABEL_0    0.48
  LABEL_1    0.40
  LABEL_2    0.20
  Name: label, dtype: float64

  Comentarios con LABEL_0 y score mayor a 0.50:
                                  comentario                         sentimiento
  5      "No recomendaría este servicio..."   {'label': 'LABEL_0', 'score': 0.58}
  34     "Mal embalaje, demoraron demasiado"   {'label': 'LABEL_0', 'score': 0.62}
  ...
  ```

---

## Cómo Personalizar y Extender

1. **Dataset Propio**  
   - Sustituye `comentariosDataset.csv` por tu propio archivo CSV.  
   - Asegúrate de que la columna que contiene el texto de los comentarios se llame exactamente `comentario`.  
   - Si tu dataset tiene otra estructura, edita la línea donde se lee el CSV y la columna, por ejemplo:
     ```python
     df = pd.read_csv('mi_archivo.csv')
     df['sentimiento'] = df['mi_columna_de_texto'].apply(lambda x: sentiment_pipeline(x)[0])
     ```

2. **Modelos Diferentes**  
   - Para probar otro modelo de análisis de sentimientos, basta con cambiar la línea:
     ```python
     model_name = "ruta/del-modelo"
     ```
     Ejemplo:  
     ```python
     model_name = "nlptown/bert-base-multilingual-cased-sentiment"
     ```
   - Verifica que el modelo seleccionado sea compatible con el pipeline `"sentiment-analysis"`.  

3. **Ajuste de Umbrales**  
   - Cada script tiene una condición para filtrar comentarios con alta “confianza negativa” o “baja calificación”.  
   - Si deseas modificar el umbral (por ejemplo, de 0.55 a 0.60), localiza la línea correspondiente:
     ```python
     comentarios_bajos = df[(df['estrellas'].isin(['1 star', '2 stars', '3 stars'])) 
                           & (df['sentimiento'].apply(lambda x: x['score'] > 0.55))]
     ```
     y reemplaza `0.55` por el nuevo valor.  

4. **Formato de Salida**  
   - Actualmente, los scripts muestran estadísticas e imprimen DataFrames filtrados en la consola.  
   - Para exportar resultados a CSV o JSON, podrías agregar, por ejemplo:
     ```python
     comentarios_bajos.to_csv('comentarios_filtrados.csv', index=False)
     ```
   - O bien usar `df.to_json('resultado.json', orient='records', force_ascii=False)`.

---

## Dependencias y Versiones Sugeridas

- Python ≥ 3.7  
- pandas ≥ 1.0.0  
- transformers ≥ 4.0.0  
- torch ≥ 1.6.0 (o tensorflow, según backend de tu preferencia)

Para verificar versiones instaladas:

```bash
python -c "import pandas as pd; print(pd.__version__)"
python -c "import transformers as tfm; print(tfm.__version__)"
```

---