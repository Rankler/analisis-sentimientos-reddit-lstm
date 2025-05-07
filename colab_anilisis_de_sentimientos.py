# ## Proyecto de PLN: Análisis de Sentimientos de Reddit con Word2Vec + LSTM

# ### Instalación de librerías

!pip install nltk gensim scikit-learn tensorflow praw
# Importación de las Librerias
import nltk
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
import praw
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# Credenciales Reddit
CLIENT_ID = "ID"
CLIENT_SECRET = "ID Secreto"
SUBREDDITS = ["Paraguay"]
USER_AGENT = "Reddit Paraguay"
REDDIT_USERNAME = "Usuario"
REDDIT_PASSWORD = "Contraseña"
# ### Descargar recursos de NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
def fetch_new_posts(reddit, subreddit):
    new_posts = []
    for submission in reddit.subreddit(subreddit).new(limit=100):
        title = submission.title
        url = submission.url
        author = submission.author.name if submission.author else "[deleted]"
        body = submission.selftext.strip() if submission.selftext else "(sin texto)"
        new_posts.append((title, url, author, body))
    return new_posts

def compile_new_digest(reddit):
    compiled_content = ""
    for subreddit in SUBREDDITS:
        new_posts = fetch_new_posts(reddit, subreddit)
        compiled_content += f"**[{subreddit}](https://www.reddit.com/r/{subreddit}) New Posts - {datetime.date.today()}**\n\n"
        for i, (title, url, author, body) in enumerate(new_posts, start=1):
            compiled_content += f"{i}. [{title}]({url}) by u/{author}\n\n"
            compiled_content += f"> {body}\n\n"
    return compiled_content

reddit = praw.Reddit(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, user_agent=USER_AGENT, username=REDDIT_USERNAME, password=REDDIT_PASSWORD, check_for_async=False)
# ### Preprocesamiento de texto

stop_words = set(stopwords.words('spanish')) # O el idioma relevante

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', str(text), re.UNICODE) # Eliminar puntuación
    text = text.lower() # Minúsculas
    tokens = word_tokenize(text) # Tokenizar
    tokens = [word for word in tokens if word not in stop_words] # Eliminar palabras vacías
    return tokens

data['tokens'] = data['comment_text'].apply(clean_text)
# ### Crear el modelo Word2Vec
# Entrenar el modelo Word2Vec
word2vec_model = Word2Vec(sentences=data['tokens'], vector_size=100, window=5, min_count=1, workers=4)  # Ajusta los parámetros

# Guardar el modelo Word2Vec (opcional)
# word2vec_model.save("word2vec.model")
### Preparar los datos para la LSTM


# Obtener el Tamaño del vocabulario
vocabulary_size = len(word2vec_model.wv.key_to_index) + 1  # +1 para tokens desconocidos

# Crear un mapeo de palabras a su vector Word2Vec
def get_index(tokens):
    vector = []
    for word in tokens:
        try:
          vector.append(word2vec_model.wv.key_to_index[word])
        except KeyError:
            vector.append(0) # Manejar palabras fuera del vocabulario (añadirlo al índice 0)
    return vector


data['indexed'] = data['tokens'].apply(get_index)
# Cargar conjunto de datos de sentimiento
url = "https://raw.githubusercontent.com/NelbaBarreto/paraguay-reddit-posts/master/muestra_posts_cripto_paraguay_etiquetado.csv"
data = pd.read_csv(url, encoding='latin1', delimiter=';', on_bad_lines='skip')



#Renombrar a la etiqueta correcta
data = data.rename(columns={'sentimiento': 'sentiment', 'cuerpo': 'comment_text', 'titulo': 'title'})

#Convertir los valores para que estén entre 0 y 1 para que se ajusten a la activación de la sigmoide
data['sentiment'] = data['sentiment'].replace({'Negativo': 0, 'Neutral': 0.5, 'Positivo': 1})

#Preprocesar, seleccionar las columnas relevantes
data = data[['comment_text', 'sentiment']] #Usar 'comment_text' y 'sentiment'

# 4. Eliminar los valores NA después de que todos los datos estén configurados
data.dropna(subset=['sentiment', 'comment_text'], inplace=True)  # Eliminar las filas con valores de sentimiento faltantes y comment_text

# ### Preprocesamiento de texto

# In[ ]:

stop_words = set(stopwords.words('spanish')) # O el idioma relevante

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', str(text), re.UNICODE) # Eliminar puntuación
    text = text.lower() # Minúsculas
    tokens = word_tokenize(text) # Tokenizar
    tokens = [word for word in tokens if word not in stop_words] # Eliminar palabras vacías
    return tokens

data['tokens'] = data['comment_text'].apply(clean_text)
# ### Crear el modelo Word2Vec

# Entrenar el modelo Word2Vec
word2vec_model = Word2Vec(sentences=data['tokens'], vector_size=100, window=5, min_count=1, workers=4)  # Ajusta los parámetros

# Guardar el modelo Word2Vec (opcional)
# word2vec_model.save("word2vec.model")

# ### Preparar los datos para la LSTM

# Obtener el tamaño del vocabulario
vocabulary_size = len(word2vec_model.wv.key_to_index) + 1  # +1 para tokens desconocidos

# Create a mapping from words to their Word2Vec vector
def get_index(tokens):
    vector = []
    for word in tokens:
        try:
          vector.append(word2vec_model.wv.key_to_index[word])
        except KeyError:
            vector.append(0) # Handle out-of-vocabulary words (add it to index 0)
    return vector

data['indexed'] = data['tokens'].apply(get_index)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['indexed'].tolist(), data['sentiment'].tolist(), test_size=0.2, random_state=42)

# Pad sequences to ensure equal length
max_sequence_length = 200 # Adjust as needed

X_train_padded = pad_sequences(X_train, maxlen=max_sequence_length, dtype='int32')
X_test_padded = pad_sequences(X_test, maxlen=max_sequence_length, dtype='int32')
# ### Crear el modelo LSTM
embedding_dim = 100  # Tamaño de los vectores Word2Vec

model = Sequential()
model.add(Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=max_sequence_length))
model.add(LSTM(128)) # Más unidades LSTM
model.add(Dense(1, activation='sigmoid')) # Capa de salida para sentimiento binario (ajusta si es multiclase)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # Ajusta la pérdida si es multiclase
print(model.summary())

# ### Entrenar el modelo LSTM

X_train_padded = np.array(X_train_padded) #Convertir a arreglos numpy
X_test_padded = np.array(X_test_padded)
y_train = np.array(y_train)
y_test = np.array(y_test)

model.fit(X_train_padded, y_train, epochs=5, batch_size=32, validation_split=0.1) # Adjust epochs and batch size

# ### Evaluar el modelo

loss, accuracy = model.evaluate(X_test_padded, y_test)
print('Accuracy: %f' % (accuracy*100))

