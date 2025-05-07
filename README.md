# analisis-sentimientos-reddit-lstm
Proyecto de PLN: Análisis de Sentimientos de Reddit con Word2Vec + LSTM
Utilizando Word2Vec y Redes Neuronales LSTM
Link del Colab: https://bit.ly/4iRXUO2 Análisis de Correlación Gráficos 
Link del Colab: https://bit.ly/3GEHaMQ Accuracy: 4.545455
Diagrama de Flujo: https://bit.ly/450koZT
Autores:
●	Nelba Barreto 
●	Miguel Franco   
Resumen:
El presente trabajo explora la aplicación de técnicas de Procesamiento del Lenguaje Natural (PLN) para el análisis de sentimientos en datos textuales extraídos de la plataforma Reddit, específicamente del subreddit "Paraguay". Se implementó un modelo híbrido que combina la representación vectorial de palabras mediante Word2Vec con una arquitectura de Red Neuronal Recurrente de tipo Long Short-Term Memory (LSTM) para clasificar la polaridad de los comentarios. El proceso incluye la recolección de datos mediante la API de Reddit, preprocesamiento de texto, entrenamiento del modelo Word2Vec para generar embeddings de palabras, y el subsiguiente entrenamiento y evaluación del modelo LSTM. Los resultados iniciales muestran una precisión del 4.54% en el conjunto de prueba, lo que indica la necesidad de futuras optimizaciones y un análisis más profundo de los factores que influyen en el rendimiento del modelo.   
1. Planteamiento del Problema:
El auge de las redes sociales ha generado un vasto volumen de datos textuales que contienen opiniones y sentimientos valiosos. Analizar estas opiniones de forma automática es crucial para diversas aplicaciones, como estudios de mercado, detección de tendencias y comprensión de la opinión pública. En el contexto paraguayo, la plataforma Reddit, a través de subreddits como "Paraguay", ofrece un espacio para discusiones sobre temas locales. Sin embargo, la extracción y análisis de sentimientos en este tipo de contenido en español, y específicamente con jerga o contextos locales, presenta desafíos significativos. Este proyecto busca abordar la tarea de clasificar automáticamente los sentimientos (positivo, negativo o neutral) expresados en los comentarios del subreddit "Paraguay", explorando la efectividad de un modelo basado en Word2Vec para la representación semántica y LSTM para la clasificación secuencial.   
2. Descripción del Corpus (Base de Datos de Textos):
El corpus utilizado en este estudio se compone de dos fuentes principales:
●	Datos Recolectados de Reddit: Se extrajeron nuevos posts del subreddit "Paraguay" utilizando la API de Reddit (PRAW). Se recopilaron los últimos 100 posts (límite ajustable), combinando el título y el cuerpo del mensaje para formar el texto de análisis.    
●	Conjunto de Datos Etiquetado: Para el entrenamiento supervisado del modelo de sentimiento, se utilizó un conjunto de datos preexistente y etiquetado. Este conjunto se cargó desde una URL externa: "https://www.google.com/search?q=https://raw.githubusercontent.com/NelbaBarreto/paraguay-reddit-posts/master/muestra_posts_cripto_paraguay_etiquetado.csv". Este archivo CSV contiene columnas para 'titulo', 'cuerpo' y 'sentimiento'. Las etiquetas de sentimiento originales ('Negativo', 'Neutral', 'Positivo') se transformaron a valores numéricos (0, 0.5, 1 respectivamente) para ser compatibles con la función de activación sigmoide del modelo LSTM. Se seleccionaron las columnas 'comment_text' (derivada de 'cuerpo') y 'sentiment' para el análisis, y se eliminaron filas con valores nulos en estas columnas.   
3. Metodología:
El flujo de trabajo metodológico se puede dividir en las siguientes etapas:
3.1. Adquisición y Preparación de Datos:
●	Obtención de Datos de Reddit: Se utilizó la librería PRAW en Python para interactuar con la API de Reddit y extraer posts del subreddit "Paraguay". Se combinaron el título y el cuerpo de cada post.   
●	Carga de Datos Etiquetados: Se cargó un dataset en formato CSV con posts previamente etiquetados con sentimientos desde una URL de GitHub.   
●	Mapeo de Etiquetas de Sentimiento: Las etiquetas categóricas ('Negativo', 'Neutral', 'Positivo') se convirtieron a valores numéricos (0, 0.5, 1).   
3.2. Preprocesamiento del Texto: Se aplicaron los siguientes pasos de preprocesamiento a los datos textuales:   
●	Eliminación de Puntuación: Se removieron los signos de puntuación utilizando expresiones regulares (re.sub(r'[^\w\s]', '', str(text), re.UNICODE)).   
●	Conversión a Minúsculas: Todo el texto se convirtió a minúsculas (text.lower()).   
●	Tokenización: El texto se dividió en tokens individuales (palabras) utilizando word_tokenize de la librería NLTK.   
●	Eliminación de Palabras Vacías (Stop Words): Se eliminaron las palabras comunes que no aportan significado relevante (e.g., artículos, preposiciones) utilizando una lista de stop words en español de NLTK (stopwords.words('spanish')). Los recursos necesarios de NLTK ('punkt', 'stopwords') fueron descargados previamente.   
3.3. Generación de Embeddings de Palabras con Word2Vec:
●	Se entrenó un modelo Word2Vec utilizando la librería Gensim sobre los tokens preprocesados del conjunto de datos combinado (datos de Reddit y datos etiquetados).   
●	Hiperparámetros de Word2Vec: 
o	vector_size: 100 (dimensionalidad de los vectores de palabra).   
o	window: 5 (tamaño de la ventana de contexto).   
o	min_count: 1 (frecuencia mínima para que una palabra sea considerada).   
o	workers: 4 (número de hilos de procesamiento).   
●	El vocabulario del modelo Word2Vec se utilizó para crear un mapeo de palabras a índices numéricos. Las palabras fuera del vocabulario se mapearon a un índice 0. Los textos tokenizados se convirtieron en secuencias de estos índices (data['indexed']).   
3.4. Construcción y Entrenamiento del Modelo LSTM:
●	División de Datos: Los datos indexados y sus correspondientes etiquetas de sentimiento se dividieron en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba) utilizando train_test_split de Scikit-learn, con un random_state=42.   
●	Padding de Secuencias: Para asegurar que todas las secuencias de entrada tuvieran la misma longitud, se aplicó padding (relleno) a las secuencias de entrenamiento y prueba hasta una longitud máxima de 200 (max_sequence_length=200) utilizando pad_sequences de Keras.   
●	Arquitectura del Modelo LSTM: Se construyó un modelo secuencial en Keras con las siguientes capas: 
1.	Capa de Embedding: Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=max_sequence_length). El embedding_dim se estableció en 100, coincidiendo con el vector_size de Word2Vec. El vocabulary_size se calculó como la longitud del vocabulario de Word2Vec más uno (para tokens desconocidos).   
2.	Capa LSTM: LSTM(128) (con 128 unidades LSTM).   
3.	Capa Densa de Salida: Dense(1, activation='sigmoid') para la clasificación binaria/regresión del sentimiento (0 a 1).   
  
●	Compilación del Modelo: El modelo se compiló con el optimizador 'adam', la función de pérdida 'binary_crossentropy' (apropiada para una salida sigmoide) y la métrica 'accuracy'.   
●	Entrenamiento del Modelo: El modelo LSTM se entrenó con los datos de entrenamiento (X_train_padded, y_train) durante 5 épocas (epochs=5) y un tamaño de lote de 32 (batch_size=32). Se utilizó un 10% de los datos de entrenamiento para validación (validation_split=0.1). Las entradas y etiquetas se convirtieron a arrays de NumPy antes del entrenamiento.   
3.5. Herramientas Utilizadas:
●	Python: Lenguaje de programación principal.
●	NLTK: Para tareas de preprocesamiento de texto como tokenización y eliminación de stop words.   
●	Gensim: Para entrenar el modelo Word2Vec.   
●	Scikit-learn: Para la división de datos en conjuntos de entrenamiento y prueba.   
●	TensorFlow (Keras): Para construir, entrenar y evaluar el modelo LSTM.   
●	PRAW (Python Reddit API Wrapper): Para la recolección de datos de Reddit.   
●	Pandas: Para la manipulación y análisis de datos tabulares.   
●	NumPy: Para operaciones numéricas, especialmente con arrays.   
●	Google Colab: Entorno de notebook utilizado para la ejecución del código.   
4. Evaluación de Resultados:
4.1. Métricas: El modelo LSTM entrenado se evaluó utilizando el conjunto de prueba (X_test_padded, y_test). Las métricas reportadas fueron:   
●	Pérdida (Loss): El valor de la función de pérdida 'binary_crossentropy' en el conjunto de prueba.
●	Precisión (Accuracy): El porcentaje de predicciones correctas en el conjunto de prueba.
Resultados Cuantitativos Reportados: Tras 5 épocas de entrenamiento, la evaluación del modelo en el conjunto de prueba arrojó los siguientes resultados:   
●	Pérdida (Loss) en prueba: 0.7051   
●	Precisión (Accuracy) en prueba: 0.04545455 (o 4.545455%)   
Durante el entrenamiento, se observaron los siguientes valores de precisión y pérdida por época en el conjunto de validación:   
●	Época 1: val_loss: 0.6889, val_accuracy: 0.2222
●	Época 2: val_loss: 0.6820, val_accuracy: 0.2222
●	Época 3: val_loss: 0.6740, val_accuracy: 0.2222
●	Época 4: val_loss: 0.6653, val_accuracy: 0.2222
●	Época 5: val_loss: 0.6619, val_accuracy: 0.2222
4.2. Análisis Cualitativo: La precisión obtenida del 4.55% es extremadamente baja, lo que sugiere que el modelo actual no está aprendiendo patrones significativos para la clasificación de sentimientos en los datos proporcionados. Una precisión tan baja, cercana al azar (o incluso peor si se considera el desequilibrio de clases o la naturaleza del problema de regresión con sigmoide para tres estados mapeados a 0, 0.5 y 1), indica problemas fundamentales en el enfoque o en los datos.   
Posibles factores que contribuyen a este bajo rendimiento:
●	Cantidad y Calidad del Conjunto de Datos Etiquetado: El conjunto de datos etiquetado (muestra_posts_cripto_paraguay_etiquetado.csv ) podría ser demasiado pequeño, no representativo del lenguaje general del subreddit "Paraguay", o tener etiquetas de baja calidad. La temática ("cripto") podría ser muy específica y no generalizar bien.   
●	Manejo de Sentimiento Neutral: La conversión de 'Neutral' a 0.5 y el uso de 'binary_crossentropy' con una activación 'sigmoid' es problemático. La 'binary_crossentropy' está diseñada para problemas de clasificación binaria (0 o 1). Tratar un sentimiento neutral como 0.5 en este contexto es una aproximación que puede confundir al modelo, ya que la función de pérdida penalizará predicciones cercanas a 0 o 1 para ejemplos que son, de hecho, neutrales. Sería más apropiado tratarlo como un problema de clasificación multiclase (Negativo, Neutral, Positivo) o un problema de regresión con una función de pérdida adecuada si se busca un espectro continuo.   
●	Representatividad de Word2Vec: El modelo Word2Vec se entrenó con los tokens del conjunto de datos combinado. Si los datos de Reddit sin etiquetar son muy diferentes de los datos etiquetados, o si el vocabulario es muy ruidoso o pequeño, los embeddings pueden no ser de buena calidad.   
●	Hiperparámetros del Modelo LSTM: Aunque se utilizaron hiperparámetros comunes (LSTM con 128 unidades, optimizador Adam, 5 épocas), estos podrían no ser óptimos para este conjunto de datos específico. El número de épocas (5) es bastante bajo y podría llevar a un subajuste.   
●	Desequilibrio de Clases: No se menciona un análisis o manejo del posible desequilibrio de clases en el conjunto de datos etiquetado, lo cual puede sesgar el aprendizaje del modelo.
●	Naturaleza del Texto de Reddit: El lenguaje en Reddit puede ser informal, contener jerga, sarcasmo e ironía, lo que dificulta el análisis de sentimientos.
La precisión en validación se mantuvo constante en 0.2222 durante todas las épocas, lo que es una fuerte indicación de que el modelo no está aprendiendo. Esta precisión podría corresponder a que el modelo siempre predice la clase mayoritaria o un valor constante si el conjunto de validación es pequeño y desbalanceado.   
________________________________________
5. Conclusiones y Recomendaciones:
5.1. Conclusiones: El proyecto se propuso realizar un análisis de sentimientos en comentarios del subreddit "Paraguay" utilizando un modelo híbrido Word2Vec + LSTM. Si bien se implementó con éxito el pipeline completo, desde la recolección de datos hasta el entrenamiento y evaluación del modelo, los resultados obtenidos indican un rendimiento insatisfactorio, con una precisión final del 4.55% en el conjunto de prueba. Este resultado sugiere que el modelo, en su configuración actual, no es capaz de generalizar y clasificar adecuadamente los sentimientos en los datos proporcionados. La precisión constante en el conjunto de validación a lo largo de las épocas de entrenamiento refuerza la idea de que el modelo no está aprendiendo patrones útiles.   
Las principales limitaciones identificadas radican potencialmente en la calidad y cantidad del conjunto de datos etiquetado, la estrategia de manejo del sentimiento neutral (mapeándolo a 0.5 y usando binary_crossentropy), y la posible falta de representatividad de los embeddings de Word2Vec para la tarea específica.   
5.2. Recomendaciones: Para mejorar el rendimiento del modelo de análisis de sentimientos, se proponen las siguientes recomendaciones:
1.	Revisión y Aumento del Conjunto de Datos Etiquetado:
o	Incrementar el Tamaño: Obtener o generar un corpus etiquetado considerablemente más grande y diverso, que sea representativo del lenguaje utilizado en el subreddit "Paraguay".
o	Calidad de las Etiquetas: Asegurar la consistencia y calidad de las etiquetas mediante múltiples anotadores o directrices claras de etiquetado.
o	Balance de Clases: Analizar el balance de clases (Negativo, Neutral, Positivo) y aplicar técnicas para manejar el desequilibrio si es necesario (e.g., sobremuestreo, submuestreo, ponderación de clases).
2.	Mejorar el Manejo del Sentimiento Neutral:
o	Clasificación Multiclase: Reformular el problema como una clasificación multiclase con tres categorías (Negativo, Neutral, Positivo). Esto implicaría cambiar la capa de salida del LSTM a Dense(3, activation='softmax') y la función de pérdida a categorical_crossentropy. Las etiquetas y_train y y_test necesitarían ser codificadas en formato one-hot.
o	Ignorar Neutral (Inicialmente): Como alternativa, comenzar con un clasificador binario (Positivo vs. Negativo), excluyendo temporalmente los comentarios neutrales para simplificar el problema y luego reintroducirlos.
3.	Optimización de Embeddings de Palabras:
o	Embeddings Pre-entrenados: Considerar el uso de embeddings de palabras pre-entrenados en español (e.g., FastText, Word2Vec entrenados en corpus grandes en español) y ajustarlos (fine-tuning) con los datos específicos del proyecto.
o	Entrenamiento de Word2Vec: Si se sigue entrenando Word2Vec desde cero, asegurar que el corpus de entrenamiento sea lo suficientemente grande y relevante.
4.	Ajuste de Hiperparámetros y Arquitectura del Modelo:
o	Número de Épocas: Incrementar significativamente el número de épocas de entrenamiento (e.g., 20, 50 o más) y utilizar callbacks como EarlyStopping para evitar el sobreajuste y encontrar el punto óptimo.
o	Regularización: Incorporar técnicas de regularización como Dropout en las capas LSTM o Dense para prevenir el sobreajuste si el modelo comienza a aprender pero no generaliza.
o	Complejidad del Modelo: Experimentar con diferentes arquitecturas LSTM (e.g., más capas, diferente número de unidades, LSTM bidireccionales).
o	Tasa de Aprendizaje: Probar diferentes tasas de aprendizaje para el optimizador.
5.	Análisis Cualitativo Detallado de Errores:
o	Una vez que se obtengamos un modelo con un rendimiento base razonable, realizaremos un análisis de los errores de clasificación para entender qué tipo de comentarios son mal clasificados y por qué. Esto puede revelar problemas con el preprocesamiento, la ambigüedad del lenguaje, o la presencia de sarcasmo/ironía.
6.	Exploración de Modelos Alternativos:
o	Consideraremos modelos más avanzados como arquitecturas basadas en Transformers (e.g., BERT pre-entrenado en español como BETO) que han demostrado ser muy efectivos en tareas de PLN, incluido el análisis de sentimientos.

