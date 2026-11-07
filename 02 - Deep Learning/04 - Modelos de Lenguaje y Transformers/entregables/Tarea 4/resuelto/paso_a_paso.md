# Tarea 4 — Transformers

**Clasificación de noticias de Reuters con un codificador Transformer**

Análisis completo del notebook: qué pedía la tarea, el enfoque, el código de cada celda, lo que se hizo y los resultados obtenidos.

| | |
|---|---|
| **Módulo** | Deep Learning, MLDS 5 · Unidad 4: Modelos de lenguaje y Transformers |
| **Notebook** | [v2_M5U4_Tarea_4.ipynb](v2_M5U4_Tarea_4.ipynb) · [versión explicada celda por celda](v2_M5U4_Tarea_4_Explicado.ipynb) |
| **Anexos** | [Análisis en PDF](Tarea4_Transformer_Reuters_Analisis.pdf), base de este documento · [Guía de estudio en PDF](Tarea4_Guia_de_estudio.pdf) |
| **Calificación** | UNCode · [página de la tarea](https://uncode.unal.edu.co/course/DL-GroupMLDS-5-2026-2/3feead7a-0040-47fc-b701-11492b6ea6c8) |
| **Resultado** | ✅ **100 %**: las 12 pruebas en 100 y *"Your submission grade: 100.0%"* |
| **Entorno** | Google Colab con GPU T4 · Python 3.13.16, NumPy 2.1.3 y Keras 3.13.2 |

---

## Contenido

1. [Qué pedía la tarea](#1-qué-pedía-la-tarea)
2. [El enfoque: un codificador Transformer con token CLS](#2-el-enfoque-un-codificador-transformer-con-token-cls)
3. [Conceptos que se repiten en todo el notebook](#3-conceptos-que-se-repiten-en-todo-el-notebook)
4. [Flujo general del notebook](#4-flujo-general-del-notebook)
5. [Explicación celda por celda](#5-explicación-celda-por-celda)
6. [El hilo conductor: una noticia de principio a fin](#6-el-hilo-conductor-una-noticia-de-principio-a-fin)
7. [Resultados y análisis](#7-resultados-y-análisis)
8. [Corrección necesaria para la nota máxima](#8-corrección-necesaria-para-la-nota-máxima)
9. [Otra versión de la solución](#9-otra-versión-de-la-solución)
10. [Conclusiones](#10-conclusiones)

---

## 1. Qué pedía la tarea

La tarea plantea un problema de **clasificación de texto multiclase**: dado el contenido de una noticia de la agencia Reuters, asignarla a uno de **46 temas**, como granos, petróleo, tasas de interés o adquisiciones.

El requisito central es que la clasificación se haga con un Transformer **construido pieza por pieza**. No se usa un modelo preentrenado: se implementan como capas personalizadas de Keras el embedding con posición, el bloque codificador y el ensamblaje final.

### Los datos

Se usa el dataset *Reuters newswire* incluido en Keras:

| | |
|---|---|
| **Noticias** | 11.228 etiquetadas, ya divididas en 8.982 de entrenamiento y 2.246 de validación |
| **Formato** | Cada noticia llega como una lista de enteros de longitud variable; cada entero es el índice de una palabra, ordenada por frecuencia |
| **Vocabulario** | Limitado a las 20.000 palabras más frecuentes |
| **Etiquetas** | Un entero entre 0 y 45 por noticia |

### Los cinco puntos calificados

| Punto | Función | Qué debe hacer | Pruebas |
|---|---|---|---|
| 1 | `seq_preprocess` | Llevar todas las secuencias a longitud 200, con el token `[CLS]` al inicio y el relleno al final | 1_1, 1_2, 1_3 |
| 2 | `get_EmbeddingLayer` | Capa que suma el embedding de cada palabra y el de su posición | 2_1, 2_2, 2_3 |
| 3 | `get_EncoderLayer` | Bloque codificador: atención multicabeza y red *feed-forward*, con residuales, normalización y dropout | 3_1, 3_2 |
| 4 | `get_model` | Ensamblar el modelo, tomar la salida del `[CLS]` y compilar para clasificación multiclase | 4_1, 4_2 |
| 5 | `train_model` | Preprocesar los datos crudos y entrenar 10 épocas validando en cada una | 5_1, 5_2 |

### Cómo califica el calificador

El calificador ejecuta cada función con sus propias entradas y la compara con una solución de referencia. Según la prueba, compara:
- las **formas** de los arreglos;
- **valores exactos**, como el contenido de las secuencias preprocesadas o el número de parámetros;
- la estructura del modelo.

Por eso una función puede producir la forma correcta y aun así perder puntos por el contenido, como ocurrió en el punto 1.

---

## 2. El enfoque: un codificador Transformer con token CLS

La arquitectura es el **codificador** del Transformer, del artículo *Attention Is All You Need* de Vaswani y colaboradores, 2017. A diferencia de la [Tarea 5](../../../../05%20-%20Modelos%20Generativos%20y%20Multimodalidad/entregables/Tarea%205/README.md), que usa codificador y decodificador para generar secuencias, aquí solo hace falta la mitad que "entiende": el objetivo no es generar texto, sino producir **una sola etiqueta por noticia**.

> **Codificador.** Lee la noticia completa y produce, para cada palabra, un vector que ya incorpora información de las demás. Es la parte que "entiende" la entrada.
>
> **Cabeza de clasificación.** Toma un único vector que resume la noticia y lo pasa por dos capas densas para decidir a cuál de los 46 temas pertenece. Es la parte que "decide".

El puente entre las dos partes es el token **`[CLS]`**, de *classification*, una idea tomada de BERT. Es un token artificial que se antepone a toda secuencia y no corresponde a ninguna palabra. Al pasar por el codificador, la atención hace que acumule información de toda la noticia. Al final, en lugar de promediar las 200 salidas, basta tomar la de la **posición 0** como resumen de la noticia.

El modelo es deliberadamente pequeño y aun así suma 327.802 parámetros, casi todos en la tabla de embeddings:
- un solo bloque codificador;
- embeddings de 16 dimensiones;
- 2 cabezas de atención;
- una red interna de 32 neuronas.

---

## 3. Conceptos que se repiten en todo el notebook

**Token.** La unidad mínima que procesa el modelo. Aquí cada palabra de la noticia es un token, representado por su índice entero.

**Token especial `[CLS]`.** Se antepone a toda secuencia y aquí tiene id 1. Su salida tras el codificador resume la noticia completa y es la que se usa para clasificar. El id 0 queda reservado para el relleno.

**Relleno y recorte.** Las redes necesitan entradas de tamaño fijo:
- las noticias **cortas** se completan con ceros **al final**;
- a las **largas** se les recorta **el inicio**, conservando el final.

Qué extremo se recorta fue la diferencia entre 50 y 100 en el punto 1.

**Embedding.** Una tabla entrenable que convierte cada índice en un vector denso, aquí de 16 números. Al inicio los vectores son aleatorios; con el entrenamiento se acomodan para que las palabras de significado parecido queden cerca.

**Embedding de posición.** La atención por sí sola no sabe en qué orden van los tokens. Se suma un segundo embedding que codifica la posición 0, 1, 2… En esta tarea es **aprendido**, no las funciones seno y coseno fijas del artículo original.

**Atención multicabeza.** Cada token calcula cuánto debe fijarse en cada uno de los demás y construye su nueva representación como una mezcla ponderada. Con 2 cabezas, cada una puede especializarse en un tipo de relación. Como la secuencia se mira a sí misma, `att(x, x)`, se llama **auto-atención**.

**Conexión residual y normalización de capa.** Después de cada subbloque se calcula `norm(x + salida)`. Sumar la entrada original evita que la información se pierda y facilita el flujo del gradiente; la normalización estabiliza la escala de los valores.

**Dropout.** Apaga neuronas al azar durante el entrenamiento para que el modelo no dependa de ninguna en particular. En inferencia no actúa, por eso el bloque codificador recibe el argumento `training`.

**Pérdida dispersa.** Las etiquetas son enteros como 3, 4 o 19, no vectores *one-hot* de 46 posiciones. Por eso corresponde `sparse_categorical_crossentropy`, la "forma dispersa" que menciona la nota del enunciado.

**Entrenamiento y validación.** El modelo ajusta sus parámetros con el conjunto de entrenamiento y se mide con el de validación, noticias que nunca usa para ajustar pesos. `val_loss` y `val_accuracy` son las métricas que indican si generaliza.

**Semillas.** Fijar en 0 las semillas de Python, NumPy y Keras hace que la inicialización de pesos, el dropout y el barajado den siempre lo mismo. Así las salidas pueden compararse con las del enunciado.

---

## 4. Flujo general del notebook

```
Reuters: 11.228 noticias como listas de índices de longitud variable
     │
     ▼
División ya hecha por Keras: 8.982 de entrenamiento y 2.246 de validación
     │
     ▼
Punto 1  seq_preprocess       [CLS] + recorte y relleno       → (n, 200)
     │
     ▼
Punto 2  get_EmbeddingLayer   token + posición                 → (n, 200, 16)
     │
     ▼
Punto 3  get_EncoderLayer     atención + feed-forward          → (n, 200, 16)
     │
     ▼
Punto 4  get_model            posición 0, el [CLS]             → (n, 16)
     │                        Dense 20 ReLU → Dense 46 softmax → (n, 46)
     ▼
Punto 5  train_model          Adam 1e-4, pérdida dispersa, 10 épocas
     │
     ▼
≈ 68 % de accuracy en validación → submit_task
```

---

## 5. Explicación celda por celda

Esta sección recorre el notebook en el orden en que aparece. Para cada celda se indica qué hace, se muestra el código tal como quedó, se explica línea por línea y se muestra la salida obtenida.

### 5a. Configuración y calificador

#### Celda 1 · Instalación del calificador

```python
%%capture
!pip install rlxcrypt
!wget --no-cache -O session.pye -q https://raw.githubusercontent.com/JuezUN/INGInious/master/external%20libs/session.pye
```

- `%%capture` oculta la salida de la celda y el signo `!` ejecuta un comando de terminal.
- `rlxcrypt` descifra el módulo del calificador, que viene cifrado para que no se pueda leer la solución de referencia.
- `wget` descarga ese módulo con el nombre `session.pye`.

#### Celda 2 · Inicio de sesión en UNCode

```python
import rlxcrypt
import session

grader = session.LoginSequence("DL-GroupMLDS-5-2026-2@3feead7a-0040-47fc-b701-11492b6ea6c8")
```

Crea el objeto `grader`, vinculado al curso y a la tarea, y pide el usuario y la contraseña de UNCode. Todas las celdas `grader.` dependen de él.

#### Celdas 3 y 4 · Librerías, semillas y versiones

```python
import keras
import numpy as np
import matplotlib.pyplot as plt
import os, random
%matplotlib inline
plt.style.use("ggplot")
# Seleccionamos las semillas para efectos de reproducibilidad
random.seed(0)
keras.utils.set_random_seed(0)
np.random.seed(0)
```

```
Python 3.13.16
NumPy 2.1.3
keras 3.13.2
```

Las tres semillas fijan el azar de Python, Keras y NumPy. Las versiones del entorno son más nuevas que las del enunciado, que pedía Python 3.11.12, NumPy 2.0.2 y Keras 3.8.0. Todas las pruebas pasaron; solo las métricas de entrenamiento difieren en decimales, como se ve en la sección 7.

### 5b. Carga de datos

#### Celda 5 · Carga de Reuters y diccionario de palabras

```python
vocab_size = 20000

(x_train, y_train), (x_val, y_val) = keras.datasets.reuters.load_data(num_words=vocab_size, index_from=2, path=0, start_char = None)
words = keras.datasets.reuters.get_word_index(path="reuters_word_index.json")
words = {valor: clave for clave, valor in words.items()}
print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")
```

```
8982 Training sequences
2246 Validation sequences
```

- **`num_words=20000`:** conserva solo las 20.000 palabras más frecuentes; las demás se descartan de las secuencias.
- **`index_from=2`:** desplaza los índices de las palabras dos lugares hacia arriba para dejar libres los ids bajos. Así el 0 queda para el relleno y el 1 para el `[CLS]`, que por eso se usa como `cls_id = 1` sin chocar con ninguna palabra.
- **`start_char=None`:** evita que Keras anteponga su propio token de inicio, porque el `[CLS]` se agrega en el punto 1.
- **`get_word_index`:** devuelve el diccionario palabra → índice; la última línea lo invierte a índice → palabra para reconstruir el texto.

#### Celda 6 · Ver un ejemplo

```python
print(f"texto: {' '.join([words[i] for i in x_train[0]])}")
```

```
texto: of of a not 3 move in by should 22 in rebates dollar 000 reuters four after about and may in on february said on some reuter after about revs that secretary at and which to but a right would sale 31 said end said been for reuter that earlier for reuter and which mln representation improved noted said domestic said high for reuter that under loss for reuter 000 a sources versus after about last with sale 2 was 12 said co reuter 1 vs
```

Recorre la primera noticia índice por índice, busca cada palabra en el diccionario invertido y las une con espacios.

> ⚠️ **El texto sale desordenado y no es un error de la solución.** Con `index_from=2`, cada índice de la secuencia es el índice original **más 2**. Pero `get_word_index` devuelve los índices originales, sin desplazar, y la celda de la plantilla no resta ese 2. Cada palabra impresa es, por eso, una que está dos lugares más arriba en el ranking de frecuencia. Para leer la noticia real habría que usar `words.get(i - 2)`. La celda es solo de inspección y no afecta al modelo, que trabaja con los índices tal como vienen.

### 5c. Punto 1 · Normalización de secuencias

> **Qué se pedía.**
> - Que todas las secuencias tengan la misma longitud, `max_len`, y que incluyan el token `[CLS]`.
> - Dos consideraciones explícitas: el `[CLS]` **cuenta** dentro del tamaño final y el relleno va **al final**.
> - La salida es un arreglo NumPy de forma `(n, max_len)`.

#### Celda 7 · Función calificada `seq_preprocess`

```python
# FUNCIÓN CALIFICADA seq_preprocess
def seq_preprocess(X, cls_id, max_len):
    # Se rellena/trunca a max_len - 1 (espacio para [CLS]); relleno al final
    X_pad = keras.preprocessing.sequence.pad_sequences(
        X, maxlen=max_len - 1, padding='post', value=0
    )
    cls_col = np.full((X_pad.shape[0], 1), cls_id, dtype=X_pad.dtype)
    X_out = np.concatenate([cls_col, X_pad], axis=1)
    return X_out
```

**Línea por línea:**
- **`pad_sequences(X, maxlen=max_len - 1, ...)`:** lleva cada noticia a 199 posiciones. Es `max_len - 1` para reservar el lugar del `[CLS]` y que el total sea exactamente 200.
- **`padding='post'`:** si la noticia es corta, completa con ceros al final, como exige el enunciado.
- **Recorte:** no se indica `truncating`, así que se usa el valor por defecto de Keras, `'pre'`. Si la noticia es larga, se descarta el **inicio** y se conserva el final. Es lo que espera el calificador; la sección 8 cuenta cómo se descubrió.
- **`value=0`:** el relleno usa el id 0, que gracias a `index_from=2` no corresponde a ninguna palabra.
- **`np.full((n, 1), cls_id, dtype=X_pad.dtype)`:** una columna de *n* filas llena del id del `[CLS]`, con el mismo tipo de dato que las secuencias.
- **`np.concatenate([cls_col, X_pad], axis=1)`:** pega esa columna al frente. Toda fila queda como `[1, w₁, w₂, …, 0, 0]`.

> **Por qué el `[CLS]` se agrega después de recortar.** Si se antepusiera antes, el recorte por el inicio se lo comería en todas las noticias largas.

**Ejemplo con `cls_id = 1` y `max_len = 4`:**

| Noticia | Tras `pad_sequences` a 3 | Con el `[CLS]` |
|---|---|---|
| `[5]` | `[5, 0, 0]` | `[1, 5, 0, 0]` |
| `[5, 6, 7]` | `[5, 6, 7]` | `[1, 5, 6, 7]` |
| `[8, 9, 10, 11, 12]` | `[10, 11, 12]`, se descarta el inicio | `[1, 10, 11, 12]` |

#### Celda 8 · Celda de prueba

```python
# TEST CELL
max_len = 200

x_train_p = seq_preprocess(x_train, 1, max_len = max_len)
x_val_p = seq_preprocess(x_val, 1, max_len = max_len)

print(f'Tamaño de la particion de entrenamiento:{x_train_p.shape}')
print(f'Tamaño de la particion de validacion:{x_val_p.shape}')
```

```
Tamaño de la particion de entrenamiento:(8982, 200)
Tamaño de la particion de validacion:(2246, 200)
```

Coincide con la salida esperada.

#### Celdas 9 a 11 · Pruebas 1_1, 1_2 y 1_3

- **1_1** verifica la forma y los casos sencillos.
- **1_2 y 1_3** comparan el contenido exacto de los arreglos, incluidas noticias de más de 199 palabras.

**Resultado: 100, 100 y 100**, después de la corrección de la sección 8. En la primera versión, 1_2 y 1_3 dieron 50.

### 5d. Punto 2 · Capa de codificación con token y posición

> **Qué se pedía.** Una capa personalizada que entregue, para cada token, su embedding más la codificación posicional. Recibe `max_len`, `vocab_size` y `embed_dim`.

#### Celda 12 · Función calificada `get_EmbeddingLayer`

```python
# FUNCIÓN CALIFICADA get_EmbeddingLayer
def get_EmbeddingLayer(max_len, vocab_size, embed_dim):

    class TokenAndPositionEmbedding(keras.layers.Layer):
        def __init__(self, max_len, vocab_size, embed_dim):
            super().__init__()
            self.token_emb = keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
            self.pos_emb = keras.layers.Embedding(input_dim=max_len, output_dim=embed_dim)

        def call(self, x):
            seq_len = keras.ops.shape(x)[-1]
            positions = keras.ops.arange(0, seq_len, 1)
            positions = self.pos_emb(positions)
            x = self.token_emb(x)
            return x + positions

    layer = TokenAndPositionEmbedding(max_len, vocab_size, embed_dim)
    return layer
```

**Línea por línea:**
- **Clase propia:** se hereda de `keras.layers.Layer`. `__init__` declara las piezas y `call` las usa; `super().__init__()` registra la capa en Keras.
- **`token_emb`:** tabla de 20.000 × 16, una fila entrenable por palabra. Responde a «qué palabra es».
- **`pos_emb`:** tabla de 200 × 16, una fila por posición. Responde a «dónde está».
- **`seq_len` y `arange`:** `keras.ops.shape(x)[-1]` lee la longitud, 200, y `keras.ops.arange(0, seq_len, 1)` genera `[0, 1, …, 199]`. Se usan operaciones de `keras.ops` y no de NumPy para que funcionen dentro del grafo de cómputo con cualquier *backend*.
- **`return x + positions`:** suma ambos embeddings. La suma se extiende sola a todas las noticias del lote: `(lote, 200, 16) + (200, 16)`.
- **Sangría:** la plantilla traía esta función con 2 espacios y aquí se pasó a 4, como el resto del notebook.

#### Celda 13 · Celda de prueba

```python
# TEST_CELL
max_len = 200
emb_layer = get_EmbeddingLayer(max_len, vocab_size, 128)
codified_x = emb_layer(x_train_p[:10])
print(f'Codificación de sequencias de entrenamiento: {codified_x.shape}')
```

```
Codificación de sequencias de entrenamiento: (10, 200, 128)
```

Con 10 noticias y embeddings de 128, cada entero de la matriz de 10 × 200 se convierte en un vector de 128 números.

#### Celdas 14 a 16 · Pruebas 2_1, 2_2 y 2_3

**Resultado: 100, 100 y 100**, en el primer intento.

### 5e. Punto 3 · Codificador

> **Qué se pedía.** Una capa personalizada con el bloque codificador de un Transformer. Recibe `embed_dim`, `num_heads`, `ff_dim` y `rate`, la tasa de dropout.

#### Celda 17 · Función calificada `get_EncoderLayer`

```python
# FUNCIÓN CALIFICADA get_EncoderLayer
def get_EncoderLayer(embed_dim, num_heads, ff_dim, rate):
    class TransformerBlock(keras.layers.Layer):
        def __init__(self, embed_dim, num_heads, ff_dim, rate):
            super().__init__()
            self.att = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
            self.ffn = keras.Sequential([
                keras.layers.Dense(ff_dim, activation="relu"),
                keras.layers.Dense(embed_dim),
            ])
            self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
            self.dropout1 = keras.layers.Dropout(rate)
            self.dropout2 = keras.layers.Dropout(rate)

        def call(self, inputs, training=False):
            attn_output = self.att(inputs, inputs)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(inputs + attn_output)
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            return self.layernorm2(out1 + ffn_output)

    layer = TransformerBlock(embed_dim, num_heads, ff_dim, rate)
    return layer
```

**Los componentes, en el orden en que se ejecutan dentro de `call`:**
1. **Auto-atención:** `self.att(inputs, inputs)` pasa la misma entrada como consulta y como valor, así que cada token atiende a todos los de su propia noticia. Las 2 cabezas aprenden patrones distintos en paralelo; `key_dim=embed_dim` fija el tamaño de cada proyección.
2. **Dropout 1:** apaga el 10 % de las activaciones solo al entrenar. Por eso `call` recibe `training` y lo pasa a los dos `Dropout`.
3. **Residual y normalización 1:** `layernorm1(inputs + attn_output)`. La suma deja pasar la señal original y la normalización estabiliza la escala; `epsilon=1e-6` evita divisiones por cero.
4. **Feed-forward:** `Dense(ff_dim, relu)` y luego `Dense(embed_dim)`, aplicadas a cada posición por separado. Expande a 32 dimensiones, aplica la no linealidad y vuelve a 16 para poder sumar el residual.
5. **Dropout 2, residual y normalización 2:** el mismo patrón cierra el bloque.

La salida conserva la forma de la entrada, `(lote, 200, embed_dim)`, lo que permitiría apilar varios bloques en serie.

> **Sobre `training=False`.** La plantilla definía `call(self, inputs)`. La versión resuelta añade `training` y lo pasa a los dropout de forma explícita. Durante `fit`, Keras llama la capa con `training=True`, así que el valor por defecto `False` solo aplica cuando se usa la capa fuera del entrenamiento.

#### Celda 18 · Celda de prueba

```python
# TEST_CELL
encoder_layer = get_EncoderLayer(128, 2, 32, .1)
encoded_x = encoder_layer(codified_x)
print(f'Codificación de sequencias de entrenamiento: {encoded_x.shape}')
```

```
Codificación de sequencias de entrenamiento: (10, 200, 128)
```

La forma no cambia, pero ahora cada token "sabe" de los demás.

#### Celdas 19 y 20 · Pruebas 3_1 y 3_2

**Resultado: 100 y 100**, en el primer intento.

### 5f. Punto 4 · Construir el modelo

> **Qué se pedía.**
> - Ensamblar las piezas anteriores y usar el token `[CLS]` como representación final después del codificador.
> - Emplear la activación, la pérdida y la métrica adecuadas para clasificación multiclase.
> - Una nota pide decidir si corresponde la forma dispersa de la pérdida.

#### Celda 21 · Función calificada `get_model`

```python
# FUNCIÓN CALIFICADA get_model
def get_model(max_len, vocab_size, embed_dim, num_heads, ff_dim, rate, num_clases, learning_rate, n_neurons):
    # Fijamos una semilla para efectos de reproducibiidad
    np.random.seed(0)
    keras.utils.set_random_seed(0)
    keras.backend.clear_session()

    inputs = keras.layers.Input(shape=(max_len,))
    x = get_EmbeddingLayer(max_len, vocab_size, embed_dim)(inputs)
    x = get_EncoderLayer(embed_dim, num_heads, ff_dim, rate)(x)
    x = x[:, 0, :]  # representación del token [CLS]
    x = keras.layers.Dense(n_neurons, activation="relu")(x)
    outputs = keras.layers.Dense(num_clases, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
```

**Línea por línea:**
- **Reinicio:** las semillas y `clear_session()` venían dadas. Reinician el estado de Keras para que el modelo salga idéntico en cada ejecución, incluidos los nombres de capa `dense_2` y `dense_3` de la salida esperada.
- **Entrada:** `Input(shape=(max_len,))` recibe secuencias ya preprocesadas de 200 enteros.
- **Capas:** se encadenan las capas de los puntos 2 y 3 con la API funcional; cada capa se crea y se aplica en la misma línea.
- **`x = x[:, 0, :]`, la línea más importante:** de las 200 salidas del codificador toma solo la posición 0, la del `[CLS]`, como resumen de la noticia. En el resumen del modelo aparece como la capa `get_item`, con salida `(None, 16)` y 0 parámetros.
- **Cabeza de clasificación:** `Dense(20, relu)` como capa intermedia y `Dense(46, softmax)` como salida. *Softmax* convierte los 46 puntajes en probabilidades que suman 1, la activación correcta para una sola etiqueta entre varias clases.
- **Pérdida, la decisión de la nota:** las etiquetas son enteros, no *one-hot*, así que corresponde `sparse_categorical_crossentropy`. Con etiquetas *one-hot* sería `categorical_crossentropy`. La métrica es `accuracy`.
- **Optimizador:** Adam con la tasa de aprendizaje recibida, 0,0001 en las pruebas.

#### Celda 22 · Resumen del modelo

```python
# TEST_CELL
max_len = 200
vocab_size = 20000
embed_dim = 16
num_heads = 2
ff_dim = 32
rate = 0.1
num_clases = 46
learning_rate = 1e-4
n_neurons = 20

model = get_model(max_len, vocab_size, embed_dim, num_heads, ff_dim, rate, num_clases, learning_rate, n_neurons)

model.summary()
```

| Capa y tipo | Forma de salida | Parámetros |
|---|---|---:|
| `input_layer` · InputLayer | `(None, 200)` | 0 |
| `token_and_position_embedding` · TokenAndPositionEmbedding | `(None, 200, 16)` | 323.200 |
| `transformer_block` · TransformerBlock | `(None, 200, 16)` | 3.296 |
| `get_item` · GetItem | `(None, 16)` | 0 |
| `dense_2` · Dense | `(None, 20)` | 340 |
| `dense_3` · Dense | `(None, 46)` | 966 |
| **Total, todos entrenables** | | **327.802**, 1,25 MB |

**De dónde salen los números:**

| Componente | Cálculo | Parámetros |
|---|---|---:|
| Embedding de palabras | 20.000 × 16 | 320.000 |
| Embedding de posiciones | 200 × 16 | 3.200 |
| Atención: Q, K y V | 3 × 544, con 16 × 2 × 16 pesos y 32 sesgos cada una | 1.632 |
| Atención: proyección de salida | 2 × 16 × 16 + 16 | 528 |
| Feed-forward | 16 × 32 + 32 + 32 × 16 + 16 | 1.072 |
| Dos normalizaciones | 2 × 32 | 64 |
| `dense_2` | 16 × 20 + 20 | 340 |
| `dense_3` | 20 × 46 + 46 | 966 |
| **Total** | | **327.802** |

El 98,6 % de los parámetros está en los embeddings. El bloque Transformer, que hace el trabajo de contexto, tiene apenas 3.296.

#### Celda 23 · Configuración interna

```python
import pprint

pprint.pprint(model.get_config())
```

Imprime la configuración completa del modelo: las capas, sus activaciones, `relu` y `softmax`, y cómo se conectan.

> **Diferencia con la salida esperada.** La salida esperada muestra, dentro de las capas propias, sus hiperparámetros: `embed_dim`, `max_len`, `vocab_size`, `num_heads`, `ff_dim` y `rate`. Para que aparezcan, las clases deben implementar `get_config()`. Esta versión no lo implementa y las pruebas 4_1 y 4_2 dieron 100 de todos modos, así que el calificador no revisa esa parte. La salida esperada, además, dice `embed_dim: 32` aunque el modelo se construye con 16: viene de otra ejecución.

#### Celdas 24 y 25 · Pruebas 4_1 y 4_2

**Resultado: 100 y 100**, en el primer intento.

### 5g. Punto 5 · Entrenamiento

> **Qué se pedía.** Completar `train_model`. Recibe el modelo compilado, las secuencias **crudas** de entrenamiento y validación, el `cls_id`, el `max_len` y las épocas. Debe preprocesar, entrenar y devolver `history`.

#### Celda 26 · Función calificada `train_model`

```python
# FUNCIÓN CALIFICADA train_model
def train_model(model, cls_id, X_train, Y_train, X_val, Y_val, max_len, epochs):
    X_train_p = seq_preprocess(X_train, cls_id, max_len)
    X_val_p = seq_preprocess(X_val, cls_id, max_len)

    history = model.fit(
        X_train_p, Y_train,
        validation_data=(X_val_p, Y_val),
        epochs=epochs,
    )
    return history
```

**Línea por línea:**
- **Preprocesamiento:** las dos primeras líneas aplican `seq_preprocess` a entrenamiento y validación con el mismo `cls_id` y `max_len`. La validación debe tener exactamente el mismo formato que el entrenamiento.
- **`model.fit(...)`:** entrena. No se indica `batch_size`, así que Keras usa 32: 8.982 / 32 da 281 pasos por época, el `281/281` de la salida esperada.
- **`validation_data`:** al final de cada época se miden `val_accuracy` y `val_loss` con noticias que no se usan para ajustar pesos.
- **Retorno:** `history` guarda las cuatro métricas por época.

> **Errata del enunciado.** Dice que la función devuelve «el modelo y `history`», pero la celda de prueba recibe un solo valor, `history = train_model(...)`. El modelo se entrena en el lugar, así que basta devolver `history`.

#### Celda 27 · Entrenar 10 épocas

```python
max_len = 200
epochs = 10
cls_id = 1
model = get_model(max_len, vocab_size, embed_dim, num_heads, ff_dim, rate, num_clases, learning_rate, n_neurons)
history = train_model(model, cls_id, x_train, y_train, x_val, y_val, max_len, epochs)
```

```
Epoch 1/10
281/281 ━━━━━━━━━━━━━━━━━━━━ 19s 36ms/step - accuracy: 0.3195 - loss: 2.9635 - val_accuracy: 0.4920 - val_loss: 2.5244
…
Epoch 10/10
281/281 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.7341 - loss: 1.0972 - val_accuracy: 0.6794 - val_loss: 1.3299
```

La tabla completa y su interpretación están en la sección 7.

#### Celdas 28 y 29 · Pruebas 5_1 y 5_2

**Resultado: 100 y 100.** Son las pruebas más lentas: cada una construye un modelo y lo entrena desde cero con estas funciones. Después de corregir `seq_preprocess` se volvieron a ejecutar para confirmar que seguían en 100.

### 5h. Evaluación y envío

#### Celda 30 · `grader.submit_task(globals())`

Vuelve a ejecutar las 12 pruebas en el servidor, incluidos los entrenamientos, y registra la entrega oficial.

```
TEST Test 1_1 , Your test grade: 100      TEST Test 1_2 , Your test grade: 100
TEST Test 1_3 , Your test grade: 100      TEST Test 2_1 , Your test grade: 100
TEST Test 2_2 , Your test grade: 100      TEST Test 2_3 , Your test grade: 100
TEST Test 3_1 , Your test grade: 100      TEST Test 3_2 , Your test grade: 100
TEST Test 4_1 , Your test grade: 100      TEST Test 4_2 , Your test grade: 100
TEST Test 5_1 , Your test grade: 100      TEST Test 5_2 , Your test grade: 100
Your submission grade: 100.0%
```

Hubo un envío anterior con 1_2 y 1_3 en 50, hecho antes de corregir el recorte.

---

## 6. El hilo conductor: una noticia de principio a fin

Para fijar la secuencia completa, se sigue una noticia hipotética de 500 palabras sobre resultados financieros.

| Etapa | Qué le pasa a la noticia | Forma |
|---|---|---|
| Entrada | Una lista de 500 enteros, por ejemplo `[3767, 282, …, 17, 12]` | 500 |
| Punto 1 · `seq_preprocess` | Como 500 > 199, se descartan las primeras 301 palabras y se conservan las últimas 199, el cierre de la noticia, donde suelen estar las cifras. Se antepone el 1 del `[CLS]` | 200, sin relleno |
| Punto 1, noticia corta | Una noticia de 80 palabras quedaría `[1, w₁, …, w₈₀, 0, …, 0]` con 119 ceros al final | 200 |
| Punto 2 · embedding | Cada entero se convierte en un vector de 16 números, al que se suma el de su posición. Cada fila sabe qué palabra es y dónde está | 200 × 16 |
| Punto 3 · codificador | La auto-atención redefine cada palabra mirando a las otras 199: *interest* junto a *rates* no significa lo mismo que en *public interest*. La fila del `[CLS]`, que no significaba nada, absorbe información de toda la noticia | 200 × 16 |
| Punto 4 · modelo | `x[:, 0, :]` toma solo la fila del `[CLS]`, 16 números que resumen la noticia; pasa por 20 neuronas ReLU y por la *softmax* de 46 salidas. El tema más probable es la predicción | 16 → 20 → 46 |
| Punto 5 · entrenamiento | Se repite 10 épocas sobre las 8.982 noticias en lotes de 32, ajustando los 327.802 parámetros para minimizar la entropía cruzada. Tras cada época se mide sobre las 2.246 de validación | — |

---

## 7. Resultados y análisis

### 7.1 Calificación de las pruebas

| Punto | Pruebas | Primera versión | Versión final |
|---|---|---|---|
| 1 · `seq_preprocess` | 1_1, 1_2, 1_3 | 100, 50, 50 | 100, 100, 100 |
| 2 · `get_EmbeddingLayer` | 2_1, 2_2, 2_3 | 100, 100, 100 | 100, 100, 100 |
| 3 · `get_EncoderLayer` | 3_1, 3_2 | 100, 100 | 100, 100 |
| 4 · `get_model` | 4_1, 4_2 | 100, 100 | 100, 100 |
| 5 · `train_model` | 5_1, 5_2 | 100, 100 | 100, 100 |

Las 12 pruebas quedaron en 100 y el envío oficial dio **100 %**.

### 7.2 Evolución del entrenamiento

| Época | Accuracy | Pérdida | Accuracy de validación | Pérdida de validación | Validación esperada |
|---:|---:|---:|---:|---:|---:|
| 1 | 0,3195 | 2,9635 | 0,4920 | 2,5244 | 0,4920 |
| 2 | 0,4963 | 2,3074 | 0,5392 | 2,0730 | 0,5392 |
| 3 | 0,5532 | 1,8922 | 0,5748 | 1,7553 | 0,5748 |
| 4 | 0,5941 | 1,6550 | 0,6051 | 1,6448 | 0,6051 |
| 5 | 0,6328 | 1,5320 | 0,6402 | 1,5449 | 0,6402 |
| 6 | 0,6650 | 1,4291 | 0,6532 | 1,4920 | 0,6523 |
| 7 | 0,6910 | 1,3291 | 0,6705 | 1,4318 | 0,6701 |
| 8 | 0,7110 | 1,2399 | 0,6714 | 1,3826 | 0,6705 |
| 9 | 0,7209 | 1,1683 | 0,6763 | 1,3534 | 0,6763 |
| 10 | **0,7341** | 1,0972 | **0,6794** | **1,3299** | 0,6808 |

**El modelo aprendió de forma sostenida.** La accuracy de entrenamiento subió de 32 % a 73,4 % y la de validación, de 49 % a 67,9 %. Las dos suben juntas durante las 10 épocas y la brecha final es de unos 5,5 puntos: no hay señales fuertes de sobreajuste.

**Cómo dimensionar el 67,9 %.**
- Con 46 clases, adivinar al azar daría alrededor de 2 %.
- Predecir siempre el tema más frecuente, *earn*, daría alrededor de 36 %.
- El modelo casi duplica esa segunda línea base, lo que confirma que la atención extrae señal temática real.

**Aún había margen.** La pérdida de validación seguía bajando en la época 10, de 1,3534 a 1,3299. Con más épocas, un embedding mayor o más bloques probablemente mejoraría, pero las pruebas fijan los hiperparámetros.

**Coincidencia con la referencia.** Las cinco primeras épocas de validación coinciden **exactamente** con la salida esperada, en accuracy y en pérdida. Eso indica que la arquitectura, el orden de creación de las capas y la inicialización son los mismos de la referencia. Desde la época 6 aparecen diferencias de milésimas, atribuibles al no determinismo de la GPU y a usar Keras 3.13 en lugar de 3.8. El calificador evalúa con sus propios márgenes y no las penaliza.

> **Ojo con la accuracy de entrenamiento.** Difiere de la esperada desde la época 1, 0,3195 frente a 0,1947, aunque la validación sea idéntica. Si los pesos al final de la época son los mismos, la diferencia no está en el modelo. Lo más probable es que cada versión de Keras promedie de forma distinta la métrica que muestra durante la época.

---

## 8. Corrección necesaria para la nota máxima

### El problema

La primera versión de `seq_preprocess` anteponía el `[CLS]` y luego recortaba y rellenaba con `truncating='post'`. Es decir, en las noticias largas **conservaba el inicio y descartaba el final**.

```python
# PRIMERA VERSIÓN · pruebas 1_2 y 1_3 en 50
def seq_preprocess(X, cls_id, max_len):
    X_cls = [[cls_id] + list(x) for x in X]
    X_out = keras.utils.pad_sequences(X_cls, maxlen=max_len,
                                      padding='post', truncating='post', value=0)
    return np.array(X_out)
```

### El diagnóstico

1. **La prueba 1_1 dio 100.** Verifica la forma y las noticias cortas, en las que las dos versiones coinciden porque no se recorta nada.
2. **Las pruebas 1_2 y 1_3 dieron 50**, y su retroalimentación mostraba el arreglo esperado frente al obtenido. En las filas de noticias largas, el esperado terminaba en índices distintos, y sus primeras palabras después del `[CLS]` no eran las primeras de la noticia sino unas de la mitad.
3. **Esa huella identifica el comportamiento:** la referencia conserva el **final** de la secuencia. Es el recorte `'pre'`, que además es el valor por defecto de Keras. El enunciado solo fijaba el lado del relleno, «en la parte FINAL»; el lado del recorte había que deducirlo de la retroalimentación.

### La solución y su matiz

No basta cambiar `'post'` por `'pre'` en la primera versión. Con el `[CLS]` ya antepuesto, un recorte por el inicio **eliminaría el propio `[CLS]`** en todas las noticias largas. La solución es recortar y rellenar primero a `max_len - 1` y anteponer el `[CLS]` después:

```python
# VERSIÓN FINAL · pruebas 1_1, 1_2 y 1_3 en 100
def seq_preprocess(X, cls_id, max_len):
    X_pad = keras.preprocessing.sequence.pad_sequences(
        X, maxlen=max_len - 1, padding='post', value=0
    )
    cls_col = np.full((X_pad.shape[0], 1), cls_id, dtype=X_pad.dtype)
    X_out = np.concatenate([cls_col, X_pad], axis=1)
    return X_out
```

Tras el cambio se volvieron a ejecutar la función, su celda de prueba y las pruebas 1_x, que pasaron a 100. También se repitieron 5_1 y 5_2, que usan `seq_preprocess` al entrenar, y luego se hizo el envío final.

El notebook guarda la huella de este proceso: los contadores de ejecución saltan del 6 al 11 justo en `seq_preprocess`.

| Punto | Problema | Diagnóstico | Solución |
|---|---|---|---|
| 1 | 50 en 1_2 y 1_3 | Retroalimentación del calificador: esperado frente a obtenido en noticias largas | Recortar por el inicio a `max_len - 1` y anteponer el `[CLS]` después |
| 2 a 5 | Ninguno | — | Pasaron al primer intento |

> **Lección.** En preprocesamiento de texto, qué extremo de la secuencia se descarta es una decisión de diseño con consecuencias: aquí valía 50 puntos. En noticias de agencia tiene sentido conservar el final, porque los cables de Reuters suelen cerrar con las cifras y la síntesis. Y cuando una prueba falla en parte, la retroalimentación de esperado frente a obtenido casi siempre contiene la pista exacta.

---

## 9. Otra versión de la solución

El [análisis de la tarea en PDF](Tarea4_Transformer_Reuters_Analisis.pdf) describe una ejecución distinta de la archivada en este repositorio. Las dos obtienen 12 de 12 y 100 %.

| Aspecto | Versión archivada | Versión del análisis |
|---|---|---|
| `seq_preprocess` | Recorte por defecto y `np.concatenate` | `truncating='pre'` explícito y `np.hstack`; el resultado es el mismo |
| Capas propias | Sin `get_config()` | Con `get_config()` y `**kwargs` |
| `call` del codificador | `training=False` | `training=None` |
| `fit` | `batch_size` por defecto | `batch_size=32` explícito; es el mismo valor |
| Época 10 | acc 0,7341 · val_acc 0,6794 · val_loss 1,3299 | acc 0,7383 · val_acc 0,6861 · val_loss 1,3119 |

`get_config()` no es necesario para la nota, pero hace que las capas propias sean **serializables**: sin él, un modelo guardado con `model.save` no se puede volver a cargar con sus hiperparámetros. Así se vería en la capa de embedding:

```python
class TokenAndPositionEmbedding(keras.layers.Layer):
    def __init__(self, max_len, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.max_len, self.vocab_size, self.embed_dim = max_len, vocab_size, embed_dim
        ...

    def get_config(self):
        config = super().get_config()
        config.update({"max_len": self.max_len,
                       "vocab_size": self.vocab_size,
                       "embed_dim": self.embed_dim})
        return config
```

---

## 10. Conclusiones

- **La solución:** se construyó desde cero en Keras un clasificador Transformer completo:
  - preprocesamiento con `[CLS]`;
  - embedding doble de token y posición;
  - un bloque codificador con atención multicabeza, residuales, normalización y dropout;
  - una cabeza densa con *softmax*.
- **Lo que exigía el enunciado:** relleno al final, `[CLS]` contado dentro de `max_len` y pérdida dispersa para etiquetas enteras. El lote de 32 reproduce los 281 pasos por época.
- **El desempeño:** cerca de **68 % de accuracy de validación** en 46 clases, con solo 327.802 parámetros y 10 épocas. Supera con amplitud el azar, de cerca de 2 %, y la clase mayoritaria, de cerca de 36 %, y queda prácticamente igual que la referencia del enunciado, 68,1 %.
- **La calificación:** 12 de 12 pruebas en 100 y envío oficial con 100 %.
- **La lección central:** fue de método. El lado del recorte, conservar el final de las noticias largas y anteponer el `[CLS]` después de recortar, era la única diferencia entre 50 y 100 en el punto 1, y se resolvió leyendo la retroalimentación del calificador en lugar de probar a ciegas.

---

*Créditos del notebook original: profesor Fabio Augusto González, asistente docente Juan Sebastián Malagón Torres y coordinador de virtualización Edder Hernández Forero. Universidad Nacional de Colombia, Facultad de Ingeniería.*
