# Tarea 1 — Clasificación lineal

| | |
|---|---|
| **Módulo** | Deep Learning (MLDS 5) · Unidad 1: Introducción al Deep Learning (TensorFlow y Keras) |
| **Archivos** | [v2_M5U1_Tarea_1.ipynb](v2_M5U1_Tarea_1.ipynb) (plantilla sin resolver) · [v2_M5U1_Tarea_1_Resuelto.ipynb](v2_M5U1_Tarea_1_Resuelto.ipynb) (versión entregada) · [Tarea1_Analisis_de_fondo.pdf](Tarea1_Analisis_de_fondo.pdf) (informe de resultados) |
| **Calificación** | Automática en **UNCode**: [página de la tarea](https://uncode.unal.edu.co/course/DL-GroupMLDS-5-2026-2/cd16b03a-f1d7-4cc0-9a82-61ed5c8d030d) |
| **Fecha límite** | 8 de octubre de 2026, 23:59 (hora de Colombia). El notebook resuelto es del 15 de octubre: falta confirmar en UNCode la fecha del envío que cuenta |
| **Resultado** | ✅ **100 %**: las 12 pruebas en 100 y *"Your submission grade: 100.0%"* |
| **Entorno de ejecución** | Google Colab · Python 3.13.16 · TensorFlow 2.21.0 · Keras 3.13.2 (la plantilla se hizo con Python 3.11, TensorFlow 2.18 y Keras 3.8) |
| **Estructura** | 71 celdas · 6 funciones · 12 pruebas + envío final. Solo se modificaron las 6 celdas de funciones |

## 1. ¿De qué trata la tarea?

Se entrena una **regresión logística**, es decir, una red de **una sola neurona con activación sigmoide**, para distinguir entre dos tipos de vino. El conjunto de datos está **desbalanceado**, así que la pérdida se modifica para que los errores en la clase minoritaria pesen más.

| | |
|---|---|
| **Problema** | Clasificación **binaria**: vino tipo 2 frente a tipo 3 |
| **Dataset** | *Wine* (UCI): 13 características químicas del vino |
| **Modelo** | `Sequential` con una capa `Dense(1)` sigmoide: 13 pesos + 1 sesgo = **14 parámetros** |
| **Pérdida** | **Entropía cruzada binaria ponderada** por clase, escrita a mano |
| **Resultado** | **98,8 %** de accuracy en entrenamiento (82 de 83) y **91,7 %** en prueba (33 de 36) tras 10 épocas |

Es el paso previo a la Tarea 2: aquí se entrena un modelo de una neurona; en la Tarea 2 se agregan capas ocultas.

## 2. Flujo de la solución

```mermaid
flowchart LR
    A[Datos Wine<br/>clases 2 y 3] --> B[class_weights<br/>0.83 / 1.258]
    B --> C[get_loss<br/>BCE ponderada]
    D[optimizer<br/>Adam lr 0.5] --> E[get_model<br/>Dense 1 + sigmoide<br/>compilado]
    C --> E
    E --> F[train_model<br/>model.fit · 10 épocas]
    F --> G[model_predict<br/>model X → tensor]
```

## 3. Preparación de los datos

1. **Conexión a UNCode:** se crea el objeto `grader` de esta tarea. La autenticación se hizo con el usuario `nremolina`; la contraseña no queda guardada en el notebook.
2. **Carga del dataset:** `wine.data` se descarga del repositorio del curso (`mindlab-unal/mlds5-datasets`).
   - 178 vinos en total: 59 de la etiqueta 1, 71 de la 2 y 48 de la 3.
   - **Solo se usan las etiquetas 2 y 3**, que pasan a ser **0 y 1** (`y − 2`).
   - Las 13 variables se reescalan a [0, 1] con `MinMaxScaler`.
   - `train_test_split(test_size=0.3, stratify=y, random_state=42)` las divide así:

| Partición | Total | Clase 0 (vino 2) | Clase 1 (vino 3) |
|---|---:|---:|---:|
| Entrenamiento | 83 | 50 | 33 |
| Prueba | 36 | 21 | 15 |
| **Total** | **119** | 71 | 48 |

Las salidas del notebook coinciden con las esperadas.

## 4. Concepto clave: pesos por clase

Si una clase tiene menos muestras, el modelo puede ignorarla y aun así obtener un buen resultado global. La solución es **darle más peso a la clase minoritaria dentro de la pérdida**:

$$w_0=\frac{n_0+n_1}{2n_0}, \qquad w_1=\frac{n_0+n_1}{2n_1}$$

$$\mathcal{L}=-\frac{1}{N}\sum_{i=1}^{N}\Big[w_1\,y_i\log(\tilde{y}_i)+w_0\,(1-y_i)\log(1-\tilde{y}_i)\Big]$$

- Ejemplo del enunciado: $n_0=100,\ n_1=50 \Rightarrow w_0=0{,}75,\ w_1=1{,}5$. **Equivocarse con la clase 1 cuesta el doble.**
- Con estos pesos, **el peso promedio por muestra es exactamente 1**: $n_0 w_0 + n_1 w_1 = N$. Por eso la pérdida ponderada queda en la misma escala que la normal.

---

## 5. Lo que se hizo en cada función

### 5.1 Pesos de cada clase · `class_weights(y)`
**Qué pide:** una lista `[w_0, w_1]` calculada a partir de las etiquetas de entrenamiento.

**Qué se hizo:**
```python
def class_weights(y):
    n_0 = np.sum(y == 0)          # muestras de la clase 0
    n_1 = np.sum(y == 1)          # muestras de la clase 1
    w_0 = float((n_0 + n_1) / (2 * n_0))
    w_1 = float((n_0 + n_1) / (2 * n_1))
    weights_list = [w_0, w_1]
    return weights_list
```
- `np.sum(y == k)` cuenta cuántas etiquetas son iguales a `k`, porque cada `True` vale 1.
- `float(...)` convierte el resultado de NumPy a un número de Python, que es el tipo que pide el enunciado.

**Resultado:** `[0.83 1.25758]`, igual al esperado. Con $n_0 = 50$ y $n_1 = 33$: $w_0 = 83/100$ y $w_1 = 83/66$. Pruebas 1_1 y 1_2: **100**.

### 5.2 Entropía cruzada ponderada · `get_loss(class_weights)`
**Qué pide:** una función que **devuelve otra función**, `loss(y_true, y_pred)`, que calcula la pérdida ponderada.

**Qué se hizo:**
```python
def get_loss(class_weights):
    w_0, w_1 = class_weights
    def loss(y_true, y_pred):
        y_true = keras.ops.cast(y_true, y_pred.dtype)       # mismo tipo de dato
        eps = keras.backend.epsilon()
        y_pred = keras.ops.clip(y_pred, eps, 1 - eps)       # evita log(0)
        w_bce = -keras.ops.mean(w_1 * y_true * keras.ops.log(y_pred)
                                + w_0 * (1 - y_true) * keras.ops.log(1 - y_pred))
        return w_bce
    return loss
```
- **Se corrigió la firma de la plantilla**, que traía `get_loss(y_true, y_pred, class_weights)`. Quedó `get_loss(class_weights)`, como la usan las pruebas.
- **Patrón *closure*:** la función interna `loss` "recuerda" `w_0` y `w_1` de la externa. Así Keras recibe una pérdida con solo los dos argumentos que espera, `(y_true, y_pred)`.
- **Dos protecciones que el enunciado no pedía pero hacen la función más robusta:**
  - `cast` iguala el tipo de dato de las etiquetas (enteros) con el de las predicciones (decimales). Sin eso, la multiplicación puede fallar.
  - `clip(y_pred, ε, 1−ε)` evita calcular `log(0)`, que da infinito, cuando el modelo predice exactamente 0 o 1.
- `keras.ops.mean` promedia sobre todas las muestras, como sugiere la Pista 1.

**Resultado:** `weighted_bce = 0.6931471805599454` $= \log 2$, igual al esperado. Pruebas 2_1 y 2_2: **100**.

### 5.3 Optimizador · `optimizer(type_opt, learning_rate)`
**Qué se hizo:**
```python
def optimizer(type_opt, learning_rate):
    if type_opt == 'SGD':
        opt = keras.optimizers.SGD(learning_rate=learning_rate)
    elif type_opt == 'Adam':
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    elif type_opt == 'RMSprop':
        opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError("type_opt debe ser 'SGD', 'Adam' o 'RMSprop'")
    return opt
```
- Elige el optimizador según el texto recibido. Si llega un nombre no válido, falla con un mensaje claro en vez de devolver algo incorrecto.

**Resultado:** `<keras.src.optimizers.adam.Adam at 0x...>`. La ruta `keras.src` y la dirección de memoria difieren de la salida esperada solo por la versión de Keras y por la ejecución. Pruebas 3_1 y 3_2: **100**.

### 5.4 Modelo · `get_model(n_inputs, n_neurons, optimizer, loss)`
**Qué se hizo:**
```python
def get_model(n_inputs, n_neurons, optimizer, loss):
    model = keras.models.Sequential([
        keras.layers.Input(shape=(n_inputs,)),
        keras.layers.Dense(n_neurons, activation='sigmoid')
    ])
    model.compile(optimizer=optimizer, loss=loss)
    return model
```
- **Regresión logística = una capa densa con sigmoide.** La sigmoide convierte la combinación lineal $w \cdot x + b$ en una probabilidad entre 0 y 1.
- El modelo sale **ya compilado** con la pérdida ponderada y el optimizador. Por eso `train_model` puede entrenarlo directamente.

**Resultado:** una capa `Dense` con **14 parámetros** (13 + 1). El nombre `sequential_28` en vez de `sequential` solo indica que la celda se ejecutó varias veces. Pruebas 4_1 y 4_2: **100**.

### 5.5 Entrenamiento · `train_model(model, epochs, optimizer, X_t, Y_t, loss)`
**Qué se hizo:**
```python
def train_model(model, epochs, optimizer, X_t, Y_t, loss):
    history = model.fit(X_t, Y_t, epochs=epochs, verbose=0)   # ya está compilado
    losses = history.history['loss']
    return losses, model.get_weights()
```
- Se entrena con `fit`, siguiendo el ejemplo de regresión logística en Keras del taller guiado de la U1.
- `history.history['loss']` es la lista con la pérdida de cada época.
- Los argumentos `optimizer` y `loss` **no se usan dentro de la función**, porque el modelo ya se compiló con ellos en `get_model`. La firma se mantiene porque el autograder la llama así.

**Resultado:**

| | Esperado | Obtenido |
|---|---|---|
| Pérdida en la época 0 | 0.9399663209915161 | 0.9399663209915161 |
| Pérdida en la época 1 | 0.3858067989349365 | 0.3858067989349365 |
| `w[0]` | 2.2023726 | 2.2023726 |
| `b` | −0.09913202 | −0.09913211 |

La diferencia en el sesgo aparece en la séptima cifra decimal y se debe a la versión de TensorFlow. Pruebas 5_1 y 5_2: **100**.

> **¿Por qué la pérdida de la época 0 es 0,94 y no 0,69, si todos los pesos empiezan en cero?** Con pesos en cero el modelo predice 0,5 para todo, y la pérdida inicial es $\log 2 \approx 0{,}69$. Pero `fit` usa lotes de 32 muestras por defecto: con 83 muestras hay **3 actualizaciones por época** (32 + 32 + 19). La pérdida que reporta es el **promedio de los lotes**, y entre un lote y otro los pesos ya cambiaron. Con una tasa de aprendizaje alta (Adam, 0,5), el primer paso se pasa de largo y la pérdida sube en los lotes siguientes. Desde la época 1 baja con rapidez (0,39). Se comprobó al reproducir el entrenamiento: con un bucle `GradientTape` de un solo paso por época sobre todo el lote, la pérdida de la época 0 es exactamente 0,6931. La sección 6 tiene el detalle.

### 5.6 Predicciones · `model_predict(model, X_test)`
**Qué se hizo:**
```python
def model_predict(model, X_test):
    y_pred = model(tf.convert_to_tensor(X_test, dtype=tf.float32))
    return y_pred
```
- **Se corrigió la firma de la plantilla**, que traía `model_predict(log_reg, b, w, X_test)`. Quedó `model_predict(model, X_test)`, como la usan las pruebas.
- **Se llama al modelo directamente**, `model(X)`, en vez de usar `model.predict(X)`. La diferencia es clave:
  - `model(X)` devuelve un **tensor** de TensorFlow, que tiene el método `.numpy()` que usa la prueba;
  - `model.predict(X)` devuelve un arreglo de NumPy, que no lo tiene, y la prueba fallaría.
- `convert_to_tensor(..., float32)` asegura el tipo de dato que espera la capa.

**Resultado:**
- Primeras dos predicciones: `[[0.9925333] [0.0018804]]`. El primer vino es casi con seguridad de clase 1 y el segundo de clase 0.
- **Accuracy de 0,9879518**, igual al esperado.

Pruebas 6_1 y 6_2: **100**.

---

## 6. Análisis de fondo de los resultados

Esta sección resume el informe [Tarea1_Analisis_de_fondo.pdf](Tarea1_Analisis_de_fondo.pdf) (6 páginas) y lo contrasta con una **reproducción independiente** del entrenamiento. La reproducción usó las mismas versiones de Colab (TensorFlow 2.21.0 y Keras 3.13.2), las mismas semillas y los mismos pesos iniciales en cero.

### 6.1 Resultados en entrenamiento y en prueba

| Indicador | Resultado |
|---|---|
| Pruebas de UNCode | 12 de 12 con nota 100 |
| Accuracy en **entrenamiento** (83 muestras) | **98,8 %** (82 de 83) |
| Accuracy en **prueba** (36 muestras) | **91,7 %** (33 de 36) |
| Pérdida | 0,940 en la época 1 y **0,038** en la época 10 |

**Pérdida por época:**

| Época | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Pérdida | 0,940 | 0,386 | 0,243 | 0,106 | 0,105 | 0,068 | 0,045 | 0,042 | 0,042 | 0,038 |

La mayor caída ocurre en las primeras 4 épocas. Después, el modelo ya separa casi todas las muestras y la mejora es marginal.

### 6.2 Matrices de confusión
Filas = clase real; columnas = clase predicha.

| | Entrenamiento: pred. 0 | Entrenamiento: pred. 1 | Prueba: pred. 0 | Prueba: pred. 1 |
|---|---:|---:|---:|---:|
| **Real 0** (vino tipo 2) | 49 | 1 | 18 | 3 |
| **Real 1** (vino tipo 3) | 0 | 33 | 0 | 15 |

- **Ningún vino tipo 3 (la clase minoritaria) se clasificó mal**, ni en entrenamiento ni en prueba: su *recall* es del 100 %.
- **Todos los errores van en la misma dirección:** son vinos tipo 2 que el modelo clasifica como tipo 3.
- **Hay una brecha de unos 7 puntos entre entrenamiento y prueba.** Con solo 36 muestras de prueba, cada error equivale a 2,8 puntos, así que la diferencia son 3 vinos.

### 6.3 ¿Qué variables usa el modelo?
Como los datos están escalados a [0, 1], la magnitud de cada peso indica cuánto influye la variable:
- un peso **positivo** empuja hacia el vino tipo 3;
- un peso **negativo** empuja hacia el vino tipo 2.

| Variable | Peso | Favorece a | Media en el vino tipo 2 | Media en el vino tipo 3 |
|---|---:|---|---:|---:|
| Flavanoids | −6,01 | Tipo 2 | 2,08 | 0,78 |
| OD280/OD315 | −5,34 | Tipo 2 | 2,79 | 1,68 |
| Color intensity | +5,08 | Tipo 3 | 3,09 | 7,40 |
| Hue | −4,46 | Tipo 2 | 1,06 | 0,68 |
| Malic acid | +3,88 | Tipo 3 | 1,93 | 3,33 |
| Proanthocyanins | −3,08 | Tipo 2 | 1,63 | 1,15 |

El signo de cada peso coincide con los datos. **El vino tipo 3 tiene más del doble de intensidad de color y menos de la mitad de flavonoides.** El modelo aprendió las diferencias químicas reales entre los dos vinos.

### 6.4 ¿Sirvió de algo ponderar la pérdida?
El informe afirma que "la ponderación funcionó" y que los 3 errores en prueba son "el costo esperado de darle más peso a la clase minoritaria". Para comprobarlo se entrenó **el mismo modelo con pesos [1, 1]**, es decir, sin ponderar, con la misma semilla y la misma configuración:

| Pérdida | Accuracy en entrenamiento | Accuracy en prueba | Matriz de prueba |
|---|---:|---:|---|
| Sin ponderar ([1, 1]) | 98,8 % | 91,7 % | [[18, 3], [0, 15]] |
| Ponderada ([0,83; 1,258]) | 98,8 % | 91,7 % | [[18, 3], [0, 15]] |

**Las predicciones son idénticas.** En este problema las dos clases se separan casi perfectamente con una frontera lineal, y un desbalance moderado (60/40) no alcanza a cambiar ninguna decisión. Entonces:
- **No se puede atribuir** el recall del 100 % ni la dirección de los errores a la ponderación. El modelo sin ponderar hace exactamente lo mismo.
- La ponderación sigue siendo la técnica correcta para el caso general, y es lo que pide la tarea. Su efecto se notaría con un desbalance más fuerte o con clases menos separables.
- La comparación se hizo con una sola semilla y 10 épocas. Con más épocas o tasas de aprendizaje menores podría aparecer alguna diferencia, pero no en la configuración de la tarea.

### 6.5 Cómo se llegó a la solución (según el informe)
- **Se leyeron las celdas de prueba** para conocer las firmas reales de `get_loss` y `model_predict`.
- **Para `train_model` se probaron dos métodos de entrenamiento** desde pesos en cero:

  | Método | Pérdida en la época 0 | ¿Coincide con el esperado (0,9399663)? |
  |---|---:|---|
  | Bucle manual con `GradientTape` (todo el lote, 1 paso por época) | 0,6931 | No |
  | `model.fit` (lotes de 32, 3 pasos por época) | 0,9399663 | Sí |

  Se eligió `model.fit`. El valor de 0,6931 del bucle manual también se reprodujo.

- **Ajustes que exigió el autograder de UNCode** (según el informe; el notebook no guarda el historial de esos intentos):

  | Síntoma | Causa | Corrección |
  |---|---|---|
  | `name 'tf' is not defined` (nota 0) | El autograder ejecuta cada función aislada y no ve los `import` del notebook | Importar `numpy`, `tensorflow` y `keras` dentro de cada función |
  | "El tipo de dato retornado al llamar loss no es el esperado" (nota 10) | La primera versión forzaba todo a `float32` con `tf.cast` | Reescribir la pérdida con `keras.ops` y respetar el tipo de `y_pred`. Con eso pasó con 100 |

### 6.6 Verificación del informe

| Afirmación del informe | Estado | Cómo se comprobó |
|---|---|---|
| 12 pruebas en 100, pesos de clase, 0,6931, 14 parámetros, pérdidas de las épocas 0 y 1, primeras predicciones, 0,9879518 | ✅ Correctas | Salidas del notebook resuelto |
| Pérdida de las 10 épocas (0,940 → 0,038) | ✅ Correcta | Reproducción con las mismas versiones y semillas |
| 91,7 % en prueba (33 de 36) y las dos matrices de confusión | ✅ Correctas | Reproducción |
| Pesos por variable (Flavanoids −6,01 … Proanthocyanins −3,08) | ✅ Correctos | Reproducción |
| `GradientTape` de lote completo da 0,6931 en la época 0 | ✅ Correcta | Reproducción |
| "La clase 1 tiene un 34 % menos de muestras" y "un error en la minoritaria pesa 1,5 veces más" | ✅ Correctas | 33 frente a 50; 1,258 / 0,83 = 1,52 |
| "Los vinos tipo 3 tienen más color y ácido málico, y menos flavonoides, OD280 y hue" | ✅ Correcta | Medias por clase del dataset |
| Diferencia de `b` en el séptimo decimal por la versión de TensorFlow | ✅ Correcta | También aparece en la reproducción (−0,09913194) |
| **"La ponderación funcionó"** y los errores son **"el costo de darle más peso a la minoritaria"** | ❌ **No se sostiene** | Sin ponderar se obtienen predicciones idénticas (sección 6.4) |
| Errores del autograder (nota 0 y nota 10) antes de la versión final | ⚪ No verificable | El notebook no guarda ese historial. Es coherente con los `import` dentro de las funciones |

## 7. Resultado del envío

| Punto | Función | Pruebas | Nota |
|---|---|---|---|
| 1 | `class_weights` | 1_1, 1_2 | 100 · 100 |
| 2 | `get_loss` | 2_1, 2_2 | 100 · 100 |
| 3 | `optimizer` | 3_1, 3_2 | 100 · 100 |
| 4 | `get_model` | 4_1, 4_2 | 100 · 100 |
| 5 | `train_model` | 5_1, 5_2 | 100 · 100 |
| 6 | `model_predict` | 6_1, 6_2 | 100 · 100 |
| | **`grader.submit_task`** | | **100.0 %** |

## 8. Lo que se aprende en esta tarea
1. **Una regresión logística es una red neuronal de una neurona:** una capa `Dense(1)` con sigmoide. Todo lo que viene después en el curso se construye sobre esta base.
2. **El desbalance se puede corregir desde la pérdida:** los pesos por clase hacen que equivocarse con la clase minoritaria cueste más, sin quitar ni duplicar datos.
3. **Se pueden escribir pérdidas propias en Keras** con `keras.ops`, siempre que la función reciba `(y_true, y_pred)`. Para pasarle parámetros extra se usa el patrón *closure*.
4. **La estabilidad numérica importa:** recortar las predicciones con ε evita `log(0)`.
5. **`fit` reporta la pérdida promedio de los lotes de cada época**, no la de un único paso. Por eso la primera época puede mostrar una pérdida mayor que la inicial.
6. **`model(X)` y `model.predict(X)` no son lo mismo:** el primero devuelve un tensor y el segundo un arreglo de NumPy.
7. **Hay que leer las celdas de prueba:** dos firmas de la plantilla estaban mal, y la solución correcta fue seguir la forma en que las pruebas llaman a las funciones.
8. **Una técnica no siempre cambia el resultado.** Ponderar la pérdida es lo correcto ante un desbalance, pero aquí las clases se separan tan bien que sin ponderar se obtiene lo mismo. Para afirmar que una técnica "funcionó" hay que compararla con no aplicarla (ver la sección 6.4).
9. **Hay que evaluar en prueba, no solo en entrenamiento:** 98,8 % en entrenamiento contra 91,7 % en prueba.

## 9. Observaciones sobre la solución
- **El notebook nunca evalúa el conjunto de prueba.** El 98,8 % es accuracy de **entrenamiento** (82 de 83), aunque el notebook diga "sobre X_test". Al evaluar las 36 muestras de prueba se obtiene **91,7 %** (ver la sección 6).
- `class_weights` asume etiquetas 0 y 1. Para más clases habría que generalizar el cálculo, por ejemplo con `np.bincount`.
- Las funciones vuelven a importar `numpy`, `tensorflow` y `keras` adentro. **Es necesario:** el autograder ejecuta cada función en su propio espacio y no ve los `import` del notebook. Sin ellos, la prueba daba `name 'tf' is not defined` con nota 0 (sección 6.5).
- Los números de ejecución no son consecutivos (1, 2, 16…90) porque algunas celdas se ejecutaron más de una vez. No afecta los resultados: las celdas de prueba fijan la semilla y ponen los pesos en cero antes de entrenar.

## 10. Erratas del enunciado
- **Firmas de la plantilla distintas a las pruebas:**
  - `get_loss(y_true, y_pred, class_weights)` frente a `get_loss(class_weights)`;
  - `model_predict(log_reg, b, w, X_test)` frente a `model_predict(model, X_t)`.

  La solución siguió las pruebas.
- **"Tenemos entonces 122 muestras"** es un error: son **119** (83 + 36, o 71 + 48).
- **El accuracy dice "sobre X_test", pero se calcula sobre el conjunto de entrenamiento.** El valor 0,9879518 es exactamente **82/83**.
- **El comentario de la celda 27** dice "FUNCIÓN CALIFICADA weighted_bce", pero la función se llama `get_loss`.
- **El escalado usa todos los datos:** `MinMaxScaler` se ajusta antes de dividir en entrenamiento y prueba, lo que es una pequeña *fuga de información*. Lo correcto sería ajustarlo solo con el entrenamiento.
- **Typos sin efecto:** `weigths`, "vairables", "utilizrá" y "mdoelo".

## 11. Relación con los apuntes del curso
- [U1 · notas.md](../../notas.md): optimizadores (SGD, Adam, RMSprop), regresión logística en TensorFlow y en Keras, y sigmoide con entropía cruzada binaria (diapositivas 57–66).
- La [Tarea 2](../../../02%20-%20Redes%20Neuronales%20Keras/entregables/Tarea%202/documentacion.md) generaliza este modelo: pasa de una neurona sigmoide a redes con capas ocultas, de binario a multiclase (softmax) y luego a regresión. Allí se usa `model.fit` con validación y callbacks.
