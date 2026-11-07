# Tarea 1 — Balance: plantilla frente a versión resuelta

Comparación celda por celda entre la [plantilla original](plantilla/v2_M5U1_Tarea_1.ipynb) y el [notebook resuelto](resuelto/v2_M5U1_Tarea_1_Resuelto.ipynb) que se envió a UNCode.

## 1. En una línea

**Solo cambiaron las 6 funciones calificadas: 53 líneas nuevas, dos de ellas firmas corregidas.** Las otras 65 celdas son idénticas a la plantilla. El resultado fue **12 de 12 pruebas en 100** y un **envío oficial con 100 %**.

| | Plantilla | Resuelto |
|---|---:|---:|
| Celdas | 71 | 71 |
| Celdas de código | 33 | 33 |
| Celdas modificadas | — | **6** |
| Celdas añadidas o eliminadas | — | 0 |
| Celdas de código con salida | 0 | 23 |

## 2. Lo que se escribió en cada función

| Punto | Función | Plantilla | Líneas | Qué se escribió |
|---|---|---|---:|---|
| 1 | `class_weights` | Tres variables en `-1` | 10 | Conteo por clase con `np.sum`, la fórmula `N / (2 n_k)` y la lista de pesos. Incluye 3 `import` y 2 comentarios |
| 2 | `get_loss` | Firma equivocada y función interna vacía | 11 | **Firma corregida** a `get_loss(class_weights)`; función interna `loss(y_true, y_pred)` con conversión de tipo, recorte con ε y la entropía cruzada ponderada con `keras.ops` |
| 3 | `optimizer` | `opt = -1` | 11 | Selección entre `SGD`, `Adam` y `RMSprop`, y error si el nombre no es válido. Incluye 3 `import` |
| 4 | `get_model` | `model = None` | 9 | `Sequential` con `Input(13)` y `Dense(1, sigmoid)`, y `compile` con el optimizador y la pérdida |
| 5 | `train_model` | `losses = None` | 6 | `model.fit` sin mensajes y lectura de `history.history['loss']` |
| 6 | `model_predict` | Firma equivocada y `y_pred = -1` | 6 | **Firma corregida** a `model_predict(model, X_test)` y llamada directa `model(X)` sobre un tensor `float32` |
| | | **Total** | **53** | |

Las funciones 1, 3, 4, 5 y 6 importan `numpy`, `tensorflow` y `keras` dentro de sí mismas, y `get_loss` importa `keras`: son 16 de las 53 líneas.

## 3. Lo que se dejó tal cual y por qué

| Celda de la plantilla | Por qué no se tocó |
|---|---|
| Carga de *Wine*, filtro de clases, escalado y división | El calificador compara pesos y predicciones con estos mismos datos |
| Semillas y pesos iniciales en cero dentro de las celdas de prueba | Hacen que las pérdidas y los pesos sean reproducibles |
| `optimizer('Adam', 0.5)` y 10 épocas en las pruebas | Son parte del enunciado |
| `run_test` y `submit_task` | Son el mecanismo de calificación |

## 4. Salida esperada frente a salida obtenida

| Punto | Salida esperada | Salida obtenida | Lectura |
|---|---|---|---|
| Datos | 83, 36 y 13; 50 y 33 en entrenamiento; 21 y 15 en prueba | Igual | ✅ |
| 1 | `[0.83 1.25758]` | Igual | ✅ |
| 2 | `weighted_bce = 0.6931471` | `0.6931471805599454` | ✅ Es ln 2: con ŷ = 0,5 la pérdida no depende de los pesos, porque su promedio es 1 |
| 3 | `<keras.optimizers.adam.Adam at 0x…>` | `<keras.src.optimizers.adam.Adam at 0x…>` | ✅ Solo cambian la ruta interna de Keras 3.13 y la dirección de memoria |
| 4 | `sequential`, 14 parámetros | `sequential_28`, 14 parámetros | ✅ El número del nombre indica que la celda se ejecutó varias veces |
| 5 | Pérdidas 0,9399663 y 0,3858068; `w[0] = 2.2023726`; `b = -0.09913202` | Iguales, salvo `b = -0.09913211` | ✅ Diferencia en la séptima cifra, por la versión de TensorFlow |
| 6 | `[[0.9925333] [0.0018804]]` y accuracy 0,9879518 | Igual | ✅ |

## 5. Lo que no estaba en la plantilla y hubo que descubrir

| Requisito | Qué decía el enunciado | Cómo se descubrió | Efecto en la nota |
|---|---|---|---|
| **Firmas de `get_loss` y `model_predict`** | La plantilla traía otras | Leyendo cómo las llaman las celdas de prueba | Necesario para que las pruebas pudieran llamarlas |
| **`import` dentro de cada función** | No lo menciona | Mensaje `name 'tf' is not defined`, con nota 0 | El calificador ejecuta cada función aislada |
| **Tipo de dato de la pérdida** | No lo menciona | Mensaje «El tipo de dato retornado al llamar loss no es el esperado», con nota 10 | Se reescribió con `keras.ops`, respetando el tipo de `y_pred` |
| **Entrenar con `model.fit`** | Solo pide entrenar | Un bucle manual con `GradientTape` daba 0,6931 en la época 0, no el 0,9399663 esperado | `fit` usa lotes de 32 y reporta el promedio de los lotes de la época |

El notebook final no guarda el historial de esos intentos; están documentados en el [análisis de fondo](resuelto/Tarea1_Analisis_de_fondo.pdf). Los contadores de ejecución, que saltan de 2 a 16 y de 21 a 46 y 70, son coherentes con esas iteraciones.

## 6. Cómo quedó ejecutado el notebook resuelto

- **Las celdas posteriores a cada corrección se ejecutaron después de ella**, así que el resultado es coherente. Las celdas de prueba fijan la semilla y ponen los pesos en cero antes de entrenar, por lo que el orden previo no afecta.
- **Las 10 celdas de código sin salida son esperables:** la instalación con `%%capture`, las importaciones, la descarga y preparación de los datos y las seis definiciones de funciones.
- **Las notas de las pruebas no quedaron guardadas en el archivo:** el calificador las muestra como HTML que no se conserva. El 100 % consta en UNCode.
- **Entorno real:** Python 3.13.16, TensorFlow 2.21.0 y Keras 3.13.2 en Colab.

## 7. Balance final

| Aspecto | Estado |
|---|---|
| Funciones completadas | 6 de 6 |
| Pruebas | 12 de 12 en 100 |
| Envío oficial | ✅ **100 %** |
| Fidelidad a la plantilla | Solo se modificaron las celdas calificadas |
| Plantilla original | ✅ [archivada](plantilla/v2_M5U1_Tarea_1.ipynb), con su [enunciado explicado](plantilla/enunciado.md) |
| Documentación de la solución | ✅ [paso a paso](resuelto/paso_a_paso.md), [notebook explicado](resuelto/v2_M5U1_Tarea_1_Explicado.ipynb), [guía de estudio](resuelto/Tarea1_Guia_de_estudio.pdf) y [análisis de fondo](resuelto/Tarea1_Analisis_de_fondo.pdf) |
