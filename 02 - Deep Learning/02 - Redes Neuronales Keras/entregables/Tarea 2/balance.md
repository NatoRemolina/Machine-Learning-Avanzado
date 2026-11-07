# Tarea 2 — Balance: plantilla frente a versión resuelta

Comparación celda por celda entre la [plantilla original](plantilla/v2_M5U2_Tarea_2.ipynb) y el [notebook resuelto](resuelto/v2_M5U2_Tarea_2_Resuelto.ipynb) que se envió a UNCode.

## 1. En una línea

**Solo cambiaron las 9 funciones calificadas: 30 líneas nuevas.** Las otras 92 celdas son idénticas a la plantilla. El resultado fue **18 de 18 pruebas en 100** y un **envío oficial con 100 %**, a tiempo.

| | Plantilla | Resuelto |
|---|---:|---:|
| Celdas | 101 | 101 |
| Celdas de código | 49 | 49 |
| Celdas modificadas | — | **9** |
| Celdas añadidas o eliminadas | — | 0 |
| Celdas de código con salida | 0 | 36 |

## 2. Lo que se escribió en cada función

| Punto | Función | Plantilla | Líneas | Qué se escribió |
|---|---|---|---:|---|
| 1.1 | `one_hot_labels` | Dos variables en `None` | 3 | Número de clases a partir de ambos conjuntos y dos llamadas a `to_categorical` |
| 1.2 | `define_model` | `model = None` | 6 | `Input`, un ciclo de capas `Dense` ReLU, salida `softmax` y `keras.Model` |
| 1.3 | `compile_model` | `model = None` | 1 | `model.compile(loss, optimizer, metrics)` |
| 1.4 | `define_callbacks` | Dos variables en `None` | 4 | `ModelCheckpoint` con `save_best_only` y `save_weights_only`, y `EarlyStopping` con `restore_best_weights` |
| 1.5 | `train_model` | `history = None` | 2 | `model.fit` con `validation_split` y los *callbacks* |
| 1.6 | `evaluate_model` | `acc = -1` | 2 | `model.evaluate` y desempaquetado de pérdida y accuracy. El encabezado perdió un «:» |
| 2.1 | `define_regressor` | `model = None` | 6 | Igual que 1.2, con salida `linear` |
| 2.2 | `train_regressor` | Dos variables en `None` | 5 | Llama a `compile_model` con MAE y Adam, a `define_callbacks` sobre `val_loss` y a `train_model` |
| 2.3 | `evaluate_regreesor` | `mae = None` | 1 | `model.evaluate` y desempaquetado del MAE |
| | | **Total** | **30** | |

Es la tarea con menos código propio: la parte 2 se resuelve casi entera **reutilizando** las funciones de la parte 1.

## 3. Lo que se dejó tal cual y por qué

| Celda de la plantilla | Por qué no se tocó |
|---|---|
| Semillas al inicio y `set_random_seed(0)` dentro de varias funciones | Las pruebas comparan pérdidas y accuracies exactas |
| Divisiones con `random_state=50` y `random_state=13` | Fijan qué muestras van a entrenamiento y prueba |
| Celdas de prueba y entrenamiento largo no calificable | Son parte del enunciado |
| `pip install -U scikit-learn` | Venía en la plantilla; actualizó scikit-learn de 1.6.1 a 1.9.1 en Colab |
| `run_test` y `submit_task` | Son el mecanismo de calificación |

## 4. Salida esperada frente a salida obtenida

| Punto | Salida esperada | Salida obtenida | Lectura |
|---|---|---|---|
| Datos | `(1797, 64)`, 1.078 y 719 | Igual | ✅ |
| 1.1 | Matrices *one-hot* de 3 clases | Igual | ✅ |
| 1.2 | 1.040 + 272 + 170 = **1.482** parámetros | Igual | ✅ |
| 1.3 | Configuración de Adam con tasa 0,001 y métrica `acc` | Igual | ✅ |
| 1.4 | `monitor = loss`, `patience = 2`, `save_best_only = True`, `restore_best_only = True` | Igual | ✅ |
| 1.5 | Pérdidas 4,02000 → 2,43602 …, «improved from inf» | Mismos valores, «improved from None» | ✅ Solo cambia el texto del mensaje en Keras 3.13 |
| 1.6 | `Accuracy en test: 0.4186` | Igual | ✅ |
| Entrenamiento largo | 92,63 % | 92,63 %; se detuvo en la época 30 y restauró la 20 | ✅ |
| Datos de regresión | `((309, 10), (133, 10), 346.0, 25.0)` | `((309, 10), (133, 10), 346.0, 40.0)` | **La esperada está mal:** 25 es el mínimo de entrenamiento, no de prueba |
| 2.1 | 88 + 72 + 9 = **169** parámetros | Igual | ✅ |
| 2.2 | Primer lote con MAE 151,87; mejor época, la 48, con validación 46,276; época 50 con validación 46,7165 | Igual | ✅ Solo difiere la pérdida de entrenamiento de la época 50, 41,75 frente a 43,16, probablemente por cómo cada versión de Keras promedia la métrica durante la época |
| 2.3 | `MEA en test: 44.6665` | Igual | ✅ |

## 5. Lo que no estaba explícito y hubo que decidir

| Decisión | Qué decía el enunciado | Qué se hizo | Por qué |
|---|---|---|---|
| **Número de clases** en `one_hot_labels` | Usar `to_categorical` | Calcularlo con el máximo de **ambos** conjuntos | Si una clase faltara en uno de ellos, las matrices tendrían anchos distintos |
| **Formato del archivo** del *checkpoint* | Guardar el mejor modelo | `save_weights_only=True` | En Keras 3, una ruta `.weights.h5` exige guardar solo los pesos |
| **Pesos al terminar** | La nota pide volver a los mejores | `restore_best_weights=True` | Sin esto, el modelo queda con los pesos de la última época |
| **Paciencia en `train_regressor`** | No la fija | 10 épocas sobre `val_loss` | Coincide con el entrenamiento largo de la parte 1 y con la salida esperada |
| **Nombre `evaluate_regreesor`** | Así lo escribe | Se conservó | El calificador busca ese nombre exacto |

## 6. Cómo quedó ejecutado el notebook resuelto

- **Una ejecución limpia de principio a fin:** los contadores van del 3 al 51 sin saltos.
- **Las 13 celdas de código sin salida son esperables:** la instalación con `%%capture`, la carga de los datasets, las nueve definiciones de funciones y una celda que solo crea el regresor.
- **Entorno real:** Python 3.13.16, NumPy 2.1.3, Keras 3.13.2 y scikit-learn 1.9.1 en Colab.

## 7. Balance final

| Aspecto | Estado |
|---|---|
| Funciones completadas | 9 de 9 |
| Pruebas | 18 de 18 en 100 |
| Envío oficial | ✅ **100 %**, a tiempo |
| Fidelidad a la plantilla | Solo se modificaron las celdas calificadas |
| Plantilla original | ✅ [archivada](plantilla/v2_M5U2_Tarea_2.ipynb), con su [enunciado explicado](plantilla/enunciado.md) |
| Documentación de la solución | ✅ [paso a paso](resuelto/paso_a_paso.md), [notebook explicado](resuelto/v2_M5U2_Tarea_2_Explicado.ipynb), [guía de estudio](resuelto/Tarea2_Guia_de_estudio.pdf) y [análisis de fondo](resuelto/Tarea2_Analisis_de_fondo.pdf) |
