# Tarea 4 — Balance: plantilla frente a versión resuelta

Comparación celda por celda entre la [plantilla original](plantilla/v2_M5U4_Tarea_4.ipynb) y el [notebook resuelto](resuelto/v2_M5U4_Tarea_4.ipynb) que se envió a UNCode.

## 1. En una línea

**Solo cambiaron las 5 funciones calificadas: 51 líneas nuevas o modificadas.** Las otras 50 celdas son idénticas a la plantilla. El resultado fue **12 de 12 pruebas en 100** y un **envío oficial con 100 %**.

| | Plantilla | Resuelto |
|---|---:|---:|
| Celdas | 55 | 55 |
| Celdas de código | 30 | 30 |
| Celdas modificadas | — | **5** |
| Celdas añadidas o eliminadas | — | 0 |
| Celdas de código con salida | 0 | 23 |

## 2. Lo que se escribió en cada función

| Punto | Función | Plantilla | Líneas | Qué se escribió |
|---|---|---|---:|---|
| 1 | `seq_preprocess` | `X_out = None` | 6 | `pad_sequences` a `max_len - 1` con relleno al final, una columna con el `[CLS]` y `np.concatenate` para anteponerla. Incluye 1 comentario |
| 2 | `get_EmbeddingLayer` | `pass`, `pass` y `layer = None` | 8 | `token_emb` y `pos_emb`, 5 líneas de `call` que suman ambos embeddings y la instancia. Además se pasó la sangría de 2 a 4 espacios |
| 3 | `get_EncoderLayer` | `pass`, `pass` y `layer = None` | 17 | Atención, red de 2 capas, 2 normalizaciones y 2 dropout; `call` con el argumento `training`, 6 líneas y la instancia |
| 4 | `get_model` | `model = None` | 12 | Entrada de 200, embedding, codificador, salida del `[CLS]` con `x[:, 0, :]`, `Dense` de 20 con ReLU, `Dense` de 46 con *softmax* y `compile` con pérdida dispersa |
| 5 | `train_model` | `history = None` | 8 | Preprocesar entrenamiento y validación con `seq_preprocess` y `fit` con `validation_data`. La firma solo perdió un espacio |
| | | **Total** | **51** | |

El codificador y `get_model` concentran el 57 % del código. Las dos líneas de firma modificadas, `call(..., training=False)` y `def train_model(`, cuentan dentro del total.

## 3. Lo que se dejó tal cual y por qué

| Celda de la plantilla | Por qué no se tocó |
|---|---|
| Semillas, en la celda de librerías y dentro de `get_model` | Las pruebas comparan valores exactos y nombres de capas |
| Carga de Reuters con `index_from=2` y `start_char=None` | Deja libres los ids 0, relleno, y 1, que la prueba usa como `[CLS]` |
| Celdas de prueba con sus hiperparámetros: `max_len = 200`, `embed_dim = 16`, 46 clases, `1e-4` y 10 épocas | Son parte del enunciado |
| `pprint.pprint(model.get_config())` | Celda de verificación del enunciado |
| `run_test` y `submit_task` | Son el mecanismo de calificación |

## 4. Salida esperada frente a salida obtenida

| Punto | Salida esperada en la plantilla | Salida obtenida | Lectura |
|---|---|---|---|
| 1 | `(8982, 200)` y `(2246, 200)` | Igual | ✅ |
| 2 | `(10, 200, 128)` | Igual | ✅ |
| 3 | `(10, 200, 128)` | Igual | ✅ |
| 4 · resumen | `functional_1`, 323.200 + 3.296 + 340 + 966 = **327.802 parámetros** | Idéntico, incluidos los nombres `dense_2` y `dense_3` | ✅ |
| 4 · `get_config` | Las capas propias muestran `embed_dim`, `max_len`, `vocab_size`, `num_heads`, `ff_dim` y `rate` | No aparecen, porque las capas no implementan `get_config()` | Las pruebas 4_1 y 4_2 dieron 100 de todos modos. Además, la salida esperada dice `embed_dim: 32` aunque el modelo se construye con 16 |
| 5 · validación | Épocas 1 a 5: 0,4920 · 0,5392 · 0,5748 · 0,6051 · 0,6402 | **Idénticas** hasta el cuarto decimal, también en la pérdida | ✅ |
| 5 · final | Época 10: acc 0,7228 · val_acc **0,6808** · val_loss 1,3179 | Época 10: acc 0,7341 · val_acc **0,6794** · val_loss 1,3299 | 0,14 puntos de diferencia, atribuible a las versiones del entorno |

**Sobre el entrenamiento.** Que la validación coincida exactamente durante cinco épocas indica que el modelo y su inicialización son los mismos de la referencia: misma arquitectura, mismo orden de creación de capas y mismas semillas. Desde la época 6 aparecen diferencias mínimas, propias del no determinismo de la GPU y de Keras 3.13 frente a 3.8. La accuracy de entrenamiento difiere desde la época 1, 0,3195 frente a 0,1947, aunque la validación sea idéntica. Lo más probable es que se deba a cómo cada versión de Keras promedia la métrica durante la época, no a una diferencia del modelo.

## 5. Lo que no estaba explícito y hubo que descubrir

| Requisito | Qué decía el enunciado | Cómo se descubrió | Efecto en la nota |
|---|---|---|---|
| **De qué lado se recorta** una noticia larga | Solo que el relleno va al final | Las pruebas 1_2 y 1_3 dieron 50 con un recorte por el final. Su retroalimentación mostraba que la referencia conserva el **final** de la noticia | `seq_preprocess`: 50 → 100 en 1_2 y 1_3 |
| **Cuándo se antepone el `[CLS]`** | Que cuenta dentro de `max_len` | Al recortar por el inicio con el `[CLS]` ya puesto, el propio `[CLS]` se perdería. Hay que recortar a `max_len - 1` y anteponerlo después | Parte de la misma corrección |
| **Pérdida dispersa** | Una nota pide decidir si usarla | Las etiquetas son enteros de 0 a 45, no vectores *one-hot* | 100 al primer intento |
| **`get_config()`** | La salida esperada lo sugiere | Las pruebas pasaron sin implementarlo | No hizo falta |

La primera versión de `seq_preprocess` anteponía el `[CLS]` y luego recortaba con `truncating='post'`. Esa versión no quedó en el notebook; el detalle está en el análisis de la tarea. El notebook sí conserva la huella del proceso: los contadores de ejecución saltan del 6 al 11 justo en `seq_preprocess`, lo que corresponde a los intentos previos dentro de la misma sesión.

## 6. Cómo quedó ejecutado el notebook resuelto

- **No es una corrida limpia de principio a fin.** Los contadores van del 1 al 35 con dos saltos: del 6 al 11, en la corrección de `seq_preprocess`, y del 19 al 21, entre las pruebas 2_2 y 2_3. Todas las celdas posteriores a la corrección se ejecutaron después de ella, así que el resultado es coherente.
- **Las 7 celdas de código sin salida son esperables:** la instalación con `%%capture`, las importaciones y las cinco definiciones de funciones.
- **Entorno real:** Python 3.13.16, NumPy 2.1.3 y Keras 3.13.2 en Colab con GPU T4, frente a 3.11.12, 2.0.2 y 3.8.0 de la referencia.

## 7. Otra versión descrita en el análisis

El análisis de la tarea en PDF describe una versión del notebook **distinta de la archivada**, que no está en el repositorio:

| Aspecto | Versión archivada | Versión del análisis |
|---|---|---|
| `seq_preprocess` | `truncating` por defecto y `np.concatenate` | `truncating='pre'` explícito y `np.hstack`; el resultado es el mismo |
| Capas propias | Sin `get_config()` | Con `get_config()` y `**kwargs` |
| `call` del codificador | `training=False` | `training=None` |
| Época 10 | acc 0,7341 · val_acc 0,6794 · val_loss 1,3299 | acc 0,738 · val_acc 0,686 · val_loss 1,3119 |
| Calificación | 12 de 12 y 100 % | 12 de 12 y 100 % |

Las dos versiones obtienen la nota máxima. Si se archiva la del análisis, conviene guardarla junto a esta como segunda ejecución.

## 8. Balance final

| Aspecto | Estado |
|---|---|
| Funciones completadas | 5 de 5 |
| Pruebas | 12 de 12 en 100 |
| Envío oficial | ✅ **100 %** |
| Fidelidad a la plantilla | Solo se modificaron las celdas calificadas |
| Plantilla original | ✅ [archivada](plantilla/v2_M5U4_Tarea_4.ipynb), con su [enunciado explicado](plantilla/enunciado.md) |
| Documentación de la solución | [Resultado y observaciones](resuelto/README.md); el paso a paso detallado está pendiente |
