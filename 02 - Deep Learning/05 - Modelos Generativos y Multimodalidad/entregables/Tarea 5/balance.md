# Tarea 5 — Balance: plantilla frente a versión resuelta

Comparación celda por celda entre la [plantilla original](plantilla/v2_M5U5_Tarea_5.ipynb) y el [notebook resuelto](resuelto/v2_M5U5_Tarea_5_Resuelto.ipynb) que se envió a UNCode.

## 1. En una línea

**Solo cambiaron las 6 funciones calificadas: 73 líneas escritas en total.** Las otras 54 celdas, incluidos todos los textos, la pérdida, la generación y las celdas del calificador, son idénticas a la plantilla. El resultado fue **11 de 11 pruebas en 100** y un **envío oficial con 100 %**.

| | Plantilla | Resuelto |
|---|---:|---:|
| Celdas | 60 | 60 |
| Celdas de código | 33 | 33 |
| Celdas modificadas | — | **6** |
| Celdas añadidas o eliminadas | — | 0 |
| Celdas de código con salida | 0 | 24 |

## 2. Lo que se escribió en cada función

| Punto | Función | Plantilla | Líneas escritas | Qué se escribió |
|---|---|---|---:|---|
| 1 | `preprocess` | 2 `pass` | 5 | Minúsculas, regex de letras y espacios, colapso de espacios, `[BOS]` y `[EOS]` en ambas columnas |
| 2 | `vectorize` | 2 variables en `None` | 8 | Dos `TextVectorization` con `seq_len + 1` y `standardize=None`, y sus `.adapt` |
| 3 | `get_EncoderLayer` | `pass`, `pass` y `layer = None` | 18 | 8 atributos en `__init__`, 8 líneas de `call` y la instancia |
| 4 | `get_DecoderLayer` | `pass`, `pass`; la instancia ya venía | 27 | 13 atributos en 14 líneas de `__init__` y 13 líneas de `call`, con máscara causal y atención cruzada |
| 5 | `get_model` | `model = None` | 7 | Dos `keras.Input` de 26 y 25, las dos capas y `keras.Model` |
| 6 | `train_model` | `history = None` | 8 | `compile` con Adam y `causal_loss`, y `fit` con el desfase `[:, :-1]` y `[:, 1:]`, más dos comentarios |
| | | **Total** | **73** | |

Casi el 62 % del código está en el codificador y el decodificador, que son el núcleo del Transformer. El código completo de cada función, explicado línea por línea, está en el [paso a paso](resuelto/paso_a_paso.md).

## 3. Lo que se dejó tal cual y por qué

| Celda de la plantilla | Por qué no se tocó |
|---|---|
| Semillas, en la celda de librerías y dentro de `get_model` | Las pruebas comparan valores numéricos exactos |
| Carga y división con `.loc` | El calificador espera exactamente 6.605 filas de entrenamiento, aunque la división real sea 67,9 / 32,1 |
| `causal_loss` | Viene dada. Obliga a que el decodificador termine en *softmax*, por su `from_logits=False` |
| Llamada `train_model(..., 1e-4, 40, 256)` | Los hiperparámetros son parte del enunciado |
| Bucle de generación | Viene dado; usa los ids fijos 3, `[BOS]`, y 2, `[EOS]` |
| Celdas de prueba, `run_test` y `submit_task` | Son el mecanismo de calificación |

## 4. Salida esperada frente a salida obtenida

| Punto | Salida esperada en la plantilla | Salida obtenida | Lectura |
|---|---|---|---|
| 1 | `[BOS] ✅ 🗳 ️ [EOS]`, emojis separados | `[BOS] ✅🗳️ [EOS]`, sin separar | **Diferencia intencional.** La salida separada daba 80; el calificador pide la instrucción escrita |
| 2 | `(6605, 26)` y `(6605, 26)` | Igual | ✅ |
| 3 | `(10, 26, 128)` | Igual | ✅ |
| 4 | Sin salida esperada | `(10, 25, 5000)` | ✅ coherente con el modelo |
| 5 | `functional_2`, 784.160 + 1.561.256 = 2.345.416 parámetros | Idéntico, hasta el nombre del modelo | ✅ |
| 6 | Época 40: loss 0,1232 · acc 0,0805 · val_loss 4,8631 · val_acc 0,0409 | Época 40: loss 3,4791 · acc 0,0502 · val_loss 3,8522 · val_acc 0,0470 | Diferente, ver abajo |

**Sobre el entrenamiento.** La referencia llega a una pérdida de entrenamiento de 0,12, es decir, casi memoriza el conjunto, pero su validación es peor: 4,86 frente a 3,85. La versión resuelta generaliza mejor y sobreajusta menos. Los valores dependen de la versión de Keras, el hardware y la inicialización, y la prueba 6_1 no los compara: calificó 100. Un dato que confirma el análisis de la métrica es que la accuracy de 0,0805 de la referencia es casi 2/25: al memorizar acierta las dos posiciones reales de cada fila, la cadena de emojis y el `[EOS]`. En validación vuelve a cerca de 1/25, porque solo acierta el `[EOS]`.

## 5. Lo que no estaba en la plantilla y hubo que descubrir

| Requisito | Dónde aparece | Cómo se descubrió | Efecto en la nota |
|---|---|---|---|
| Emojis sin separar | En la instrucción, contradicha por la salida esperada | Cuatro variantes probadas contra el calificador | `preprocess`: 80 → 100 |
| Nombres de atributos: `token_emb`, `pos_emb`, `self_att`, `self_layernorm`, `self_dropout`, `ffn`, `dropout`, `layernorm`, `cross_att`, `cross_layernorm`, `cross_dropout`, `out_dropout`, `out_layer` | No aparecen en el enunciado | El mensaje de error y una celda temporal que registraba los atributos leídos | Codificador y decodificador: 0 → 100 |
| `dropout=rate` dentro de `MultiHeadAttention` | No aparece en el enunciado | El mensaje «Se esperaba 0.1, pero se obtuvo 0.0» | Necesario para las pruebas de estructura y valores |
| `train_model` debe compilar | El enunciado dice que el modelo ya llega compilado | Lectura de `get_model`, que no compila | Ninguno: se compiló desde el primer intento |

## 6. Cómo quedó ejecutado el notebook resuelto

- **Una ejecución limpia de principio a fin:** los contadores de ejecución van de 39 a 71 sin saltos, es decir, después de los intentos de depuración se reinició y se corrió todo en orden.
- **Sin rastros de diagnóstico:** no quedan celdas añadidas, así que la celda temporal que registraba atributos se eliminó antes de la ejecución final.
- **Las 9 celdas de código sin salida son esperables:** la instalación con `%%capture`, las importaciones, las seis definiciones de funciones calificadas y `causal_loss`.
- **Entorno real:** Python 3.13.16, NumPy 2.4.6 y Keras 3.13.2 en Colab con GPU T4, versiones más nuevas que las de referencia y sin incompatibilidades.

## 7. Balance final

| Aspecto | Estado |
|---|---|
| Funciones completadas | 6 de 6 |
| Pruebas | 11 de 11 en 100 |
| Envío oficial | ✅ **100 %** |
| Fidelidad a la plantilla | Solo se modificaron las celdas calificadas |
| Plantilla original | ✅ [archivada](plantilla/v2_M5U5_Tarea_5.ipynb), con su [enunciado explicado](plantilla/enunciado.md) |
| Documentación de la solución | ✅ [paso a paso](resuelto/paso_a_paso.md) y [guía de estudio en PDF](resuelto/Tarea5_Guia_de_estudio.pdf) |
