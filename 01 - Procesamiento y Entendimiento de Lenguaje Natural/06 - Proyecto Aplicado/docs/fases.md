# Paso a paso por fase

Este documento resume lo que se hizo en cada entrega del proyecto, las decisiones que se tomaron y los resultados. Las cifras salen de las salidas de los notebooks.

| Fase | Fecha límite | Criterio de la rúbrica, 10 % del proyecto cada uno |
|---|---|---|
| 1 · Entendimiento del negocio y carga de textos | 3-sept-2026 | Objetivo del negocio, objetivo del proyecto, planeación y código de carga |
| 2 · Entendimiento y preparación del corpus | 10-sept-2026 | Análisis exploratorio, descriptivo, visualización y preprocesamiento |
| 3 · Extracción de características | 17-sept-2026 | Selección e implementación de la representación de los textos |
| 4 · Modelamiento | 24-sept-2026 | Selección, entrenamiento y evaluación de modelos |
| Entrega final | 28-sept-2026 | Proyecto completo y presentación |

---

## Fase 1 · Entendimiento del negocio y carga de textos

**Notebooks:**
- [`M4U2_Fase_1_… Propuesta.ipynb`](../Proyecto/Fase%201-Entrega%201/M4U2_Fase_1_Analisis_Sentimiento_Noticias_Economicas%20Propuesta.ipynb), ejecutado el 3-sept;
- [`M4U2_Fase_1_….ipynb`](../Proyecto/Fase%201-Entrega%201/M4U2_Fase_1_Analisis_Sentimiento_Noticias_Economicas.ipynb), del 25-sept.

Las dos versiones tienen el **mismo contenido**; la del 25-sept solo vuelve a ejecutar la carga del corpus.

1. **Introducción y trasfondo:** el sentimiento financiero depende del contexto, como en «superó las expectativas», «reducción de pérdidas» o «revisión a la baja».
2. **Marco del proyecto:**
   - preguntas clave: ¿puede el NLP clasificar el sentimiento de noticias financieras? ¿cómo apoya eso la toma de decisiones?;
   - un objetivo general: clasificar en positivo, neutral y negativo;
   - tres objetivos específicos: preprocesamiento en español, modelo de *machine learning* y evaluación.
3. **Plan:** metodología CRISP-DM y [diagrama de Gantt](../Proyecto/Diagrama%20de%20Gantt.xlsx) de 6 semanas, del 12 de agosto al 28 de septiembre. Va desde la búsqueda del negocio y del corpus hasta el modelado y la entrega final.
4. **Carga del corpus:** se lee `all_languages_clean.csv` desde Hugging Face con `pandas`, desde `hf://datasets/Kenpache/multilingual-financial-sentiment`, y se filtra `language == "es"`. Resultado: **7.125 textos con 4 columnas**: `sentence`, `label`, `source` y `language`.

## Fase 2 · Entendimiento y preparación del corpus

**Notebook:** [`M4U3_Fase_2_…_MEJORADO.ipynb`](../Proyecto/Fase%202-Entrega%202/M4U3_Fase_2_Analisis_Sentimiento_Noticias_Economicas_MEJORADO.ipynb), versión mejorada del 17-sept. La versión entregada el 10-sept no está en el material.

**1. Análisis exploratorio del corpus**

| Aspecto | Resultado |
|---|---|
| Corpus completo | 39.829 oraciones en 7 idiomas: ja 20,8 %, zh 19,9 %, **es 17,9 %**, en 17,3 %, de 12,6 %, fr 9,9 % y ar 1,6 % |
| Subconjunto en español | 7.125 textos de 14 fuentes; la principal es *estrategias_inversion* con el 24,7 % |
| Calidad | 0 nulos, 0 vacíos y 0 duplicados |
| Codificación | 435 textos con indicios de problemas: 313 con «ń», 120 con «�» y 2 con «â». Se conservan y se marcan |
| Legibilidad, índice Fernández-Huerta | Media de 72,1, legibilidad "normal" |
| Longitud | 207,6 caracteres y 33,1 palabras de media; diversidad léxica TTR de 0,87 |

**2. Variables y relaciones**
- **Clases:** negativo 1.641, neutral 3.842 y positivo 1.642. El desbalance es de 2,34:1, por eso se elige el **F1 macro** como métrica.
- **Medio de origen frente a etiqueta:** V de Cramér = 0,229, con p ≈ 8·10⁻¹⁴¹. El medio se deja **solo como variable de auditoría**, para que el modelo no aprenda el sesgo editorial.
- **Redundancia:** la longitud en caracteres y en palabras están casi perfectamente correlacionadas, con un Spearman de 0,949.

**3. Limpieza** con spaCy `es_core_news_sm`, procesada por lotes:
1. Normalización Unicode NFKC; `�` se reemplaza por espacio en lugar de adivinar el carácter.
2. Marcadores `num`, `pct` y `moneda` para cifras, porcentajes y símbolos monetarios.
3. Se eliminan la puntuación y los tokens no alfabéticos.
4. Se quitan las *stopwords* **conservando las negaciones**: no, nunca, jamás, sin, ni, tampoco, nadie y ninguno.
5. Lematización y minúsculas, conservando las tildes.

**Resultado:** 16,5 tokens por texto en promedio, ningún documento vacío y la misma distribución de clases. Se exporta `financial_sentiment_es_prepared.csv`, de 7.125 filas y 6 columnas con el texto original y el limpio, que se publica en GitHub para las fases siguientes.

## Fase 3 · Extracción de características

**Notebooks:**
- [`…_propuesta.ipynb`](../Proyecto/Fase%203-Entrega%203/M4U4_Fase_3_Analisis_Sentimiento_Noticias_Economicas_propuesta.ipynb), ejecutado el 17-sept;
- [`M4U4_Fase 3_….ipynb`](../Proyecto/Fase%203-Entrega%203/M4U4_Fase%203_Analisis_Sentimiento_Noticias_Economicas.ipynb), del 25-sept y base de la fase 4.

Las dos versiones comparan las mismas cuatro representaciones, con distinta configuración.

**Partición:** 70/15/15 → entrenamiento 4.987, ajuste 1.069 y prueba 1.069, con semilla 2402. Cada representación se ajusta **solo con entrenamiento**.

**Versión del 25-sept**, con unigramas, 512 dimensiones y validación cruzada de 5 folds sobre el conjunto de ajuste:

| Representación | Redundancia, \|r\| medio | F1 macro |
|---|---:|---:|
| BoW | 0,021 | 0,515 |
| **TF-IDF** | **0,019** | 0,517 |
| Word2Vec | 0,509 | **0,551** |
| Doc2Vec | 0,297 | 0,524 |

**Versión "propuesta"**, con partición estratificada, `min_df` y `max_df`, escala sublineal y 200 dimensiones para los densos, entrenada en entrenamiento y evaluada en ajuste:
- **F1 macro:** TF-IDF **0,610**, Word2Vec 0,598, BoW 0,592 y Doc2Vec 0,538.
- **Selección de características:** se agrega TF-IDF con **uni y bigramas** y **selección χ²**. `k = 2048` se elige con validación cruzada dentro de entrenamiento:
  - el F1 macro en ajuste es 0,663 con la selección y 0,673 sin ella;
  - el número de características baja de 5.000 a 2.048, un 59 % menos, y la memoria de 0,66 a 0,42 MB.

**Decisión:** **TF-IDF**, por su baja redundancia, su interpretabilidad, ya que cada dimensión es una palabra, y su costo bajo. En la versión del 25-sept Word2Vec tuvo el mejor F1 por unos 0,03, así que se conserva como alternativa; Doc2Vec se descarta.

**Términos más asociados a cada clase:**
- negativo: *caída, inflación, precio, guerra, riesgo*;
- positivo: *avance, impulsar, crecer, superar, ingreso, desarrollo*.

## Fase 4 · Modelamiento

**Notebook:** [`M4U5_…_fase_4_modeling.ipynb`](../Proyecto/Fase%204-Entrega%204/M4U5_Analisis_Sentimiento_noticias_fase_4_modeling.ipynb), ejecutado el 24-sept en Colab con GPU T4.

1. **Tipo de tarea:** clasificación supervisada multiclase con 3 clases.
2. **Partición:** 85/15 → entrenamiento 6.056 y prueba 1.069. Usa la misma semilla, así que la prueba es la misma de la fase 3.
3. **Modelos:** 6 *pipelines* que combinan TF-IDF o Word2Vec con regresión logística, SVM con kernel RBF o bosque aleatorio. Todos usan `class_weight="balanced"`.
4. **Búsqueda de hiperparámetros:** `GridSearchCV` de 5 folds con `scoring="f1_macro"`.
   - tamaño de la representación: 128 a 1.024;
   - regularización `C`: 0,01 a 10;
   - criterio y número de árboles del bosque.
5. **Resultados de la validación cruzada, en F1 macro:**

   | Modelo | TF-IDF | Word2Vec |
   |---|---:|---:|
   | Regresión logística | **0,647** | 0,567 |
   | SVM | 0,432 | 0,313 |
   | Bosque aleatorio | 0,451 | 0,356 |

6. **Modelo elegido:** regresión logística con TF-IDF, con `C = 1` y 1.024 términos. En prueba obtiene accuracy 0,67 y F1 macro 0,65.
7. **Análisis cualitativo:** ejemplos de aciertos y errores con las probabilidades de cada clase, y una muestra de titulares negativos con las palabras clave resaltadas. Los [borradores](../Proyecto/Fase%204-Entrega%204/borradores/) de esa figura están guardados aparte.

## Entrega final

**Notebook:** [`Entrega_final_….ipynb`](../Proyecto/Entrega%20final/Entrega_final_Analisis_Sentimiento_Noticias_Economicas.ipynb), con 188 celdas y todas ejecutadas.
**Presentación:** [`Analisis de sentimiento en Noticias economicas.pptx`](../Proyecto/Entrega%20final/Analisis%20de%20sentimiento%20en%20Noticias%20economicas.pptx), de 23 diapositivas.

El notebook integra las fases 1 a 4 en un solo flujo y repite el modelamiento:
- **Validación cruzada:** el mejor modelo vuelve a ser **regresión logística + TF-IDF**, con F1 macro de 0,646.
- **Prueba:** accuracy **0,67** y F1 macro **0,66**. Por clase:

  | Clase | F1 | Recall |
  |---|---:|---:|
  | Negativo | 0,62 | 0,71 |
  | Neutral | 0,70 | 0,62 |
  | Positivo | 0,65 | 0,74 |

- **Conclusión del equipo:** el modelo lineal sobre TF-IDF fue el más efectivo, sus errores se concentran entre clases vecinas y su mejor desempeño está en las noticias neutrales.
