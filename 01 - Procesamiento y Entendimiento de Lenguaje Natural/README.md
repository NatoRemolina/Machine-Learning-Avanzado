# Módulo 1 — Procesamiento y entendimiento de lenguaje natural (SIA 2029448)

**Fechas:** del 12 de agosto al 29 de septiembre de 2026. El folleto indica que termina el 19 de septiembre, pero en Classroom la entrega final del proyecto es el 28 de septiembre y la autoevaluación el 29. **64 horas** · 4 créditos · **Estado:** terminado

**Profesor:** Felipe Restrepo Calle, PhD · **Asistentes docentes:** Juan Sebastián Lara Ramírez y Juan Sebastián Malagón Torres
En la plataforma del curso es el *Módulo 4 — Procesamiento y entendimiento del lenguaje natural* (MLDS 4). Las tareas se califican en **UNCode**, curso `PYEDLN-GroupMLDS-4-2026-2`.

## Objetivo general

Abordar los temas principales del procesamiento de lenguaje natural por medios computacionales, con énfasis en textos escritos. Incluye técnicas de lingüística computacional, análisis estadístico del lenguaje y minería de textos, aplicadas a extracción de información, clasificación y agrupamiento de textos, análisis de sentimientos y modelos de tópicos, todo con Python.

## Proyecto aplicado destacado

**[Análisis de sentimiento en noticias económicas en español](06%20-%20Proyecto%20Aplicado/README.md):** clasificador de sentimiento negativo, neutral o positivo para 7.125 oraciones financieras, con spaCy, TF-IDF y regresión logística. F1 macro de 0,66 en prueba.

## Unidades

| # | Unidad | Temas | Publicada | Carpeta |
|---|---|---|---|---|
| 0 | Bienvenida y configuración del entorno | Introducción al curso, plataformas de trabajo, presentación del proyecto y calificaciones | 12 ago | [00 - Presentacion del Curso](00%20-%20Presentacion%20del%20Curso/README.md) |
| 1 | Introducción a NLP | Manejo de archivos · obtención de textos (PDF, OCR, web) · manipulación de *strings* | 17 ago | [01 - Introduccion a NLP](01%20-%20Introduccion%20a%20NLP/README.md) |
| 2 | Preprocesamiento | Expresiones regulares · spaCy · preprocesamiento de texto | 21 ago | [02 - Preprocesamiento](02%20-%20Preprocesamiento/README.md) |
| 3 | Embeddings | Bolsa de palabras · N-grams · Word2Vec, FastText y Doc2Vec | 28 ago | [03 - Embeddings](03%20-%20Embeddings/README.md) |
| 4 | Análisis supervisado en textos | Clasificación de textos · *word tagging* · Transformers | 4 sept | [04 - Analisis Supervisado en Textos](04%20-%20Analisis%20Supervisado%20en%20Textos/README.md) |
| 5 | Análisis no supervisado | Agrupamiento · modelos de tópicos · Transformers | 11 sept | [05 - Analisis No Supervisado](05%20-%20Analisis%20No%20Supervisado/README.md) |
| 6 | Proyecto aplicado | 4 fases y entrega final | 21 ago | [06 - Proyecto Aplicado](06%20-%20Proyecto%20Aplicado/README.md) |

Cada unidad de Classroom sigue la misma estructura:
- introducción a la unidad;
- clase magistral;
- talleres guiados;
- **una tarea calificada en UNCode**;
- un taller de práctica;
- desde la unidad 2, **una fase del proyecto aplicado** con su caso de estudio;
- tutoría del grupo 1.

## Evaluación (presentación del curso, diapositiva 20)

| Componente | Peso |
|---|---|
| Talleres (tareas) | 50 % |
| Proyecto | 30 % |
| Participación | 15 % |
| Autoevaluación | 5 % |

## Calendario y seguimiento de entregas

| Entregable | Unidad | Fecha límite | Estado | Nota |
|---|---|---|---|---|
| Tarea 1: Introducción al NLP ([doc](01%20-%20Introduccion%20a%20NLP/entregables/Tarea%201/README.md)) | 1 | 27-ago-2026 | Plantilla documentada | |
| Tarea 2: preprocesamiento A ([doc](02%20-%20Preprocesamiento/entregables/Tarea%202/README.md)) | 2 | 3-sept-2026 | Plantilla documentada | |
| Proyecto aplicado · [Fase 1](06%20-%20Proyecto%20Aplicado/Proyecto/Fase%201-Entrega%201/README.md) | 2 | 3-sept-2026 | Entregada | |
| Tarea 3: embeddings ([doc](03%20-%20Embeddings/entregables/Tarea%203/README.md)) | 3 | 10-sept-2026 | Plantilla documentada | |
| Proyecto aplicado · [Fase 2](06%20-%20Proyecto%20Aplicado/Proyecto/Fase%202-Entrega%202/README.md) | 3 | 10-sept-2026 | Entregada (en el material está la versión mejorada del 17-sept) | |
| Tarea 4: supervisado ([doc](04%20-%20Analisis%20Supervisado%20en%20Textos/entregables/Tarea%204/README.md)) | 4 | 17-sept-2026 | Plantilla documentada | |
| Proyecto aplicado · [Fase 3](06%20-%20Proyecto%20Aplicado/Proyecto/Fase%203-Entrega%203/README.md) | 4 | 17-sept-2026 | Entregada | |
| Tarea 5 | 5 | 24-sept-2026 | Por cargar | |
| Proyecto aplicado · [Fase 4](06%20-%20Proyecto%20Aplicado/Proyecto/Fase%204-Entrega%204/README.md) | 5 | 24-sept-2026 | Entregada | |
| Proyecto aplicado · [Entrega final](06%20-%20Proyecto%20Aplicado/Proyecto/Entrega%20final/README.md) | 6 | 28-sept-2026 | Entregada · F1 macro 0,66 en prueba | |
| Autoevaluación | — | 29-sept-2026 | | |
| Participación | — | Sin fecha | | |

## Material pendiente de cargar
- Diapositivas de las unidades 1, 3, 4, 5 y 6.
- Talleres guiados, talleres de práctica y casos de estudio.
- Versiones resueltas de las Tareas 1 a 4, y la Tarea 5 completa.
- Enunciado y rúbrica de la fase 4 del proyecto, y notas del proyecto si las hay.

## Convenciones
- `material/`: diapositivas de clase, renombradas `UN - Tema.pdf`.
- `notas.md`: apuntes de la unidad.
- `entregables/Tarea N/`: cada tarea en dos partes. `plantilla/` guarda el notebook original sin resolver y `enunciado.md`, con lo que pide cada función. `resuelto/` guarda el notebook resuelto y `paso_a_paso.md`, con lo que se hizo, por qué y qué resultado dio. El `README.md` de la tarea indexa ambas partes.
