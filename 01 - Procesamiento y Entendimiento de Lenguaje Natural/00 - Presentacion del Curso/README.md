# Unidad 0 — Presentación del curso

> Fuente: [U0 - Presentacion del curso.pdf](material/U0%20-%20Presentacion%20del%20curso.pdf) (22 diapositivas).

## Identificación del curso (diap. 1–2, 22)

| Campo | Dato |
|---|---|
| Programa | Programa de formación Machine Learning and Data Science MLDS |
| Institución | Facultad de Ingeniería, Universidad Nacional de Colombia |
| Módulo | Módulo 4 — *Procesamiento y Entendimiento del Lenguaje Natural* (NLP) |
| Título de la presentación | "Contenido inicial" |
| Profesor | Felipe Restrepo Calle, PhD (diap. 2; en créditos: "Felipe Restrepo-Calle, PhD") |
| Asistente docente | Juan Sebastián Lara Ramírez (diap. 22) |
| Coordinador de virtualización | Edder Hernández Forero (diap. 22) |
| Diagramadora PPT | Rosa Alejandra Superlano Esquibel (diap. 22) |
| Fecha del material | 2023-I (diap. 22) |

- Las diapositivas numeran este módulo como **Módulo 4** ("cuarto módulo del programa de formación MLDS: NLP").
- No aparece correo ni otro dato de contacto de docentes o asistentes.

## Descripción y meta (diap. 3–4)

**Bienvenida (diap. 3):**
- Aprender los conceptos fundamentales del NLP, sus aplicaciones y las particularidades de los proyectos de análisis automático de textos.
- Obtener experiencia práctica en la selección de herramientas adecuadas según la naturaleza de distintos problemas.

**Meta (diap. 4):**
- Aprender temas y conceptos fundamentales del NLP por medios computacionales, con énfasis en **textos escritos**.
- Técnicas de lingüística computacional, análisis estadístico del lenguaje y minería de textos.
- Problemas de aplicación: análisis sintáctico, extracción de información, clasificación y agrupamiento de texto, análisis de sentimientos, modelos de tópicos, entre otros.
- Todo mediante **Python** y sus librerías especializadas para NLP.

## Mapa de contenidos (diap. 5)

| Unidad | Nombre | Temas |
|---|---|---|
| 1 | Introducción al Procesamiento de Lenguaje Natural | Manejo de archivos · Obtención de textos · Manipulación de strings |
| 2 | Preprocesamiento | Preprocesamiento de texto · Expresiones regulares · Introducción a spaCy |
| 3 | Embeddings | Bolsa de palabras · N-grams · Word2Vec, FastText y Doc2Vec |
| 4 | Análisis Supervisado en Textos | Clasificación de textos · Word tagging con spaCy · Análisis supervisado con Transformers |
| 5 | Análisis no Supervisado en Textos | Agrupamiento de textos · Modelos de tópicos · Análisis no supervisado con Transformers |
| 6 | Proyecto aplicado | — |

## Objetivos de aprendizaje

### Del módulo (diap. 7)

Al finalizar el módulo se deberá ser capaz de:

1. Entender los conceptos fundamentales del NLP, sus aplicaciones y las particularidades de los proyectos de análisis automático de textos.
2. Conocer y aplicar fundamentos de programación y computación científica para el procesamiento automático de textos con *Python* y sus librerías para NLP.
3. Cargar y preprocesar un *corpus* textual con *Python* y la librería *spaCy*.
4. Entender las técnicas clásicas de extracción de características o representaciones semánticas en NLP y aplicarlas con librerías como *sklearn* y *gensim*.
5. Aplicar técnicas de aprendizaje de máquina supervisado y no supervisado en las tareas típicas de NLP para un análisis detallado de información textual.

### Por unidad (diap. 8–13)

| Unidad | Objetivos |
|---|---|
| U1 — Introducción a NLP (diap. 8) | 1) Describir conceptos generales del NLP, sus retos y los tipos de aplicaciones actuales. 2) Usar librerías y utilidades de *Python* para obtener información textual. 3) Manipular textos como cadenas en *Python*: operaciones con *strings*, formatos, búsquedas exactas y mediante patrones. |
| U2 — Preprocesamiento (diap. 9) | 1) Describir la relación entre un proyecto de NLP y uno de Ciencia de Datos/*Machine Learning*. 2) Entender las expresiones regulares para búsqueda y limpieza. 3) Seleccionar la estrategia de preprocesamiento más adecuada para diversos datos textuales. 4) Usar *spaCy* y entender sus módulos y componentes básicos. |
| U3 — *Embeddings* (diap. 10) | 1) Entender la extracción de características y la representación semántica. 2) Usar *embeddings* clásicos (bolsas de palabras, *Word2Vec*, *Doc2Vec*, *FastText*) para representaciones vectoriales. 3) Usar *sklearn* y *gensim*. |
| U4 — Análisis Supervisado en Textos (diap. 11) | 1) Entender cómo se aplica el aprendizaje supervisado en NLP. 2) Seleccionar el tipo de tarea supervisada según la necesidad de clasificación de texto. 3) Usar modelos preentrenados para clasificar palabras o documentos y entrenar modelos personalizados para una tarea específica. |
| U5 — Análisis no Supervisado en Textos (diap. 12) | 1) Entender cómo se aplica el aprendizaje no supervisado en NLP. 2) Seleccionar el tipo de tarea no supervisada según la necesidad de análisis. 3) Usar modelos de agrupamiento de texto y de tópicos. |
| U6 — Proyecto aplicado (diap. 13) | 1) Ejecutar un proyecto de procesamiento y entendimiento automático de lenguaje natural usando las técnicas y herramientas del curso, para hallar información relevante en los textos analizados. |

La diap. 6 es solo una portada de sección ("Objetivos de aprendizaje") con una imagen.

## Palabras clave del módulo (diap. 14)

Nube de palabras: Chunk, Strings, Token, Etiquetas, OCR, Vocabularios, sklearn, Bolsa de palabras, Corpus, Agrupamiento de textos, Regex, Análisis de sentimientos, Clasificación de textos, Web scrapping, HuggingFace, Gensim, Tópicos, spaCy, Documento, Lenguaje.

## Conceptos previos (diap. 15)

- Programación básica en Python.
- Uso de librerías científicas: NumPy, matplotlib, scikit-learn, pandas.

## Habilidades y competencias a desarrollar (diap. 16–19)

| Saber (diap. 17) | Saber hacer (diap. 18) | Saber ser (diap. 19) |
|---|---|---|
| Qué es el Procesamiento de Lenguaje Natural | Uso de *spaCy* para NLP en general | Entendimiento de lenguajes a través de la lingüística computacional |
| Librerías de Python para el NLP | Uso de *Transformers* para modelos de NLP basados en redes neuronales | Solucionar problemas relacionados con información en lenguajes humanos |
| Adquisición, manipulación, procesamiento y análisis de textos | Uso de *Gensim* para modelos de *embeddings* | Análisis de lenguajes con técnicas computacionales |
| Aplicación de análisis supervisado en textos | Implementación de modelos de *Machine Learning* y *scikit-learn* para NLP | Capacidad de usar modelos de Deep learning para NLP en escenarios más prácticos |

La diap. 16 es solo portada de sección con imagen.

## Evaluación formativa del módulo (diap. 20)

| # | Componente | Peso |
|---|---|---|
| 1 | Talleres | 50 % |
| 2 | Proyecto | 30 % |
| 3 | Participación | 15 % |
| 4 | Autoevaluación | 5 % |
| | **Total** | **100 %** |

- No se detalla el número de talleres ni el peso de cada uno.

## Metodología (diap. 3–4, 21)

- Enfoque teórico-práctico: conceptos fundamentales + experiencia práctica en selección de herramientas.
- Trabajo en Python con librerías especializadas.
- Cierre (diap. 21): tras la introducción, se invita a continuar "en la plataforma" con la primera unidad del módulo.
- No hay una diapositiva específica de metodología (sesiones, modalidad, horarios).

## Herramientas y plataformas (diap. 4, 7, 10, 14–15, 18)

| Tipo | Elementos |
|---|---|
| Lenguaje | Python |
| Librerías NLP | spaCy, Gensim, Transformers / HuggingFace |
| ML y computación científica | scikit-learn (sklearn), NumPy, pandas, matplotlib |
| Plataforma del curso | Se menciona "la plataforma" (diap. 21) sin nombrarla |

## Cronograma y fechas

- No hay cronograma ni fechas de entrega en estas diapositivas.
- Única fecha: "2023-I" en los créditos (diap. 22), que corresponde a la elaboración del material.
