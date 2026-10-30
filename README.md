# Machine Learning Avanzado

Material, apuntes y entregables del **Diplomado Machine Learning and Data Science Avanzado** de la Facultad de Ingeniería, Universidad Nacional de Colombia (sede Bogotá), cohorte 2026-2.

**Autora:** Natalia Patricia Remolina Rodríguez

El diplomado suma 192 horas en 3 módulos de 64 horas, con 4 créditos homologables cada uno. Fuente: [Machine learning Avanzado.pdf](Machine%20learning%20Avanzado.pdf).

## Proyecto destacado

**[Análisis de sentimiento en noticias económicas en español](01%20-%20Procesamiento%20y%20Entendimiento%20de%20Lenguaje%20Natural/06%20-%20Proyecto%20Aplicado/README.md):** proyecto aplicado del módulo de NLP. Corpus de 7.125 oraciones financieras, preprocesamiento con spaCy, comparación de BoW, TF-IDF, Word2Vec y Doc2Vec, y modelo final TF-IDF con regresión logística, con F1 macro de 0,66 en prueba.

## Módulos

| # | Módulo | SIA | Fechas (2026) | Estado | Carpeta |
|---|--------|-----|---------------|--------|---------|
| 1 | Procesamiento y entendimiento de lenguaje natural | 2029448 | 12 ago – 29 sep (el folleto dice 19 sep) | Terminado · material en carga | [01 - Procesamiento y Entendimiento de Lenguaje Natural](01%20-%20Procesamiento%20y%20Entendimiento%20de%20Lenguaje%20Natural/README.md) |
| 2 | Deep Learning | 2029446 | 23 sep – ~7 nov (el folleto dice 31 oct) | **En curso** | [02 - Deep Learning](02%20-%20Deep%20Learning/README.md) |
| 3 | Metodologías Ágiles para el Desarrollo de Aplicaciones con Machine Learning | 2029447 | 4 nov – 12 dic | Pendiente | [03 - Metodologias Agiles para Aplicaciones con ML](03%20-%20Metodologias%20Agiles%20para%20Aplicaciones%20con%20ML/README.md) |

## Estructura del repositorio

```
├── 01 - Procesamiento y Entendimiento de Lenguaje Natural/
│   ├── 00 - Presentacion del Curso/
│   ├── 01 - Introduccion a NLP/              (Tarea 1)
│   ├── 02 - Preprocesamiento/                (material/ + notas.md, Tarea 2)
│   ├── 03 - Embeddings/                      (Tarea 3)
│   ├── 04 - Analisis Supervisado en Textos/  (Tarea 4)
│   ├── 05 - Analisis No Supervisado/
│   └── 06 - Proyecto Aplicado/               (fases 1–4 y entrega final)
├── 02 - Deep Learning/
│   ├── 00 - Presentacion del Curso/
│   ├── 01 - Intro TensorFlow PyTorch Keras/   (material/ + notas.md)
│   ├── 02 - Redes Neuronales Keras/           (entregables/Tarea 2)
│   ├── 03 - Redes Convolucionales/            (material/ + notas.md)
│   ├── 04 - Modelos de Lenguaje y Transformers/
│   ├── 05 - Modelos Generativos y Multimodalidad/
│   └── 06 - Proyecto Aplicado/                (Proyecto/Fase N)
└── 03 - Metodologias Agiles para Aplicaciones con ML/
```

Convenciones de cada módulo:
- `material/`: PDFs de clase.
- `notas.md`: apuntes de la unidad.
- `entregables/Tarea N/`: cada tarea separada en `plantilla/` (notebook sin resolver y `enunciado.md`) y `resuelto/` (notebook resuelto y `paso_a_paso.md`).

## Evaluación (según el folleto)

Cada módulo puede definir su propia distribución. Deep Learning, por ejemplo, usa Talleres 50 %, Proyecto 30 %, Participación 15 % y Autoevaluación 5 % (ver su [README](02%20-%20Deep%20Learning/README.md)).

| Componente | Peso |
|------------|------|
| Actividades 1 a 5 | 10 % c/u (50 %) |
| Proyecto aplicado | 30 % |
| Participación | 20 % |

- Una actividad cuenta como cumplida si se entrega y obtiene al menos **2.5 / 5**.
- Para aprobar el módulo: nota global **≥ 3.0 / 5** y al menos el **80 %** de las actividades cumplidas.
- El proyecto aplicado se entrega por partes acumulativas a lo largo del módulo; cada entrega se evalúa y se puede mejorar hasta la entrega final.
- Certificado de diplomado: aprobar los 3 módulos en un plazo máximo de 1 año desde el último módulo cursado.

## Logística

- Clases sincrónicas: 6:00 – 8:00 pm. El PDF dice **miércoles** en la tabla de módulos y **martes** en la sección de equipo docente.
- Tutoría en grupo pequeño: sábados en la mañana (1 h 30 min).
- Dedicación por módulo: 21 h sincrónicas + 43 h de trabajo autónomo.
- Herramientas: Google Colab + Google Drive.

## Equipo docente (del PDF)

Felipe Restrepo Calle · Fabio Augusto González Osorio · Jorge Eliécer Camargo · Elizabeth León Guzmán · Oscar Julián Perdomo Charry.
