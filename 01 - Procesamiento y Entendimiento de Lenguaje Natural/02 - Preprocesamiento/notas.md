# Unidad 2 — Preprocesamiento · Apuntes

> Fuente: [U2 - Preprocesamiento.pdf](material/U2%20-%20Preprocesamiento.pdf) (32 diapositivas).

Módulo 4 "Procesamiento y Entendimiento del Lenguaje Natural", Unidad 2 "Preprocesamiento", clase sincrónica. Profesor: Felipe Restrepo Calle, PhD (https://ferestrepoca.github.io/). Año en créditos: 2026.

---

## 0. Ubicación en el módulo y agenda (diap. 1–4)

- **Mapa de contenidos (diap. 3):** la Unidad 2 cubre tres bloques:
  - Preprocesamiento de texto
  - Expresiones regulares
  - Introducción a spaCy
- Resto del módulo, como contexto: U1 Introducción al NLP · U3 Embeddings · U4 Análisis supervisado en textos · U5 Análisis no supervisado en textos · U6 Proyecto aplicado.
- **Agenda (diap. 4):**
  1. Ciclo de vida de proyectos de NLP
  2. Preparación del *corpus*: 2.1 Tokenización · 2.2 Filtrado de palabras · 2.3 Lematización · 2.4 Normalización de textos · 2.5 Modificación de grafía · 2.6 Expresiones regulares
  3. Herramientas para preprocesamiento

---

## 1. Ciclo de vida de proyectos de NLP (diap. 5–11 y 22)

- El desarrollo de aplicaciones de NLP sigue el enfoque **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*).
- Figura (diap. 5): ciclo circular con el *corpus* al centro; Entendimiento del negocio ⇄ Entendimiento del *corpus* → Preparación del *corpus* ⇄ Modelamiento → Evaluación → (vuelve a negocio) y → Despliegue.

| Etapa | Ideas clave | Elementos asociados (recuadro de la figura) |
|---|---|---|
| **Entendimiento del negocio** (diap. 6) | Comprensión de objetivos y requisitos; identificar necesidades empresariales del proyecto NLP; garantizar éxito y cumplimiento de expectativas | Trasfondo del negocio, alcance, plan |
| **Entendimiento del *corpus*** (diap. 7) | Conocer calidad, cantidad y relevancia de los datos; algoritmos precisos requieren datos adecuados; identificar y resolver problemas de integridad de datos | Carga de textos, carga y adquisición de datos, web scrapping, OCR, EDA |
| **Preparación del *corpus*** (diap. 8) | Limpieza, normalización y transformación de los datos para usarlos en modelos; paso clave para precisión y eficiencia; incluye normalización, tokenización, eliminación de palabras, etc. | Extracción de características, tokenización, normalización de textos, expresiones regulares, lematización, filtrado de palabras, modificaciones en la grafía |
| **Modelamiento** (diap. 9) | Selección y entrenamiento de modelos para tareas específicas a partir de datos preprocesados | Clasificación de textos, agrupamiento de textos, modelos de tópicos, modelos generativos, resumen automático, otros |
| **Evaluación** (diap. 10) | Evaluar rendimiento en tareas específicas; comparar con datos de referencia usando métricas apropiadas; mejora continua | Accuracy, Precision, Recall, F1 |
| **Despliegue** (diap. 11) | Integración y puesta en producción; integración con sistemas existentes, pruebas en producción, monitoreo continuo; disponibilidad y escalabilidad | Aplicaciones web, APIs, CLIs, dashboards, otros |

**Tareas de modelamiento (diap. 9):**

| Supervisadas | No supervisadas |
|---|---|
| Clasificación de tokens, clasificación de textos, traducción automática, llenado de máscaras, resumen abstractivo, *question answering*, … | Similitud semántica, *zero-shot classification*, generación de texto, agrupamiento de textos, modelado de tópicos, … |

**Etapa actual (diap. 22):** el preprocesamiento de textos es importante porque da una **versión estandarizada de los datos** y posibilita el entrenamiento de modelos. Hace parte de la etapa de **preparación del *corpus***.

---

## 2. Preparación del *corpus* (diap. 12–21)

### 2.0 Flujo general de preprocesamiento (diap. 13–14)

- Objetivo: dejar los textos en un **formato más simple**.
- Se aplican técnicas de **lingüística computacional** y modelos de NLP para la limpieza de datos.
- Figura (diap. 13): Texto original → Tokenización → Filtrado de palabras → Lematización → Normalización → Modificación de la grafía → Limpieza → Texto procesado.

**Ejemplo paso a paso (diap. 14):**

| Paso | Acción | Resultado mostrado |
|---|---|---|
| Texto original | — | "El BIG data ayuda a La recopilación de grandes cantidades de dATOs y la búsqueda de TENDenciAS dentro de los MISmos." |
| Filtrado de palabras | Eliminar "de", "la", "a", "del", "y", "e", "las", "los", "el" | "BIG data ayuda recopilación grandes cantidades dATOs búsqueda TENDenciAS dentro MISmos." |
| Categorización POS | Extraer todos los verbos y sujetos del texto | "BIG data ayuda recopilación cantidades dATOs búsqueda TENDenciAS MISmos." |
| Normalización | Eliminar los acentos | "BIG data ayuda recopilacion cantidades dATOs busqueda TENDenciAS MISmos." |
| Modificación de la grafía | Conversión a minúsculas | "big data recopilacion grandes cantidades datos busqueda tendencias mismos" |

- Nota: en la diapositiva la entrada del último paso ("BIG data recopilacion grandes cantidades dATOs busqueda TENDenciAS MISmos") no coincide exactamente con la salida del paso anterior (reaparece "grandes" y desaparece "ayuda"); se transcribe tal como aparece.

### 2.1 Tokenización (diap. 15)

- **Definición:** proceso de dividir un texto en fragmentos significativos (**tokens**).
- Los tokens pueden ser palabras, símbolos del lenguaje e incluso frases completas, según el objetivo.

| Tipo | Texto de entrada | Tokens |
|---|---|---|
| De carácter | "La RAM almacena los datos de forma temporal." | L, a, R, A, M, a, l, m, a, c, e, n, a, … , l, . |
| De oración | "El Grey Hat es un tipo de hacker. Actúa ilegalmente, aunque con buenas intenciones. Estos prestan sus servicios a agencias de inteligencia" | 3 oraciones: "El Grey Hat es un tipo de hacker" / "Actúa ilegalmente, aunque con buenas intenciones" / "Estos prestan sus servicios a agencias de inteligencia" |
| De palabra | "El procesador es el cerebro de todo el funcionamiento del sistema y se encarga de ejecutar el código de los diferentes programas." | El, procesador, es, el, cerebro, de, todo, el, funcionamiento, del, sistema, y, se, encarga, de, ejecutar, el, código, de, los, diferentes, programas, . |

- Observación: en los tokens de carácter se omiten los espacios; el punto final también es un token.

### 2.2 Filtrado de palabras (diap. 16)

- **Definición:** eliminar o reducir la frecuencia de palabras no relevantes o irrelevantes.
- Mejora eficiencia y precisión al **reducir la dimensionalidad** y eliminar la interferencia de palabras irrelevantes.
- Ejemplo con dos filtros:
  - Texto: "Una tormenta llegó y arrasó con todo el pueblo."
  - **Primer filtro — stopwords:** eliminar "de", "la", "a", "del", "y", "e".
  - Tokens: `["Una", "tormenta", "llegó", "arrasó", "con", "todo", "pueblo", "."]`
  - **Segundo filtro — longitud:** tokens con longitud mayor a 3 y menor a 10 caracteres.
  - Tokens finales: `["tormenta", "llegó", "arrasó", "todo", "pueblo"]`
- Detalle: "el" no está en la lista de stopwords del ejemplo, pero no aparece en los tokens intermedios; así figura en la diapositiva.

### 2.3 Lematización (diap. 17)

- **Definición:** convertir una palabra en su forma base o **lema**.
- Figura: comería (1.ª sing., condicional), comías (2.ª sing., pretérito imperfecto), comemos (1.ª plural, presente), comeré (1.ª sing., futuro simple), comerán (3.ª plural, futuro simple) → **COMER**.
- Es una generalización del *stemming* (obtener prefijos de palabras): considera estructuras gramaticales (tiempos verbales, conjugaciones, etc.), no solo sufijos.
- Mejora eficiencia y precisión: reduce dimensionalidad y elimina redundancia.

### 2.4 Normalización de textos (diap. 18)

- **Definición:** transformar el texto para obtener una **forma canónica**, con el fin de mejorar eficiencia y precisión de los modelos.
- Incluye: eliminación de puntuación, eliminación de acentos y corrección de ortografía.
- Ejemplo:
  - Original: "El perro ladró, la vaca mugió, el gato maulló y el pato graznó y corrió todo lo que pudo. Fue así como transcurrió otro apacible día en la granja. Bueno, quizás, debí decir un día raro, y es que el león, que se había escapado de la selva."
  - Normalizado: "el perro ladro la vaca mugio el gato maullo y el pato grazno y corrio todo lo que pudo fue asi como transcurrio otro apacible dia en la granja bueno quizas debi decir un dia raro y es que el leon que se habia escapado de la selva"
  - En el ejemplo también se pasa a minúsculas.

### 2.5 Modificación de la grafía (diap. 19)

- Estandariza textos al eliminar mayúsculas o minúsculas.
- Según la diapositiva, esto permite eliminar efectos de signos de puntuación y algunos patrones de escritura.

| Transformación | Método de `str` | Resultado |
|---|---|---|
| Minúsculas | `lower()` | "flores blancas" |
| Mayúsculas | `upper()` | "FLORES BLANCAS" |
| Texto capitalizado | `capitalize()` | "Flores blancas" |
| Título | `title()` | "Flores Blancas" |

### 2.6 Expresiones regulares (diap. 20–21)

- **Definición:** lenguaje de patrones para buscar y manipular texto.
- Usos en NLP: extracción de información, validación de formato y limpieza de texto.
- Son bastante generales: definen patrones sobre *strings*.
- Además del preprocesamiento, sirven para lematización, extracción de frases, entre otras.
- Ejemplo (diap. 20) — placas de automóviles: `[A-Z]{3}[0-9]{3}` → AGD125, VAL202, SOV933, …

**Tabla de ejemplos (diap. 21):**

| Expresión | Descripción | Ejemplos |
|---|---|---|
| `[a-zA-Z]+` | Una o más letras mayúsculas y/o minúsculas | a, H, l, m, Hola, hola, mAr, flor, ESTRella, safFlsPS, mCnjUD, … |
| `[a-z]+[0-9]{2}` o `[a-z]+\d{2}` | Una o más minúsculas seguidas de dos números del 0 al 9 | base10, planeta52, escuela63, cancion27, iron14, piano99, sfgrd02, ifkos31, … |
| `A[a-z]*` | Empieza por una A mayúscula seguida de 0 o más minúsculas | Ala, Ardilla, Ambulancia, Arroyo, Anillo, Alma, Ancdop, Ayrgf, A, … |
| `[AEIOU]*[a-z]+[sS]` | Cero o más vocales mayúsculas al inicio y termina en s minúscula o mayúscula | Almohadas, Urbanos, Esmeraldas, Ostras, IlusionesssS, AkodckS, pdclss, akodckS, OmskcSS, … |

**Elementos de sintaxis que aparecen en estos ejemplos:**

| Elemento | Significado |
|---|---|
| `[...]` | Clase de caracteres (uno de los incluidos) |
| `a-z`, `A-Z`, `0-9` | Rangos dentro de la clase |
| `\d` | Dígito (equivale a `[0-9]` en el ejemplo) |
| `+` | Una o más repeticiones |
| `*` | Cero o más repeticiones |
| `{n}` | Exactamente n repeticiones |

- Las diapositivas **no** incluyen tablas completas de metacaracteres, anclas (`^`, `$`), operadores como `|` ni grupos `( )`; solo lo anterior.

---

## 3. Herramientas para preprocesamiento (diap. 23–29)

- Hay distintas librerías en Python; las más comunes: **spaCy, re, unidecode, pandas** (diap. 24; figura con sus logos).

### 3.1 spaCy (diap. 25–26)

- Biblioteca de procesamiento de lenguaje natural para Python.
- Herramientas: tokenización, lematización, identificación de *stopwords*, entre otras.
- Aplica técnicas típicas de NLP por medio de un ***Pipeline***.
- Figura del pipeline (diap. 25): `Text` → [`nlp`: tokenizer → tagger → parser → ner → …] → `Doc`. Fuente: https://course.spacy.io/en/chapter3
- Estructura de objetos (diap. 26): `Doc` contiene oraciones (`Sent`), que contienen `Token`; un `Span` es una secuencia de tokens.
  - Figura con el texto: "spaCy es una librería de software para procesamiento de lenguajes naturales. spaCy proporciona modelos preentrenados de diferentes lenguajes. Además, junto a una sintaxis clara, hace que sea ideal para principiantes en el campo de la NLP." (resalta tokens y spans dentro de cada oración).
- Con las clases **`Doc`** y **`Span`** se extrae información de todo un documento o de una secuencia: entidades nombradas, *part-of-speech*, *tokens*, entre otros.
- Las diapositivas no muestran código de spaCy ni nombres concretos de modelos.

### 3.2 re (diap. 27)

- Módulo de Python para expresiones regulares (*regular expressions*): buscar y manipular texto; útil para extracción de información, validación de formato y limpieza.

Texto de partida:

```python
text = "La Guayana Francesa se ubica en América del Sur pero es territorio de Francia"
```

**a) Buscar todas las palabras que empiecen por mayúscula seguidas de una o más minúsculas (incluyendo acentos) — método `findall()`:**

```python
regex = r"[A-Z][a-záéíóú]+"
compile_regex = re.compile(regex)

results = re.findall(compile_regex, text)
>>> ['La', 'Guayana', 'Francesa', 'América', 'Sur', 'Francia']
```

- La clase de caracteres acentuados se lee con baja resolución; se interpreta como `[a-záéíóú]`.

**b) Sustituir "de" + palabra que empiece por mayúscula seguida de una o más minúsculas por la palabra "francés" — método `sub()`:**

```python
to_replace = "francés"
regex = "de [A-Z][a-z]+"

results = re.sub(regex, to_replace, text)
>>> La Guayana Francesa se ubica en América del Sur pero es territorio francés
```

- Funciones de `re` vistas: `re.compile`, `re.findall`, `re.sub`. No aparecen otras (`search`, `match`, `split`, etc.).

### 3.3 unidecode (diap. 28)

- Módulo de Python para convertir texto con caracteres **Unicode a ASCII**.
- Útil para normalizar y limpiar texto, en especial con varios idiomas y alfabetos.

```python
a = 'Cántame esa canción.'
b = unidecode(a)   # 'Cantame esa cancion.'

c = 'Encontré una solución.'
d = unidecode(c)   # 'Encontre una solucion.'

e = 'El árbol podría caerse.'
f = unidecode(e)   # 'El arbol podria caerse.'
```

- (La diapositiva lo muestra como diagrama; no incluye la línea de `import`.)

### 3.4 pandas (diap. 29)

- Biblioteca de análisis de datos para manejar datos en formato tabular.
- En NLP: cargar, manipular y preparar los datos.
- Ofrece funciones de manipulación de strings (accesor `.str`) útiles para corpus completos.

```python
df.assign(Helado = df.Helado.str.lower())
```

| Helado (antes) | Helado (después) | Color (sin cambios) |
|---|---|---|
| MAraCuYÁ | maracuyá | AmarIllO |
| FRESa | fresa | ROjo |
| Mora | mora | MorAdO |
| LImÓN | limón | VERDE |
| MandaRIna | mandarina | ANaranJaDo |

- Solo cambia la columna `Helado`; `Color` queda igual. `lower()` conserva las tildes.

---

## 4. Referencias (diap. 30)

- Bird, S., Klein, E., Loper, E. (2019). *Natural Language Processing with Python*. Capítulo 3: Processing Raw Text. (Primera edición). O'Reilly Media. https://www.nltk.org/book/
- Clark, A., Fox, C., Lappin, S. (2010). *The Handbook of Computational Linguistics and Natural Language Processing*. (Primera edición). Blackwell Publishing Ltd. http://santini.se/teaching/sais/ClarkEtAl2010_HandbookNLP.pdf
- Linguistic Features. (7 de noviembre de 2016). spaCy. https://spacy.io/usage/linguistic-features
- Rule-based matching. (1 de noviembre de 2016). spaCy. https://spacy.io/usage/rule-based-matching
- Language Processing Pipelines. (1 de noviembre de 2016). spaCy. https://spacy.io/usage/processing-pipelines

Diap. 31: derechos de imágenes (íconos y logos de Wikipedia/Flaticon). Diap. 32: créditos del equipo de producción.

### Temas del programa que no aparecen en las diapositivas

- Carga de datos y estructura típica de un proyecto.
- Identificadores de caracteres, operadores y grupos en regex más allá de lo listado en la sección 2.6.
- Modelos de spaCy y lista de componentes con código.

---

## Preguntas de repaso

1. ¿Qué metodología sigue el ciclo de vida de un proyecto de NLP y cuáles son sus seis etapas?
2. ¿En qué etapa del ciclo se ubica el preprocesamiento y por qué es importante?
3. ¿Qué diferencia hay entre tokenización por carácter, por oración y por palabra?
4. En el ejemplo de filtrado, ¿qué dos filtros se aplican y qué condición impone el de longitud?
5. ¿Por qué la lematización se considera una generalización del *stemming*?
6. ¿Qué acciones incluye la normalización de textos y qué librería convierte Unicode a ASCII?
7. ¿Qué cadenas reconoce `[AEIOU]*[a-z]+[sS]`? ¿Qué significan `*`, `+` y `{2}`?
8. En spaCy, ¿qué componentes forman el pipeline `nlp` y qué relación hay entre `Doc`, `Sent`, `Span` y `Token`?
