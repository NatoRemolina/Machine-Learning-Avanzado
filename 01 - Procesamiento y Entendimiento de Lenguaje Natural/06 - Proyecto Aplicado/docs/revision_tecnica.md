# Revisión técnica

Observaciones de una revisión posterior de los notebooks, ordenadas por impacto. Sirven como guía para una versión mejorada del proyecto y no cambian las entregas ya calificadas.

## 1. Afectan la comparación de modelos

| # | Observación | Dónde | Efecto | Corrección propuesta |
|---|---|---|---|---|
| 1 | El `Word2VecWrapper` crea el modelo con `workers=-1`. Con ese valor Gensim no entrena: avisa en cada época que procesó 0 ejemplos de los esperados. La fase 3 ya advertía este caso en un comentario | Fase 4 y entrega final | Los resultados con Word2Vec, de 0,567, 0,313 y 0,353, vienen de **vectores sin entrenar** | Usar `workers ≥ 1`; con `workers=1` el resultado además es reproducible |
| 2 | El mismo envoltorio **suma** los vectores de palabra en lugar de promediarlos, usa `min_count=5`, cuando la fase 3 usaba 2, y llama `max_features` a lo que en realidad es `vector_size` | Fase 4 y entrega final | Representación distinta a la justificada en la fase 3 | Promediar y alinear los hiperparámetros con la fase 3 |
| 3 | La SVM tiene `max_iter=50`; scikit-learn lanza `ConvergenceWarning` | Fase 4 y entrega final | La SVM se detiene antes de converger, por eso sus F1 salen bajos, 0,43 y 0,31 | Quitar el límite o usar `LinearSVC` sobre TF-IDF |
| 4 | El bosque aleatorio usa `max_samples=100`: cada árbol ve 100 de 6.056 textos | Fase 4 y entrega final | Subestima al bosque, con 0,45 y 0,36 | Usar el valor por defecto |
| 5 | La partición de la fase 4 **no está estratificada** y la representación se eligió en la fase 3 con validación cruzada dentro del conjunto de ajuste de 1.069 textos, donde las diferencias, entre 0,51 y 0,55, caben en una o dos desviaciones estándar | Fases 3 y 4 | La elección de la representación es poco concluyente | Estratificar y usar validación cruzada anidada sobre entrenamiento |

Aun con estas limitaciones, la conclusión principal se sostiene: **TF-IDF con regresión logística es un modelo sólido para este corpus**. Lo que no está demostrado es que supere a SVM y a Word2Vec bien configurados.

## 2. Datos

- **«ń» en lugar de «ñ»:** está en 313 textos; la fase 2 los marca pero no los corrige. En el texto limpio aparecen lemas como *compańía* o *ańo*, y el token `nş` llega a estar entre los de mayor χ².
- **Fichas empresariales:** algunos textos son plantillas repetidas con campos como teléfono, web, sede social, facturación y número de empleados. El modelo las aprende como *neutral* con mucha confianza, 0,98. Son fáciles de acertar e inflan un poco el desempeño de esa clase.
- **Sesgo por medio:** con un V de Cramér de 0,229 entre medio y etiqueta, el modelo puede aprender el estilo de cada medio. Una validación que deje un medio fuera en cada ronda lo mediría.
- **Dependencia externa:** las fases 3 y 4 cargan el corpus preparado desde el GitHub de un integrante y no desde el archivo exportado.

## 3. Coherencia del texto con los resultados

- **Matrices de confusión:** los títulos de SVM y bosque con Word2Vec están intercambiados. Además, el segundo panel de la comparación final dice "SVM con TF-IDF", pero muestra la regresión logística "simplificada".
- **Modelo "simplificado":** repite exactamente la misma búsqueda que la regresión logística con TF-IDF, así que da el mismo resultado. Su texto habla de una "recodificación de categorías" que no se hizo.
- **Captura de pantalla:** la imagen de la matriz de confusión de la regresión logística no coincide con la salida real del notebook.
- **Conclusión final:** dice que el F1 macro "está entre 0,62 y 0,70", pero esos son los F1 **por clase** de un solo modelo. El F1 macro en prueba es 0,66.
- **Restos de plantillas de otros proyectos:** "reseñas", "calificaciones", "Distribución de las calificaciones".
- **Encuadre como LLM:** el proyecto se presenta como un análisis con grandes modelos de lenguaje, LLM, y con "CRISP-DM adaptada a LLMs", pero todos los modelos son clásicos. Para un portafolio conviene presentarlo como clasificación de sentimiento con NLP clásico.
- **Contradicciones sobre la etiqueta y el corpus:**
  - la etiqueta se trata como **nominal** en la fase 2 y como **ordinal** para las correlaciones de la fase 3;
  - el objetivo habla de un "corpus informal", pero son noticias, que tienen registro formal.

## 4. Forma

- **Texto sobre el propio notebook en lugar del análisis:**
  - tablas "Checklist de la rúbrica" con estado "Cubierto";
  - frases como "esta sección materializa el criterio de la rúbrica";
  - metadatos `completed_for_phase2_rubric`.
- **Texto dirigido al estudiante** copiado de las guías del curso, como «Recuerde» o «Les recordamos».
- **Referencias internas:** algunas apuntan a secciones con otra numeración, y falta la 3.4.4.
- **Erratas:** "Corups", "CRIPS-DM", "exahustivo", "bsoque", "Probabiliadad", "cocimiento", "En está sección" y tildes faltantes.
- **Tercer integrante vacío** en la fase 1.
