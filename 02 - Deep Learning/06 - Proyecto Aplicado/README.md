# Unidad 6 — Proyecto aplicado (30 % del módulo)

## Objetivo
Ejecutar un proyecto de deep learning usando las técnicas y herramientas del curso para resolver un problema específico.

## Reglas generales (de *Presentación del proyecto aplicado*)
- Grupos de **3 personas**, conformados en la primera semana.
- Hay **entregas semanales a partir de la Unidad 2** (segunda semana del curso), una por fase:
  1. Entendimiento del negocio y carga de datos
  2. Entendimiento y preparación de los datos
  3. Diseño e implementación
  4. Entrenamiento y validación
  5. Evaluación y entrega final
- **Solo la primera entrega recibe retroalimentación.** Las dudas posteriores se resuelven en el foro o en las tutorías.
- Las entregas parciales se pueden modificar para la entrega final.
- El tema es libre, siempre que aplique lo visto en el curso.
- **Cada integrante sube el mismo `.ipynb`** a la plataforma y hace clic en "Entregar". No se aceptan entregas por otros medios.
- **Plataforma: Google Classroom**, curso *Deep Learning 2026-2*, tema "Entregas proyecto aplicado". Cada fase vale 100 puntos y **no admite entregas después de la fecha límite**.
- La rúbrica completa está en la Unidad 6 (*MLDS5_U6_Proyecto_Aplicado_Conjunto_de_Datos.pdf*, aún no está en este repositorio).

## Nuestro proyecto: clasificación visual de residuos

| | |
|---|---|
| Problema | Sugerir la categoría de un residuo aislado a partir de una foto, como apoyo a una estación educativa de separación en un campus |
| Dataset | **TrashNet** (Thung & Yang), `dataset-resized.zip` en Hugging Face con la revisión `94cd17eb…` fijada; 2.527 imágenes de 512×384 en 6 clases: cardboard, glass, metal, paper, plastic, trash |
| Modelos | **A:** CNN pequeña entrenada desde cero · **B:** MobileNetV2 con pesos de ImageNet (transfer learning, con fine tuning opcional; ver [U3](../03%20-%20Redes%20Convolucionales/notas.md)) |
| Métrica principal | F1 macro. Metas: F1 macro ≥ 0,80 y recall ≥ 0,60 por clase en prueba |
| Integrantes | **A** · Juan Carlos Andrade Unigarro (datos, trazabilidad y evaluación) · **B** · Natalia Patricia Remolina Rodriguez (modelos y experimentos). Equipo de 2, aceptado por el docente (las instrucciones piden 3) |

**Hallazgos de la Fase 1**
- Las 2.527 imágenes se leen sin errores y los conteos coinciden con la fuente.
- Hay 3 pares de imágenes idénticas con etiquetas distintas (`glass115/metal91`, `glass176/plastic152`, `glass389/plastic332`). Se apartaron los 6 archivos y quedan **2.521 imágenes utilizables**.
- Se encontraron 28 pares de imágenes muy parecidas. Se agruparon para que cada par quede en la misma partición.
- Partición estratificada por grupos con semilla 42. **En Colab** (scikit-learn 1.6.1) da entrenamiento **1.772**, validación **375** y prueba **374**. Con scikit-learn 1.8 o más reciente sale 1.766 / 378 / 377. **Las fases siguientes deben reutilizar `manifiesto_modelado.csv` de la ejecución en Colab**, no volver a generar la partición.
- La clase `trash` es la más escasa (137 imágenes, 5,4 %). La mayoritaria es `paper` (23,5 %).

## Entregas

| Fase | Peso en el proyecto | Fecha | Estado | Nota | Archivos |
|------|---------------------|-------|--------|------|----------|
| 1 · Entendimiento del negocio y carga de datos | 10 % | Dom 18-oct | ✅ Entregada (15-oct) · retroalimentación pendiente | | [Fase 1-Entrega 1](Proyecto/Fase%201-Entrega%201/) |
| 2 · Entendimiento y preparación de los datos | | **Jue 22-oct (oficial, Classroom)** | Pendiente: faltan las instrucciones y la rubrica 2.png | | |
| 3 · Diseño e implementación | | Mar 27-oct | Pendiente | | |
| 4 · Entrenamiento y validación | | Dom 1-nov | Pendiente | | |
| 5 · Evaluación y entrega final + presentación | | Sáb 7-nov | Pendiente | | |

### Contenido de `Proyecto/Fase 1-Entrega 1/`
- `Proyecto_Fase1_Clasificacion_Residuos.ipynb`: **nuestro notebook de entrega**.
- `M5U2_1_business_data_load.ipynb`: plantilla guía del curso, de uso opcional.
- `Instrucciones de Proyecto aplicado - Fase 1.docx`: enunciado y rúbrica.
- `Presentación del proyecto aplicado.docx`: reglas generales del proyecto.

### Rúbrica de la Fase 1 (10 %)

Imagen completa: [Rubrica Fase 1.png](Proyecto/Fase%201-Entrega%201/Rubrica%20Fase%201.png)

| Criterio | [4.0 – 5.0] |
|----------|-------------|
| Trasfondo del problema o negocio | Descrito de forma precisa |
| Objetivo del proyecto | Coherente y ordenado |
| Planeación | Muy buena |
| Código de carga u obtención de datos | Ejecuta, es eficiente y entendible |

### Cierre de la Fase 1
- [x] Integrantes: el equipo es de 2 personas y los roles A y B ya están asignados (15-oct).
- [x] Planeación alineada con las 5 fases oficiales, con responsable y horas por fase (15-oct).
- [x] Fechas de las 5 fases: es la propuesta del equipo para cerrar hacia el 7 de noviembre, con la presentación esa semana (15-oct). Hay que ajustarlas si la plataforma indica otras.
- [x] Ejecutado en Colab el 15-oct: 12 celdas numeradas, sin errores y los 10 controles en OK. Esa versión es la que está en la carpeta, con la fecha de las referencias y la frase del horizonte corregidas (solo celdas de texto).
- [x] Entregada en Classroom (15-oct).
- [ ] Guardar `residuos_fase1/reportes/manifiesto_modelado.csv` de la sesión de Colab en Drive. Ahí está la partición oficial y Colab borra los archivos al cerrar la sesión.
- [x] El docente aceptó el equipo de 2 personas (16-oct).
- [x] Referencia 10, formato de miles (2.527), título de la gráfica y metadato `validation` corregidos (15-oct).

**Verificación (15-oct-2026):** el notebook editado se ejecutó completo desde cero, sin errores. Reproduce los mismos resultados (6 conflictos, 28 pares similares, 2.521 imágenes utilizables, partición 1.766 / 378 / 377) con Python 3.14, pandas 3.0 y scikit-learn 1.9.
