# Unidad 3 — Redes neuronales convolucionales

**Vista en:** Clase 3.

## Material
- Diapositivas: [U3 - Redes neuronales convolucionales.pdf](material/U3%20-%20Redes%20neuronales%20convolucionales.pdf)
- Apuntes: [notas.md](notas.md). Cubren el Neocognitron y LeNet-5, la convolución 2D/3D, el pooling, data augmentation, transfer learning, fine tuning y el flujo completo en Keras (TL 0.63 vs. FT 0.73 de accuracy en el ejemplo), con fórmulas de tamaño de salida y conteo de parámetros. Al final hay 8 preguntas de repaso.

## Objetivos
- Entender el procesamiento de una capa convolucional y sus ventajas en análisis de imagen.
- Usar técnicas como Data Augmentation para mejorar modelos de detección de objetos y clasificación de imágenes.
- Entender transferencia de aprendizaje y ajuste fino (fine-tuning) para aplicar modelos preentrenados a tareas específicas de imagen.

## Inconsistencias en las diapositivas
- Diap. 52: la tabla de capas corresponde a EfficientNetB0, aunque el código carga MobileNet.
- Diap. 56: se compila `tl_model` pero se entrena `ft_model`.
- Diap. 56: usa `fit_generator` y `lr=`, APIs antiguas de Keras. Hoy se usan `model.fit` y `learning_rate=`.
- Diap. 34: una frase parece estar en la columna equivocada.

## Entregables
_Sin registrar._
