# Unidad 1 — Introducción a TensorFlow, PyTorch y Keras

**Vista en:** Clase 1 (junto con la U0, presentación del curso).

## Material
- Diapositivas: [U1 - Introduccion al Deep Learning.pdf](material/U1%20-%20Introduccion%20al%20Deep%20Learning.pdf)
- Apuntes: [notas.md](notas.md). Cubren la historia del DL, la "receta" (tamaño, datos, feature learning, algoritmos, HPC, trucos), las aplicaciones y TensorFlow (grafos, tensores, variables, GradientTape, optimización, TensorBoard, GPU, regresión logística en TF y en Keras).

## Objetivos
- Describir los conceptos, herramientas generales y campos de aplicación del desarrollo de modelos de Deep Learning.
- Entender los componentes específicos necesarios para implementar modelos de Deep Learning en la práctica.
- Usar herramientas básicas de TensorFlow, PyTorch y Keras en Python para diseñar, implementar y evaluar modelos básicos de aprendizaje.

## Posibles erratas en las diapositivas
- `tf.constant(4. tf.float32...)`: falta una coma entre los argumentos.
- El tensor de orden 3 está rotulado con shape (3,2,3), pero el código produce (3,3,2).
- En el `print` del ejemplo de `GradientTape` parece decir `my_func1` en lugar de `my_func` (la imagen no es del todo legible).

## Entregables
_Sin registrar._
