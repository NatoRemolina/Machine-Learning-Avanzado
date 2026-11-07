# Tarea 4 · Versión resuelta

| | |
|---|---|
| **Archivo** | [v2_M5U4_Tarea_4.ipynb](v2_M5U4_Tarea_4.ipynb), ejecutado en Colab con GPU T4 el 29 de octubre de 2026 |
| **Entorno** | Python 3.13.16 · NumPy 2.1.3 · Keras 3.13.2 |
| **Resultado** | ✅ Las **12 pruebas en 100** y envío oficial con *«Your submission grade: 100.0%»* |

## Resultado del entrenamiento

| Época | Accuracy | Pérdida | Accuracy de validación | Pérdida de validación |
|---:|---:|---:|---:|---:|
| 1 | 0,3195 | 2,9635 | 0,4920 | 2,5244 |
| 5 | 0,6328 | 1,5320 | 0,6402 | 1,5449 |
| 10 | 0,7341 | 1,0972 | **0,6794** | 1,3299 |

Las cinco primeras épocas de validación coinciden exactamente con la salida esperada del enunciado. Al final, la salida esperada da 0,6808 de accuracy de validación; la diferencia de 0,14 puntos viene de las versiones del entorno.

## Observaciones
- **`seq_preprocess`:** rellena o recorta cada noticia a `max_len - 1` y antepone el `[CLS]` después. Así, en las noticias largas, el recorte por defecto de Keras conserva el final de la noticia y el `[CLS]` nunca se pierde.
- **`get_config()`:** las capas personalizadas de este notebook no lo implementan y aun así las pruebas 4_1 y 4_2 dieron 100. La advertencia que había aquí antes no se confirmó.

`paso_a_paso.md` queda pendiente: se escribirá con el análisis de la tarea.
