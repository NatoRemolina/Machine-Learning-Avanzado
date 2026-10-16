# Unidad 3 — Redes neuronales convolucionales · Apuntes (Clase 3)

> Fuente: [U3 - Redes neuronales convolucionales.pdf](material/U3%20-%20Redes%20neuronales%20convolucionales.pdf) (67 diapositivas; archivo original `v2 MLDS5_U3_Redes_neuronales_convolucionales-v2025-1.pdf`). Apuntes generados a partir de las diapositivas.


> Módulo 5 — *Deep learning: Introducción al Aprendizaje Profundo con Python*. Unidad 3.
> Profesor (autor del material): Fabio Augusto González Osorio, PhD. Bienvenida/despedida: Oscar Julián Perdomo Charry, PhD (ojperdomoc@unal.edu.co), Depto. de Ingeniería Eléctrica y Electrónica, Facultad de Ingeniería, UNAL Bogotá. Créditos 2023 (archivo fuente original: `v2 MLDS5_U3_…-v2025-1.pdf`).

**Agenda (diap. 4–5, repetida en 12, 24, 32, 42, 46):**
1. Introducción a Redes Neuronales Convolucionales
2. Capas Convolucionales y de Pooling
3. Data Augmentation
4. Transfer Learning
5. Fine Tuning
6. Transfer Learning y Fine Tuning en Keras

---

## 1. Introducción a las redes neuronales convolucionales (diap. 6–11)

**Motivación — jerarquía de características (diap. 7)**
- Una CNN aprende una cadena: imagen → *característica de nivel bajo* → *nivel medio* → *nivel alto* → *clasificador entrenable* → etiqueta (ej. "Carro").
- La figura muestra los filtros aprendidos en cada nivel: bordes/colores simples (bajo), texturas y formas parciales (medio), patrones complejos parecidos a partes de objetos (alto).

**Inspiración biológica — corteza visual (diap. 8–9)**
- Diagrama de la vía visual: Retina → LGN → V1 → V2 → V3/V3A → V4 → TEO (PIT) → TE (AIT).
  - **Vía ventral**: "vía de reconocimiento de objetos" (V4 → TEO → TE).
  - **Vía dorsal**: "vía de espacio / acción" (MT, MST, VIP, LIP, AIP, 7a…).
- Diap. 9: paralelo entre las áreas visuales (V1, V2, V4, IT) y las capas de una CNN (Convolution → Pooling → Convolution → Pooling → Dense → Output). Enlace "EEG + Imágenes" (video de YouTube).
  - Figura tomada de: https://www.researchgate.net/figure/llustration-of-the-corrispondence-between-the-areas-associated-with-the-primary-visual_fig7_317679065

**Neocognitron — Fukushima (1980) (diap. 10)**
- "Un modelo de red neuronal autoorganizada para un mecanismo de reconocimiento de patrones el cual **no se ve afectado por el cambio de posición**."
- Arquitectura por etapas $U_0$ (retina) → $U_{S1}, U_{C1}, U_{S2}, U_{C2}, \dots$ con *S-cells* y *C-cells* organizadas en *cell-plane stacks* (antecesor de las capas conv + pooling).
- Ref.: Fukushima, K. *Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position.* Biol. Cybernetics 36, 193–202 (1980).

**LeNet-5 — LeCun (1998) (diap. 11)**
- Arquitectura CNN para **reconocimiento de dígitos**.
- Capas (según la figura):

| Capa | Tipo | Salida |
|---|---|---|
| INPUT | — | $32\times32$ |
| C1 | Convolutions | 6 feature maps @ $28\times28$ |
| S2 | Subsampling | 6 @ $14\times14$ |
| C3 | Convolutions | 16 @ $10\times10$ |
| S4 | Subsampling | 16 @ $5\times5$ |
| C5 | capa (Full connection) | 120 |
| F6 | capa (Full connection) | 84 |
| OUTPUT | Gaussian connections | 10 |

- Ref.: LeCun, Y., Bottou, L., Bengio, Y. & Haffner, P. (1998). *Gradient-Based Learning Applied to Document Recognition.* Proceedings of the IEEE, pp. 2278–2324.
- Demo LeNet (visualización 3D interactiva): https://adamharley.com/nn_vis/cnn/3d.html

---

## 2. Capas convolucionales y de pooling (diap. 13–23)

**Capas importantes (diap. 14):** capas convolucionales y capas de pooling (animaciones enlazadas en la diapositiva).

### 2.1 Capa convolucional (diap. 15–16)
- Es el **principal bloque de construcción de una CNN**.
- Contiene un conjunto de **filtros (o kernels)** cuyos **parámetros se aprenden** durante el entrenamiento.
- **Cada filtro "convoluciona" con la imagen y crea un mapa de activación** (feature map).
- Ejemplo numérico (diap. 15): entrada binaria $8\times8$ $*$ kernel $3\times3$ = salida $6\times6$.
  - Kernel: $\begin{bmatrix}1&1&0\\0&1&0\\1&0&1\end{bmatrix}$. Cada salida es la suma de los productos elemento a elemento entre el kernel y la ventana de la entrada (p. ej. la ventana resaltada da 3).

### 2.2 Convolución 2D — parámetros (diap. 17–18)
- `filters`: número de filtros.
- `kernel_size`: tamaño de los filtros.
- `strides`: tamaño del salto de la convolución.
- `padding`: cómo se manejan los bordes de la imagen resultante.
  - `"valid"`: resultado más pequeño.
  - `"same"`: conserva el tamaño.

```python
keras.layers.Conv2D(
                    filters=1,
                    kernel_size=(3,3),
                    strides=(1,1),
                    padding="valid",
)
```

- Diap. 18 (ejemplo): una imagen procesada por 4 filtros distintos produce 4 mapas de activación diferentes (cada filtro resalta rasgos distintos; uno de ellos queda prácticamente negro).

### 2.3 Convolución 3D (diap. 19)
- "Los parámetros de la convolución 3D son los mismos que en una convolución 2D": `filters`, `kernel_size`, `strides`, `padding`.
- Esquema: entrada volumétrica × filtro 3D → salida volumétrica.

```python
keras.layers.Conv3D(
                    filters,
                    kernel_size,
                    strides,
                    padding,
)
```

### Fórmulas útiles (complemento — no aparecen explícitas en las diapositivas, pero se verifican con sus ejemplos)
- **Tamaño de salida** de una convolución/pooling por dimensión, con entrada $N$, kernel $K$, padding $P$ y stride $S$:
  $$O = \left\lfloor \frac{N - K + 2P}{S} \right\rfloor + 1$$
  - `"valid"` ⇒ $P=0$. Ej. diap. 15: $\frac{8-3}{1}+1 = 6$. LeNet C1: $32-5+1=28$ (kernel $5\times5$).
  - `"same"` con $S=1$ ⇒ $O = N$.
  - Pooling $2\times2$, stride 2: $O=N/2$ (LeNet: $28\to14$, $10\to5$; ejemplo de pooling $4\times4\to2\times2$).
- **Parámetros de una capa Conv2D** con $C_{in}$ canales de entrada y $F$ filtros $K_h\times K_w$:
  $$\#\text{params} = (K_h \cdot K_w \cdot C_{in} + 1)\cdot F \quad (\text{el } +1 \text{ es el bias})$$
  - Diap. 52: `top_conv` ($1\times1$, $320\to1280$, sin bias) $=1\cdot1\cdot320\cdot1280 = 409{,}600$.
- **Parámetros de una capa Dense**: $(n_{in}+1)\cdot n_{out}$.
  - Diap. 52: `dense` $=(1280+1)\cdot32 = 40{,}992$; `dense_1` $=(32+1)\cdot1 = 33$.
- **Pooling, Dropout, Activation, GlobalAveragePooling**: 0 parámetros. BatchNormalization: $4\times$ canales (diap. 52: $4\cdot1280=5{,}120$; $4\cdot320=1{,}280$).

### 2.4 Pooling (diap. 20–23)
- **Problema:** la salida de las capas convolucionales registra la **posición precisa** de las características; pequeños movimientos de una característica en la imagen de entrada dan un mapa de características diferente.
- **Solución — Down Sampling:** crear una versión de menor resolución de la señal que sigue conteniendo los elementos estructurales grandes o importantes, sin los detalles finos que pueden no ser útiles para la tarea.
- Propiedades (diap. 21):
  - Se aplica **después de la convolución**, especialmente **después de la función de activación**.
  - Actúa en **cada canal / feature map por separado**.
  - La salida **conserva el número de feature maps**.
- Tipos:
  - **Max Pooling**: toma el valor máximo de la ventana de muestreo.
  - **Average Pooling**: toma el promedio de la ventana.
  - **Sum Pooling**: toma la suma de la ventana.
- Ejemplo (ventana $2\times2$, stride 2) sobre el feature map
  $\begin{bmatrix}6&6&6&6\\4&5&5&4\\2&4&4&2\\2&4&4&2\end{bmatrix}$:
  - Max: $\begin{bmatrix}6&6\\4&4\end{bmatrix}$ · Average: $\begin{bmatrix}5.25&5.25\\3&3\end{bmatrix}$ · Sum: $\begin{bmatrix}21&21\\12&12\end{bmatrix}$
- Capas Keras (diap. 22): `keras.layers.MaxPooling2D`, `AveragePooling2D`, `MaxPooling3D`, `AveragePooling3D`.
- Parámetros: `pool_size` (dimensiones de la ventana de muestreo), `strides` (tamaño del paso de la ventana), `padding` (manejo de los bordes del feature map).

```python
keras.layers.MaxPooling2D(
                          pool_size=(2, 2),
                          strides=(2,2),
                          padding="valid",
)
```

```python
keras.layers.AveragePooling2D(
                          pool_size=(2, 2),
                          strides=(2,2),
                          padding="valid",
)
```

- Diap. 23: ejemplo visual — imagen original vs. Average Pooling (versión pixelada suavizada) vs. Maximum Pooling (resalta valores más claros).
- Recursos: CNN Explainer https://poloclub.github.io/cnn-explainer/ · Video CNN https://www.youtube.com/watch?v=JboZfxUjLSk

---

## 3. Data Augmentation (diap. 25–31)

**Problema / solución (diap. 26)**
- Problema: las CNN **necesitan una gran cantidad de datos** para aprender.
- Solución: extender el conjunto de entrenamiento generando datos (**Data Augmentation**). En imágenes: generar nuevas imágenes transformadas mediante traslaciones, rotaciones, intensidad, contraste, etc.
- Ejemplos con un león: volteado horizontal, volteado vertical, reescalado 10:1, recortado, más brillante, ruido añadido, más oscuro, 60° de rotación.
- Demo Data Augmentation (Albumentations): https://explore.albumentations.ai/

**Ventajas (diap. 27)**
- Ayuda a prevenir **overfitting**.
- Permite entrenar modelos cuando hay **pocos datos**.
- Mejora la **robustez** del modelo.
- Puede mejorar el **desempeño**.

**¡Importante! (diap. 27)**
- Si hay **sesgos** en el dataset original, **persisten** en el dataset aumentado.
- Encontrar los parámetros adecuados de un buen augmentation puede ser **retador**.
- Data Augmentation **solo se aplica al conjunto de entrenamiento**, no a validación ni prueba.

**Herramientas en Keras (diap. 28–29)** — capas de *Image Augmentation* (https://keras.io/api/layers/preprocessing_layers/image_augmentation/):
- *Transformaciones geométricas:* `RandomFlip`, `RandomRotation`, `RandomZoom`, `RandomTranslation`, `RandomShear`, `RandomCrop`, `RandomPerspective`.
- *Ajustes de color y brillo:* `RandomBrightness`, `RandomContrast`, `RandomHue`, `RandomSaturation`, `RandomColorJitter`, `RandomGrayscale`, `RandomInvert`, `Solarization`, `Equalization`.
- *Desenfoque y nitidez:* `RandomGaussianBlur`, `RandomSharpness`.

**Image Data Generator — código (diap. 30–31)**

```python
train_aug = keras.Sequential(
                             [
                                 keras.layers.Input(shape=(None, None, 3)),
                                 keras.layers.Rescaling(scale=1./255),
                                 keras.layers.RandomFlip("horizontal"),
                                 keras.layers.RandomRotation(factor=0.02),
                                 keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2),
                             ], name="train_aug")
```
→ "Esto define un generador de imágenes. Aquí se definen los parámetros del Data Augmentation."

```python
train_generator = keras.utils.image_dataset_from_directory(directory=train_dir,
                                                           image_size=(150, 150),
                                                           batch_size=batch_size,
                                                           labels ='inferred')
```
→ Se definen: el **directorio** de las imágenes, el **tamaño final** de las imágenes, el **tamaño del batch** y **cómo manejar las etiquetas**.

```python
model.fit( train_generator,
           steps_per_epoch=...,
           epochs=...
         )
```
→ "El generador se pasa como argumento al método `fit` del modelo."

---

## 4. Transfer Learning (diap. 33–41)

**Motivación (diap. 34–35)**

| Machine Learning tradicional | Transfer Learning |
|---|---|
| Aprendizaje aislado, de una sola tarea. | El aprendizaje de una nueva tarea se basa en tareas aprendidas anteriormente. |
| El conocimiento no se retiene ni se acumula. | El aprendizaje puede ser más rápido, más preciso y/o necesitar menos datos de entrenamiento. |

- Esquema (diap. 35): en ML tradicional, Dataset 1 → Sistema tarea 1 y Dataset 2 → Sistema tarea 2 de forma independiente; en TL, el sistema de la tarea 1 genera **conocimiento** que se transfiere al sistema de la tarea 2.
- *(Nota: la diap. 34 incluye en la columna "tradicional" la frase "El aprendizaje se realiza teniendo en cuenta los conocimientos adquiridos en otras tareas", que en realidad describe a TL; parece un error de la diapositiva.)*

**¿Para qué se usa? (diap. 36)**
- Ahorrar tiempo y recursos al no entrenar múltiples modelos desde cero para tareas similares.
- Eficiencia en tareas que requieren grandes recursos (categorización de imágenes, PLN).
- Compensar la falta de datos etiquetados en una organización, usando **modelos pre-entrenados**.

**¿Cómo funciona? (diap. 37)**
- Transfiere tantos conocimientos como sea posible de un modelo existente a un nuevo modelo para una tarea similar.
  1. Se escoge un **modelo pre-entrenado** en una tarea similar.
  2. Se conserva la parte que aprende la representación en un espacio latente: **el extractor de características** (base convolucional pre-entrenada).
  3. La parte final, por lo general un **MLP**, **se entrena**.
- Esquema: *Modelo pre-entrenado*: Entrada → Base convolucional pre-entrenada → Clasificador (MLP) pre-entrenado → Salida. *Transfer Learning (extracción de características)*: Entrada → Base convolucional pre-entrenada → *features* → Clasificador (MLP) **se entrena** → Salida.

**Modelos pre-entrenados en Keras (diap. 38, 41)** (entrenados en ImageNet):

| Modelo | Tamaño (MB) | Top-1 | Top-5 | Parámetros | Profundidad | ms/paso CPU | ms/paso GPU |
|---|---|---|---|---|---|---|---|
| Xception | 88 | 79.0% | 94.5% | 22.9M | 81 | 109.4 | 8.1 |
| VGG16 | 528 | 71.3% | 90.1% | 138.4M | 16 | 69.5 | 4.2 |
| VGG19 | 549 | 71.3% | 90.0% | 143.7M | 19 | 84.8 | 4.4 |
| ResNet50 | 98 | 74.9% | 92.1% | 25.6M | 107 | 58.2 | 4.6 |
| ResNet50V2 | 98 | 76.0% | 93.0% | 25.6M | 103 | 45.6 | 4.4 |
| ResNet101 | 171 | 76.4% | 92.8% | 44.7M | 209 | 89.6 | 5.2 |
| ResNet101V2 | 171 | 77.2% | 93.8% | 44.7M | 205 | 72.7 | 5.4 |
| ResNet152 | 232 | 76.6% | 93.1% | 60.4M | 311 | 127.4 | 6.5 |
| ResNet152V2 | 232 | 78.0% | 94.2% | 60.4M | 307 | 107.5 | 6.6 |
| InceptionV3 | 92 | 77.9% | 93.7% | 23.9M | 189 | 42.2 | 6.9 |
| InceptionResNetV2 | 215 | 80.3% | 95.3% | 55.9M | 449 | 130.2 | 10.0 |

- Diap. 39: ejemplo de dataset de 10 clases (avión, automóvil, pájaro, gato, ciervo, perro, rana, caballo, barco, camión) — es **CIFAR-10** (créditos: Alex Krizhevsky, https://www.cs.toronto.edu/~kriz/cifar.html).
- Diap. 41: gráfica de burbujas *Top-1 accuracy (ImageNet-1k) vs. operaciones (G-FLOPs)*, tamaño de burbuja = nº de parámetros (fuente MDPI). Muestra que más FLOPs/parámetros no siempre implica mayor precisión (p. ej. VGG muy costosas con accuracy moderada).

**ImageNet / ILSVRC (diap. 40)**
- ImageNet tiene una competencia anual: **The ImageNet Large Scale Visual Recognition Challenge (ILSVRC)**, donde métodos compiten para **clasificar y detectar objetos**.
- Desde **2012** el Deep Learning gana atención cuando **AlexNet** gana la competencia.
- En **2015** **ResNet** demostró un error menor que el de un ser humano.
- Gráfica del error de clasificación por año:

| Año | Modelo | Error | Tipo |
|---|---|---|---|
| 2010 | — | 28% | Poco profundo |
| 2011 | — | 26% | Poco profundo |
| 2012 | AlexNet (8 capas) | 16% | Profundo |
| 2013 | ZF (8 capas) | 12% | Profundo |
| 2014 | VGG (19 capas) | 7.3% | Profundo |
| 2014 | GoogLeNet (22 capas) | 6.7% | Profundo |
| 2015 | ResNet (152 capas) | 3.6% | Profundo (bajo la línea de "error humano") |
| 2016 | CUImage | 3.0% | Profundo |

- Demo MediaPipe (Google), clasificador de imágenes: https://mediapipe-studio.webapps.google.com/demo/image_classifier?hl=es-419

---

## 5. Fine Tuning (diap. 43–45)

**¿Cuándo? (diap. 44)**
- "Algunas veces la tarea requiere un poco más de esfuerzo", p. ej. cuando el dataset es **muy diferente a ImageNet** (ejemplo: imágenes de resonancia magnética cerebral).
- Tres esquemas comparados:
  - *Modelo pre-entrenado*: base conv. pre-entrenada + MLP pre-entrenado.
  - *Transfer Learning (extracción de características)*: base conv. pre-entrenada (congelada) + MLP **se entrena**.
  - *Fine Tuning o ajuste fino*: base convolucional **(entrenar)** + MLP **(entrenar)**.

**¿Cómo funciona? (diap. 45)**
1. Se escoge un modelo pre-entrenado.
2. **Warming Up (calentamiento):**
   - Se **congela** el bloque convolucional.
   - La parte final (por lo general un MLP) se entrena durante **pocas epochs**.
3. Luego se **liberan los pesos de todo el modelo** y se entrena con un **learning rate muy pequeño**:
   - Se busca un ajuste fino, muy sutil.
   - **Un learning rate grande puede dañar todos los pesos.**
- Ejemplo de la figura: CNN pre-entrenada en ImageNet (salida 1000) → transfer learning → ajuste fino con imágenes de frotis sanguíneo (salida 2).
  - Ref.: Vogado, L., Veras, R., Aires, K., Araújo, F., Silva, R., Ponti, M., & Tavares, J. M. R. S. (2021). *Diagnosis of Leukaemia in Blood Slides Based on a Fine-Tuned and Highly Generalisable Deep Learning Model.* Sensors, 21(9), 2989. https://doi.org/10.3390/s21092989
- Demo: Teachable Machine (Google) https://teachablemachine.withgoogle.com/train/image

---

## 6. Transfer Learning y Fine Tuning en Keras (diap. 47–60)

### 6.1 Usar un modelo pre-entrenado directamente (diap. 48)

```python
model =
keras.applications.MobileNet(weights='imagenet',
                             include_top=True,
                             input_shape=(224, 224, 3))
```
- `weights='imagenet'`: se cargan los pesos del entrenamiento con ImageNet.
- `include_top=True`: se **incluye** el clasificador final de **1000 clases** de ImageNet.
- `input_shape`: tamaño de las imágenes de entrada.
- Se puede usar directamente para predecir (ejemplo: foto de un koala → probabilidad ≈0.9 para *koala*; otras clases: indri, madagascar_cat, wombat, …):

```python
model.predict(...)
```

### 6.2 Problema de ejemplo (diap. 49)
- Clasificación **binaria** (dos clases, `label 0` / `label 1`) sobre imágenes de rostros.

### 6.3 Transfer Learning (diap. 50–54)

```python
extractor =
keras.applications.MobileNet(weights='imagenet',
                             include_top=False,
                             input_shape=(224, 224, 3))
```
- Pesos de ImageNet; **no** se incluye el clasificador de 1000 clases (`include_top=False`); se define el tamaño de entrada.

```python
for layer in extractor.layers:
    layer.trainable=False
```
- El extractor **no se va a entrenar; se congela**.

```python
pool = keras.layers.GlobalAveragePooling2D()(extractor.output)
dense1 = keras.layers.Dense(units=32, activation="relu")(pool)
drop1= keras.layers.Dropout(0.2)(dense1)
dense2 = keras.layers.Dense(units=1, activation="sigmoid")(drop1)

tl_model = keras.models.Model(inputs=[extractor.input], outputs=[dense2])
```
- Se agregan **capas densas nuevas** (un **MLP**): la entrada son las *features* de la CNN, la salida es de **dos clases** (1 neurona sigmoide); estas capas **no están entrenadas**.

**Compilación (diap. 51)** — "El modelo se compila como siempre":
```python
tl_model.compile(loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(lr=1e-3),
    metrics=["accuracy"])
```

**Resumen del modelo (diap. 52)** — parte final del `summary()`:

| Capa | Salida | Parámetros |
|---|---|---|
| block7a_project_bn (BatchNormalization) | (None, 7, 7, 320) | 1,280 |
| top_conv (Conv2D) | (None, 7, 7, 1280) | 409,600 |
| top_bn (BatchNormalization) | (None, 7, 7, 1280) | 5,120 |
| top_activation (Activation) | (None, 7, 7, 1280) | 0 |
| global_average_pooling (GlobalAveragePooling2D) | (None, 1280) | 0 |
| dense (Dense) | (None, 32) | 40,992 |
| dropout (Dropout) | (None, 32) | 0 |
| dense_1 (Dense) | (None, 1) | 33 |

- **Total params:** 4,090,596 (15.60 MB) · **Trainable params:** 41,025 (160.25 KB) · **Non-trainable params:** 4,049,571 (15.45 MB).
- Hasta `global_average_pooling` = **extractor de características** (pesos no se entrenan); desde `dense` = **MLP nuevo** (sí se entrena). "No son muchos los parámetros para entrenar."
- *(Nota: los nombres `block7a_…`/`top_conv` y el total no entrenable corresponden a EfficientNetB0, no a MobileNet como dice el código; la tabla y el código de la diapositiva no coinciden exactamente.)*
- Herramientas para visualizar arquitecturas ("Visualize CNN"): https://github.com/ashishpatel26/Tools-to-Design-or-Visualize-Architecture-of-Neural-Network

**Entrenamiento (diap. 53)** — "El entrenamiento y la evaluación se hacen como siempre":
```python
tl_model.fit(x=X_train,
             validation_data=...,
             epochs=...,
             callbacks=...)
```

**Resultados Transfer Learning (diap. 54)** — conjunto de 613 imágenes:

| Clase | precision | recall | f1-score | support |
|---|---|---|---|---|
| 0 | 0.62 | 0.54 | 0.58 | 284 |
| 1 | 0.64 | 0.71 | 0.68 | 329 |
| accuracy | | | 0.63 | 613 |
| macro avg | 0.63 | 0.63 | 0.63 | 613 |
| weighted avg | 0.63 | 0.63 | 0.63 | 613 |

### 6.4 Fine Tuning (diap. 55–59)
- "Después de transfer learning, podemos hacer fine tuning."
- Mismo código de definición que en 6.3 (MobileNet `include_top=False`, congelar extractor, MLP nuevo con `GlobalAveragePooling2D` → `Dense(32, relu)` → `Dropout(0.2)` → `Dense(1, sigmoid)`), pero ahora **el extractor se congela para el Warming Up**.

**Warming up y liberación de capas (diap. 56)**
```python
tl_model.compile(loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(lr=1e-3),
    metrics=["accuracy"])
```
```python
ft_model.fit_generator(X_train,
                       validation_data=...,
                       epochs=2)
```
```python
for layer in ft_model.layers:
    layer.trainable = True
```
- El modelo se compila como siempre.
- Las capas añadidas se entrenan **unas pocas epochs** antes de entrenar todo el modelo.
- ⚠️ Si se entrena todo sin ajustar primero las capas nuevas, se **propagan errores hacia atrás** que **dañan el modelo pre-entrenado** con ImageNet.
- Luego se **liberan las capas del extractor** de características: pueden liberarse **todas o un subconjunto**.
- *(Nota: en la diapositiva se compila `tl_model` pero se entrena/libera `ft_model`; además `fit_generator` y el argumento `lr=` son APIs antiguas de Keras — en versiones actuales se usan `fit` y `learning_rate=`.)*

**Re-compilación con learning rate menor (diap. 57)**
```python
tl_model.compile(loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(lr=1e-4),
    metrics=["accuracy"])
```
- "Se compila de nuevo con un learning rate más pequeño" ($10^{-3} \to 10^{-4}$).

**Entrenamiento (diap. 58)**
```python
tl_model.fit(x=X_train,
             validation_data=...,
             epochs=...,
             callbacks=...)
```

**Resultados Fine Tuning (diap. 59)**

| Clase | precision | recall | f1-score | support |
|---|---|---|---|---|
| 0 | 0.73 | 0.67 | 0.70 | 284 |
| 1 | 0.73 | 0.78 | 0.76 | 329 |
| accuracy | | | 0.73 | 613 |
| macro avg | 0.73 | 0.73 | 0.73 | 613 |
| weighted avg | 0.73 | 0.73 | 0.73 | 613 |

### 6.5 Comparación (diap. 60)
- **Transfer Learning:** accuracy **0.63** (f1 clase 0 = 0.58, clase 1 = 0.68).
- **Fine Tuning:** accuracy **0.73** (f1 clase 0 = 0.70, clase 1 = 0.76).
- El ajuste fino mejora ~10 puntos de accuracy y todas las métricas por clase.

---

## Resumen del flujo TL → FT en Keras
1. Cargar backbone con `weights='imagenet'`, `include_top=False`.
2. Congelar (`layer.trainable=False`).
3. Añadir cabeza MLP (GAP → Dense → Dropout → Dense salida).
4. Compilar (Adam, lr $10^{-3}$) y entrenar pocas epochs (*warming up* / transfer learning).
5. Descongelar todo o un subconjunto (`layer.trainable=True`).
6. Re-compilar con lr pequeño ($10^{-4}$) y entrenar (fine tuning).
7. Evaluar (classification report).

---

## Referencias y enlaces citados (diap. 62–66 y a lo largo de la presentación)

**Papers**
- Fukushima, K. (1980). *Neocognitron…* Biol. Cybernetics 36, 193–202.
- LeCun, Y., Bottou, L., Bengio, Y. & Haffner, P. (1998). *Gradient-Based Learning Applied to Document Recognition.* Proc. IEEE, 2278–2324.
- Howard, A.G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M., & Adam, H. (2017). *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.* arXiv:1704.04861.
- Vogado, L. et al. (2021). *Diagnosis of Leukaemia in Blood Slides Based on a Fine-Tuned and Highly Generalisable Deep Learning Model.* Sensors 21(9), 2989. https://doi.org/10.3390/s21092989

**Artículos / tutoriales**
- Wikipedia — Kernel (image processing): https://en.wikipedia.org/wiki/Kernel_(image_processing)
- Brownlee, J. (2019). *How to Visualize Filters and Feature Maps in CNNs.* https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/
- TensorFlow — Image classification: https://www.tensorflow.org/tutorials/images/classification
- Kim, J. (2021). *Take a look inside the CNN.* https://medium.com/@Joocheol_Kim/take-a-look-inside-the-cnn-eb65d0dfdf94
- HackerEarth — Transfer Learning Introduction: https://www.hackerearth.com/practice/machine-learning/transfer-learning/transfer-learning-intro/tutorial/
- Von Csefalvay, C. (2021). *Transfer learning: the dos and don'ts.* https://medium.com/starschema-blog/transfer-learning-the-dos-and-donts-165729d66625
- Sharma, H. (2019). *Transfer Learning: Approaches and Empirical Observations.* https://hackernoon.com/transfer-learning-approaches-and-empirical-observations-efeff9dfeca6
- Sarkar, D. (2022). *A Comprehensive Hands-on Guide to Transfer Learning…* https://towardsdatascience.com/a-comprehensive-hands-on-guide-to-transfer-learning-with-real-world-applications-in-deep-learning-212bf3b2f27a
- Fine-tuning guidelines (imagen, AWS re:Invent 2016).
- Martins, A. & Niculae, V. (2019). *Lecture 4: Representation Learning and Convolutional Networks.* https://andre-martins.github.io/docs/dsl2019/lecture_04.pdf
- Karpathy — *What a Deep Neural Network thinks about your #selfie* (imagen).
- Keras API — Image augmentation layers: https://keras.io/api/layers/preprocessing_layers/image_augmentation/

**Demos y recursos interactivos**
- EEG + Imágenes (video): https://www.youtube.com/watch?v=9B4TlCm-UjY&t=91s
- Demo LeNet 3D: https://adamharley.com/nn_vis/cnn/3d.html
- CNN Explainer: https://poloclub.github.io/cnn-explainer/
- Video CNN: https://www.youtube.com/watch?v=JboZfxUjLSk
- Albumentations explorer: https://explore.albumentations.ai/
- MediaPipe Studio (clasificador de imágenes): https://mediapipe-studio.webapps.google.com/demo/image_classifier?hl=es-419
- Teachable Machine: https://teachablemachine.withgoogle.com/train/image
- Herramientas para visualizar arquitecturas: https://github.com/ashishpatel26/Tools-to-Design-or-Visualize-Architecture-of-Neural-Network

---

## Menciones de entregables
- La presentación **no menciona actividades evaluables, talleres, entregas ni fechas**. Solo incluye demos interactivas opcionales (LeNet 3D, CNN Explainer, Albumentations, MediaPipe, Teachable Machine) y un ejemplo práctico en Keras (clasificación binaria de rostros con MobileNet: transfer learning y fine tuning).

---

## Preguntas de repaso
1. ¿Qué aprende una capa convolucional y qué produce cada filtro al convolucionar con la imagen?
2. Una entrada $8\times8$ se convoluciona con un kernel $3\times3$, stride 1 y `padding="valid"`. ¿Cuál es el tamaño de salida? ¿Y con `padding="same"`?
3. ¿Qué problema resuelve el pooling y en qué se diferencian Max, Average y Sum Pooling? Calcula los tres sobre la ventana $\begin{bmatrix}6&6\\4&5\end{bmatrix}$.
4. ¿Por qué Data Augmentation se aplica solo al conjunto de entrenamiento y qué limitación tiene respecto a los sesgos del dataset?
5. En transfer learning, ¿qué parte del modelo pre-entrenado se conserva/congela y qué parte se entrena? ¿Qué significa `include_top=False`?
6. ¿Qué hito marcó AlexNet en ILSVRC 2012 y qué logró ResNet en 2015?
7. Describe las fases de *warming up* y *fine tuning*. ¿Por qué no conviene entrenar todo el modelo desde el inicio ni usar un learning rate grande?
8. En el ejemplo de la diap. 52, ¿por qué solo hay 41,025 parámetros entrenables y de dónde salen los 40,992 de la capa `dense`?
