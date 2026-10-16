# Unidad 1 — Introducción al Deep Learning · Apuntes (Clase 1)

> Fuente: [U1 - Introduccion al Deep Learning.pdf](material/U1%20-%20Introduccion%20al%20Deep%20Learning.pdf) (77 páginas). Apuntes generados a partir de las diapositivas; los números "diap." corresponden a la página del PDF.


**Unidad 1 — Introducción al Deep Learning.** Profesor Fabio Augusto González Osorio, PhD (diap. 1–3).

## Agenda de la unidad (diap. 4–5, 16, 36)
1. ¿Qué es el aprendizaje profundo?: Generalidades e Historia.
2. Elementos fundamentales del Deep Learning.
3. Herramientas para el Deep Learning.
4. Aplicaciones del Deep Learning.
5. TensorFlow.

(En este rango se cubren 1, 2 y el inicio de 3; Aplicaciones y TensorFlow quedan para las diap. 39+.)

---

## 1. ¿Qué es el aprendizaje profundo? Generalidades e historia (diap. 6–15)

### Portada de sección (diap. 6)
Ilustración de "evolución": de primates a humanos y luego a robots — metáfora de la progresión hacia la IA.

### Perceptrón de Rosenblatt, 1957 (diap. 7)
- Implementación en **hardware** (Mark I Perceptron; fotos de la máquina y de Rosenblatt).
- **Entrada:** matriz de **20 × 20 fotoceldas**.
- **Pesos** implementados con **potenciómetros**.
- **Actualización de pesos** realizada por **motores eléctricos**.
- Video embebido: "Investigación sobre el perceptrón en los años 50 y 60" (duración 1:11; no se ve URL).

### Redes neuronales: línea del tiempo (diap. 8)
| Año | Hito (según figuras) |
|---|---|
| 1943 | Neurona artificial de McCulloch & Pitts (diagrama de neurona con pesos, sumatoria y activación) |
| 1957 | Perceptrón de Rosenblatt (foto) |
| 1969 | Libro *Perceptrons* (Minsky & Papert, edición expandida en la portada) — señaló limitaciones del perceptrón |
| 1980 | Neocognitrón (Fukushima): figura con "retina", capas U_S/U_C, "S-cell / C-cell", "cell-plane stack" — precursor de las CNN |

### McCulloch & Pitts — Neurona artificial (diap. 9)
- Paper: **W. S. McCulloch y W. Pitts, "A Logical Calculus of the Ideas Immanent in Nervous Activity"**, *Bulletin of Mathematical Biophysics*, vol. 5, 1943 (Univ. of Illinois / Univ. of Chicago).
- Modelo de neurona artificial $j$:
  - Entradas $x_1,\dots,x_n$ con pesos $w_{1j},\dots,w_{nj}$.
  - **Función de transferencia** (suma ponderada): $net_j = \sum_{i=1}^{n} w_{ij}\,x_i$
  - **Función de activación** $\varphi$ con **umbral** $\theta_j$: $o_j = \varphi(net_j - \theta_j)$ (en la figura: "Entrada net $net_j$" → $\varphi$ → "Activación $o_j$", umbral $\theta_j$).
- Figuras: diagrama de redes de neuronas lógicas del paper original ↔ (flecha bidireccional) una **máquina de Turing** ilustrada — idea de que las redes de neuronas lógicas tienen poder computacional.

### Backpropagation (diap. 10)
- Figura de **Geoffrey Hinton**.
- Paper: **D. E. Rumelhart, G. E. Hinton & R. J. Williams, "Learning representations by back-propagating errors"**, *Nature* 323, 533–536 (9 oct. 1986), doi:10.1038/323533a0.
- Diagrama de red multicapa (entradas $x_1, x_2$, neuronas $f_i(e)$, salida $y$) con la regla de actualización de pesos de la primera capa:
  $w'_{(x_1)1} = w_{(x_1)1} + \eta\,\delta_1\,\dfrac{df_1(e)}{de}\,x_1$, $\quad w'_{(x_2)1} = w_{(x_2)1} + \eta\,\delta_1\,\dfrac{df_1(e)}{de}\,x_2$
  donde $\eta$ = tasa de aprendizaje, $\delta_1$ = error retropropagado a la neurona 1, $\frac{df_1(e)}{de}$ = derivada de la activación.

### Deep Learning: línea del tiempo (diap. 11)
| Año | Hito (según figuras) |
|---|---|
| 1986 | Backpropagation (mismo diagrama de la diap. 10) |
| 1995 | Captura de **LeNet-5** reconociendo dígitos manuscritos; y libro **V. Vapnik, *The Nature of Statistical Learning Theory*** (época de auge de SVM / teoría estadística del aprendizaje) |
| 2007 (fig. 2006) | Paper **G. E. Hinton, S. Osindero, Y.-W. Teh, "A Fast Learning Algorithm for Deep Belief Nets"**, *Neural Computation* 18, 1527–1554 (2006) |
| 2012 | Grilla de imágenes de ImageNet (resultados de AlexNet) |

### El nacimiento del Deep Learning (diap. 12)
- Geoffrey Hinton + paper de Deep Belief Nets (Hinton, Osindero & Teh, 2006; Univ. of Toronto / National Univ. of Singapore).
- Figura de la arquitectura DBN: imagen de **28 × 28 píxeles** → **500 units** → **500 units** → **2000 top-level units**, con **10 label units** conectadas al nivel superior ("This could be the top level of another sensory pathway").
- Idea clave: entrenamiento rápido capa por capa (preentrenamiento no supervisado) que permitió entrenar redes profundas.

### Modelo de Deep Learning gana el desafío ILSVRC 2012 (diap. 13)
- Ejemplos de clasificación (top-5 predicciones) sobre ImageNet: *lens cap, abacus, slug, hen, tiger, chambered nautilus, tape player, planetarium*.
- **Tasa de error** (barras):

| Equipo | Error |
|---|---|
| **SuperVision** (AlexNet, Krizhevsky–Sutskever–Hinton) | **15.315 %** |
| ISI | 26.172 % |
| OXFORD-VGG | 26.979 % |
| XRCE/INRIA | 27.058 % |
| Univ. of Amsterdam | 29.576 % |

→ Mejora de ~11 puntos sobre el segundo lugar: punto de quiebre.

### Deep Learning: línea del tiempo reciente (diap. 14)
| Año | Hito |
|---|---|
| 2012 | **AlexNet**: empieza el boom del Deep Learning |
| 2014 | Nacimiento de las **GAN** (Ian Goodfellow) |
| 2016 | **AlphaGo** (Google DeepMind) |
| 2017 | Nacimiento de los **Transformers**: *Attention Is All You Need* |
| 2019 | **Turing Award** (fotos de Yann LeCun, Geoffrey Hinton, Yoshua Bengio) |

### Deep Learning en la actualidad (diap. 15)
Collage de titulares:
- "Google hires brains that helped supercharge machine learning" (Hinton y colegas).
- "Facebook taps 'Deep Learning' giant for new AI lab" (Yann LeCun).
- "Google says its speech recognition technology now has only an 8% word error rate".
- Artículo *Nature*: "Mastering the game of Go with deep neural networks and tree search" (AlphaGo).
- "IBM touts SoC as solution for faster deep learning training".
- MIT Technology Review: "Chinese Search Giant Baidu Hires Man Behind the 'Google Brain'".
- "Here's How Deep Learning Will Accelerate Self-Driving Cars" (Danny Shapiro, 24 feb. 2015).
- Science (2 nov. 2022): "Integrating digital twins and deep learning for medical image analysis in era of Covid".
- Foto de la partida AlphaGo (Google DeepMind Challenge Match).

---

## 2. Elementos fundamentales del Deep Learning (diap. 17–35)

### La receta para el Deep Learning (diap. 18)
Diagrama central (cabeza con engranaje) con **6 ingredientes**:
1. **Tamaño** (modelos grandes/profundos)
2. **Datos**
3. **Feature learning**
4. **Trucos**
5. **HPC** (cómputo de alto rendimiento)
6. **Algoritmos**
- Miniatura de "Tamaño": diagrama de capas de AlexNet (CONV/MAX POOLING/LOCAL CONTRAST NORM/FULL CONNECT, ReLU…) — texto pequeño ilegible (parece indicar ~60 millones de parámetros).

### 2.1 Tamaño (diap. 19–21)
**Clasificación de imágenes en ImageNet — "Deep → Bigger", "Revolution of Depth" (diap. 20)**
- Error top-5 (%) en ImageNet por año/modelo y número de capas:

| Competencia / modelo | Capas | Error top-5 |
|---|---|---|
| ILSVRC'10 | shallow | 28.2 |
| ILSVRC'11 | shallow | 25.8 |
| ILSVRC'12 AlexNet | 8 | 16.4 |
| ILSVRC'13 | 8 | 11.7 |
| ILSVRC'14 VGG | 19 | 7.3 |
| ILSVRC'14 GoogleNet | 22 | 6.7 |
| ILSVRC'15 **ResNet** | **152** | **3.57** |

- Idea: más profundidad → menor error.
- Referencia: Abdelbaki, Ahmed (2016). *Computer Vision Lab SS16 – P-CNN features for Action Recognition*. 10.13140/RG.2.2.14789.86247.

**Clasificación en ImageNet en la actualidad (diap. 21)**
- Curva de "máxima precisión" (top-5 de mayor precisión) vs. año (2013–2021 aprox.): AlexNet (~84–85 %) → Five Base + Five HiRes → MSRA → VGG-19 (~92 %) → Inception V3 (~94–95 %) → NASNET-A(6) (~96 %) → FixResNeXt-101 32x48d (~98 %) → ~99 % hacia 2021. Tendencia: saturación cerca del 100 %.

### 2.2 Datos (diap. 22–23)
**ImageNet (diap. 23):** imágenes anotadas con conceptos de **WordNet**.
- Conceptos: **21,841**
- Imágenes: **21,197,122**
- Anotaciones de cuadro delimitador (bounding boxes): **1,034,908**
- Construido mediante ***crowdsourcing***.

### 2.3 Feature learning (diap. 24–25)
- El modelo **aprende las características** (representaciones) jerárquicamente, en lugar de diseñarlas a mano.
- Ejemplo (diap. 25) — conjunto de entrenamiento: imágenes alineadas de rostros:
  **Píxeles → Bordes → Partes del objeto (combinación de los bordes: ojos, narices, bocas) → Modelos de objetos (rostros completos)**.
- Figuras: filtros aprendidos en cada capa (bordes orientados tipo Gabor; partes faciales; caras prototipo).

### 2.4 Algoritmos (diap. 26–27)
- **Backpropagation**
- **Backpropagation through time** (para redes recurrentes)
- **Online learning** (*stochastic gradient descent*)
- **Softmax (hierarchical)**
- Figura: superficie de costo $J(\theta_0,\theta_1)$ en 3D con dos trayectorias de **descenso de gradiente** que, desde puntos iniciales cercanos, terminan en mínimos locales distintos (dependencia de la inicialización). Regla general: $\theta \leftarrow \theta - \eta\,\nabla_\theta J(\theta)$.

### 2.5 HPC (diap. 28–33) — encabezado de las diap. 29–33: "Herramientas para el Deep Learning"
**Unidad de procesamiento gráfico (GPU) (diap. 29)**
- De la computación gráfica al procesamiento numérico general (**GPGPU**).
- Arquitectura de **instrucción única y datos múltiples** (SIMD).
- Cálculos de alto rendimiento con **paralelismo de datos**.
- Hardware básico.
- Dos proveedores principales: **NVidia, AMD**.
- **TPU** (unidades de procesamiento tensorial): circuitos integrados desarrollados por **Google** para acelerar cargas de Machine Learning.

**Evolución del rendimiento de la GPU (diap. 30)**
- Gráfica TFLOPS vs. año (2007–2020): CPUs (Intel Core, Intel Xeon, AMD Ryzen, AMD EPYC) se mantienen mayormente < ~5 TFLOPS (Xeon llega a ~9 en 2019); GPUs (NVIDIA Titan, GeForce, Tesla; AMD Radeon, AMD MI) crecen hasta ~14–16 TFLOPS hacia 2018–2019. La brecha GPU–CPU se amplía.

**NVIDIA Tesla — especificaciones (diap. 31)**

| | Tesla P4 (Ultra-Efficient Scale-Out Servers) | Tesla P40 (Maximum-Inference Throughput Servers) | Tesla P100 (Universal Datacenter GPU) |
|---|---|---|---|
| FP32 | 5.5 TFLOPS | 12 TFLOPS | 10.6 TFLOPS |
| FP16 | — | — | 21 TFLOPS |
| INT8 | 22 TOPS* | 47 TOPS* | — |
| Memoria GPU | 8 GB | 24 GB | 16 GB |
| Ancho de banda | 192 GB/s | 346 GB/s | 732 GB/s |
| Interfaz | Low-Profile PCI Express | Dual-Slot, Full-Height PCIe | Dual-Slot, Full-Height PCIe o SXM2 con NVLink |
| Potencia | 50 W/75 W | 250 W | 250 W (PCIe) / 300 W (SXM2) |
| Video HW | 1 decode, 2 encode | 1 decode, 2 encode | — |

**GPU vs CPU para Deep Learning — NVIDIA Tesla P100 Performance (diap. 32)**
- *Speed-up* de aplicación vs. CPU (servidor dual CPU Intel E5-2698 v3 @ 2.3 GHz, 256 GB RAM; P100 pre-producción; *M40 para Caffe/AlexNet) para 2×K80, 2×/4×/8× P100 en: Caffe/AlexNet (~37× con 2 GPUs), VASP, HOOMD-blue (~32× con 8 P100), COSMO, MILC, AMBER, HACC (~48× con 8 P100). Conclusión: aceleraciones de decenas de veces.

**Programando GPUs (diap. 33)**
- **CUDA**: plataforma de computación paralela de Nvidia; acceso al conjunto de instrucciones virtual de la GPU para ejecutar núcleos (*kernels*) de cómputo en los elementos paralelos; **CUDA C** (versión especializada de C; también CUDA Fortran); librerías especializadas.
- **OpenCL**: similar a CUDA pero multiplataforma, sin depender del proveedor; el camino a seguir con GPUs AMD; "un paso por detrás de CUDA".

### 2.6 Trucos (diap. 34–35)
- Referencia: libro **G. Montavon, G. B. Orr, K.-R. Müller (Eds.), *Neural Networks: Tricks of the Trade*, 2.ª ed., Springer, LNCS 7700**.
- Ideas:
  - El Deep Learning es principalmente un **problema de ingeniería**.
  - Las redes de DL son **difíciles de entrenar**.
  - Varios trucos son producto de años de experiencia.
- Trucos listados: **Layer-wise training**, **ReLU units**, **Dropout**, **Adaptive learning rates**, **Initialization**, **Preprocessing**.

---

## 3. Herramientas para el Deep Learning (diap. 36–38, continúa en diap. 39+)

### Frameworks de Deep Learning (diap. 38)
- **PyTorch** — Facebook (Meta).
- **TensorFlow** — Google → con **Keras** como API de alto nivel (flecha TensorFlow → Keras).
- **JAX** (logo de bloques "JAX", Google).

---

## Referencias citadas (diap. 1–38)
- McCulloch, W. S. & Pitts, W. (1943). A Logical Calculus of the Ideas Immanent in Nervous Activity. *Bull. Math. Biophysics*, 5.
- Rosenblatt — Perceptrón (1957).
- Minsky, M. & Papert, S. *Perceptrons* (1969; edición expandida en la imagen).
- Fukushima — Neocognitrón (1980) (solo figura).
- Rumelhart, D. E., Hinton, G. E. & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature* 323, 533–536. doi:10.1038/323533a0.
- Vapnik, V. *The Nature of Statistical Learning Theory* (1995).
- Hinton, G. E., Osindero, S. & Teh, Y.-W. (2006). A Fast Learning Algorithm for Deep Belief Nets. *Neural Computation* 18, 1527–1554.
- Vaswani et al. (2017). *Attention Is All You Need* (citado por título).
- Silver et al. Mastering the game of Go with deep neural networks and tree search. *Nature* (titular).
- Abdelbaki, A. (2016). Computer Vision Lab SS16 – P-CNN features for Action Recognition. 10.13140/RG.2.2.14789.86247.
- Montavon, Orr & Müller (Eds.). *Neural Networks: Tricks of the Trade*, 2nd ed., Springer LNCS 7700.
- Contacto profesor: https://dis.unal.edu.co/~fgonza/ · fagonzalezo@unal.edu.co

---


> Segunda mitad de "U1 - Introducción al Deep Learning" (Programa de Formación Machine Learning and Data Science MLDS, Facultad de Ingeniería, UNAL). Cubre las secciones 4 (Aplicaciones del Deep Learning) y 5 (TensorFlow) de la agenda, más despedida, referencias y créditos.

## Agenda — segunda parte (diap. 39, 46)

1. ¿Qué es el aprendizaje profundo?: Generalidades e Historia.
2. Elementos fundamentales del Deep Learning.
3. Herramientas para el Deep Learning.
4. **Aplicaciones del Deep Learning** (resaltada en diap. 39).
5. **Tensorflow** (resaltada en diap. 46).

Figura: icono de un libro abierto con una nube y conexiones tipo circuito.

---

## 4. Aplicaciones del Deep Learning (diap. 40–45)

Portada de sección (diap. 40): ilustración de un robot leyendo un libro con un micrófono.

### Visión por ordenador (diap. 41)
- **Imagen**: anotación, detección, segmentación, subtitulado.
- **Vídeo**: seguimiento de objetos, reconocimiento de acciones, segmentación.
- Figura: escena urbana ilustrada (personas corriendo, en bicicleta, caminando) con *bounding boxes* de colores alrededor de las personas detectadas.

### Texto, habla e imágenes biomédicas (diap. 42)
- **Texto**: modelización del lenguaje, representación de palabra/texto, clasificación de textos, traducción.
  - Figura: diagrama **codificador–decodificador**: *Vectores de palabras* → *Codificador* → *Decodificador* → *Traducción*.
- **Reconocimiento y síntesis del habla** (figura: cabeza hablando con onda de sonido y un check).
- **Análisis de imágenes biomédicas** (figura: cortes de tomografía/resonancia cerebral).

### Modelos generativos: DALL-E (diap. 43)
- Una IA generativa como **DALL-E** funciona tomando como referencia una **descripción o una imagen de partida** para crear imágenes y obras de arte.
- Figura: logo de OpenAI y tres imágenes generadas (astronauta a caballo en la Luna, ositos de peluche haciendo química, un personaje frente a un portátil con un gato de noche).

### Large Language Models (LLM) (diap. 44)
- Un **Large Language Model (LLM)** es un tipo de modelo de Machine Learning que abre nuevas posibilidades de comprensión y generación de textos en sistemas de software.
- Casos de uso:
  - **Generate**: generación de textos originales y coherentes.
  - **Summarize**: resúmenes de textos.
  - **Rewrite**: correcciones de textos, parafraseo de textos.
  - **Extract**: extracción de información o palabras clave provenientes de textos largos.
  - **Search/Similarity**: *text matching*.
  - **Classify**: incrustaciones (*embeddings*) de texto, clasificación de intenciones en chatbots, análisis de actividad sentimental en redes sociales.
- Figura: seis tarjetas (Generate, Summarize, Rewrite, Extract, Search, Classify) con esquemas de líneas (fuente: Cohere, "LLM use cases").

### ChatGPT (diap. 45)
- **ChatGPT** es un prototipo de chatbot artificial entrenado para mantener conversaciones con cualquier persona.
- Lo que lo hace distinto de otros chatbots: la posibilidad de **mantener un estado**, es decir, recordar las indicaciones anteriores de la conversación.
- Ha sido tema de conversación por su capacidad de responder de manera completa y acertada múltiples preguntas de millones de usuarios.

---

## 5. TensorFlow: panorama general (diap. 47–48)

Portada (diap. 47): logo de TensorFlow.

Características (diap. 48):
- Computación numérica basada en **grafos de flujo de datos**.
- Desarrollado en **C++**.
- Interfaces en **Python y C++**.
- **Diferenciación automática**.
- Fácil visualización mediante **TensorBoard**.
- **Capas de abstracción**:
  - `tf.contrib.learn`, `tf.contrib.slim`
  - TFLearn, **Keras**
- **Soporte de arquitecturas heterogéneas**: Multi-CPU, GPU, Multi-GPU, distribuida, móvil.
- Figura: logos de C++, Python y TensorBoard.

---

## Grafos computacionales (diap. 49–51)

### Concepto (diap. 49)
- Un **grafo computacional (CG)** define las operaciones que deben realizarse sobre un conjunto de constantes y variables.
- TensorFlow (TF) trabaja sobre CG donde las variables suelen ser **tensores** (escalares, vectores, matrices, matrices multidimensionales).
- En TF **primero se crea el CG y luego puede ejecutarse**.
- El CG puede **manipularse simbólicamente**, por ejemplo, para calcular su gradiente o simplificarlo.
- Figura: grafo de la expresión $ax^2 + bx + c$: nodos hoja $a$, $x$, $b$, $c$; nodos de operación `*` ($a\cdot x$, luego $\cdot x$), `*` ($b\cdot x$), `+` (suma de ambos productos) y `+` final con $c$.

### Creando un grafo en TF (diap. 50)

```python
@tf.function
def my_func(x):
    a = tf.constant(10, tf.float32, name= 'a')
    b = tf.constant(-5, tf.float32, name= 'b')
    c = tf.constant(4. tf.float32, name= 'c')
    y = a * x * x + b * x + c
    return y
```
*(Nota: en la diapositiva la línea de `c` aparece como `4.` seguido del dtype; lo correcto sintácticamente sería `tf.constant(4., tf.float32, name='c')`.)*

- `@tf.function`: se define una función **decorada** para que TensorFlow pueda construir el grafo computacional.
- `tf.constant(...)`: declaración de constantes.
- `y = a * x * x + b * x + c`: definición de la función, es decir $f(x) = 10x^2 - 5x + 4$.

### Grafo obtenido (diap. 51)
- Figura (estilo TensorBoard): nodos `x`, `a`, `b`, `c`, operaciones `mul`, `mul_1`, `mul_2`, `add`, `add_1`, y salidas `Identity` → `identity_...`.

---

## Tensores (diap. 52–54)

### Conceptos (diap. 52)
- **¿Por qué son importantes?** Son un formato de datos que permite realizar algunas operaciones de forma muy eficiente.
- **¿Para TF qué es un tensor?** En TF, un tensor es un **manejador simbólico de una de las salidas de una operación**.
- **¿Qué es un tensor?** En general, un tensor es una **matriz multidimensional**:
  - **Vector**: tensor unidimensional.
  - **Matriz**: tensor bidimensional.

### Ejemplos (diap. 53–54)
Los dos atributos principales de un tensor son su **tipo de dato** (*dtype*) y su **forma** (*shape*).

Tensor de orden 0 → `Shape ()` (escalar `10`):
```python
tensor_de_orden_0 = tf.constant(10)
```

Tensor de orden 1 → `Shape (2,)` (vector `[5, 7]`):
```python
tensor_de_orden_1 = tf.constant([5, 7])
```

Tensor de orden 2 → `Shape (3,2)` (3 filas, 2 columnas):
```python
tensor_de_orden_2 = tf.constant([[4, 13],
                                 [22,15],
                                 [8, 17]], dtype=tf.float16)
```

Tensor de orden 3 → `Shape (3,2,3)` según la diapositiva:
```python
tensor_de_orden_3  = tf.constant([[[4,  13], [22,   15], [8,   17]],
                                  [[3,  22], [5,    14], [9,   12]],
                                  [[8,  11], [2,    85], [7,   42]]])
```
- Figura: tres "capas" de matrices 3×2 apiladas (verde, roja, azul oscuro) con dimensiones marcadas 3, 3 y 2.
- *Observación*: tal como está escrito el código (3 bloques × 3 filas × 2 elementos), la forma real que devolvería TF sería `(3, 3, 2)`; la diapositiva rotula `Shape (3,2,3)`.

---

## Variables (diap. 55–56)

### Definición y operaciones (diap. 55)
- El constructor **`Variable()`** requiere un **valor inicial** para la variable, que puede ser un tensor de cualquier tipo y forma.
- El valor inicial **define el tipo y la forma** de la variable.
- En un tensor variable **pueden cambiar los valores durante la ejecución**.

Se define un 2D-tensor (matriz) variable a partir de una lista:
```python
t = tf.Variable([[1, 2], [3, 4]], dtype=tf.float32)
```
Al tensor variable se le puede asignar un nuevo valor:
```python
t.assign([[-2, -1], [-3, -7]])
```
También se le puede sumar o restar un valor:
```python
t.assign_add([[1, 1], [1, 1]])
```

### Gradientes con `GradientTape` (diap. 56)
Se puede calcular el gradiente de cualquier expresión con respecto a una variable que aparezca en ella, y usar **gradiente descendente** para encontrar el **mínimo de un polinomio**.

```python
x = tf.Variable(5.0)
dy_dx = 1
while np.abs(dy_dx) > 0.01:
    with tf.GradientTape() as t:
        t.watch(x)
        y = my_func(x)
    dy_dx = t.gradient(y, x)
    x.assign(x - 0.01*dy_dx)
    print("f(x)=", my_func1(x).numpy(), "df(x)/dx=", dy_dx.numpy())
```
*(Nota: en el `print` la diapositiva parece leer `my_func1`; probablemente se refiere a `my_func`.)*

- Se define una variable $x$ para que su valor se vaya actualizando en diferentes evaluaciones.
- Para calcular el gradiente de la expresión se usa un objeto **`GradientTape`**.
- Se usa la función definida anteriormente, `my_func(x)`.
- Regla de actualización (tasa de aprendizaje 0.01), iterando hasta que $|f'(x)| \le 0.01$:
  $$x \leftarrow x - 0.01\,\frac{df}{dx}$$

---

## Optimización (diap. 57–61)

### Conceptos (diap. 57)
- TF calcula los **gradientes de un grafo**.
- Puede implementar su **propia estrategia de optimización** usando gradientes.
- Los parámetros a optimizar se declaran como **variables**.
- Una instancia de un optimizador recibe como parámetro **la tasa de aprendizaje**.
- El optimizador debe ser llamado con la **función objetivo**.
- Figura: curva con pasos sucesivos en dirección $-\nabla f$ que descienden (de ~4.0 hasta ~3.4) hasta el mínimo ("Min").

### Ejemplo con `SGD` (diap. 58–59)
Se define el optimizador:
```python
sgd = tf.keras.optimizers.SGD(learning_rate=0.01)
```
Se define `create_f_x`, que crea una variable de TensorFlow $x$ y una función $f_x$, con $f(x) = 10x^2 - 5x + 4$, que dependerá de $x$ y será la función a optimizar:
```python
def create_f_x(init_val):
    x = tf.Variable(init_val)
    def f_x():
        return 10 * x * x - 5 * x + 4
    return x, f_x
```
Siguiendo con el proceso anterior:
```python
x, f_x = create_f_x(5.0)
for i in range(40):
    print('f(x)=', f_x().numpy())
    opt = sgd.minimize(loss=f_x,
                       var_list=[x])
```
- `loss=f_x`: función a optimizar.
- `var_list=[x]`: lista de variables respecto a las cuales optimizar.
- Se realizan **40 iteraciones de gradiente descendente** con la función `minimize`.
- (Referencia: el mínimo analítico de $10x^2-5x+4$ está en $x = 5/20 = 0.25$.)

### Optimización de un CG (diap. 60–61)
- Figura (diap. 60): grafo en TensorBoard con el nodo `x`, la operación `mul` (con `a`), el bloque agrupado `mul[1-2]` (`mul_1`, `mul_2` con `b`), `add`, `add_1` (con `c`), un bloque `gradients` y los nodos auxiliares `GradientDescent` e `init`.
- Figura (diap. 61): el bloque `gradients` expandido, mostrando los nodos de gradiente generados automáticamente: `Fill` (con `Shape`/`Const`), `add_1_grad`, `add_grad`, `mul_1_grad`, `mul_2_grad`, `mul_grad` y un `AddN` que acumula; ilustra la **diferenciación automática** construida sobre el mismo grafo.

---

## Optimizadores (diap. 62)
Algunos optimizadores importantes en TF/Keras:
- `Class Adadelta`, `Class Adagrad`, `Class Adam`, `Class Adamax`, `Class Ftrl`, `Class Nadam`, `Class Optimizer` (clase base), `Class RMSprop`, `Class SGD`.
- Figura (GIF): superficie 3D tipo punto de silla comparando trayectorias de SGD, Momentum, NAG, Adagrad, Adadelta y RMSprop (fuente: S. Ruder, "Overview of various Optimizers…", `saddle_point_evaluation_optimizers.gif`).

---

## Monitorización con TensorBoard (diap. 63)
- **TensorBoard** es una aplicación de visualización proporcionada por TensorFlow.
- Visualiza **datos resumidos** que se escriben en **archivos de registro** durante el entrenamiento.
- También visualiza el **grafo computacional**, así como información complementaria, como imágenes.
- Figuras: pestaña GRAPH (nodos `weights`, `biases`, `Variable`, `softmax`, `cross_entropy`, `Accuracy`, `input`) y pestaña EVENTS con la curva `xentropy_mean` decreciendo (~2.2 → ~0.2) a lo largo de ~1.600 pasos; opciones X Type: STEP / RELATIVE / WALL.

---

## Dispositivos (diap. 64)
- TF admite diferentes dispositivos de destino: **CPU, GPU, multi GPU**.
- Un grafo puede **distribuirse entre diferentes dispositivos**.
- TF se encarga de **consolidar los datos**.

El siguiente código especifica que el dispositivo para la ejecución debe ser la GPU:
```python
%%timeit
with tf.device("GPU:0"):
    X = tf.random.uniform(shape=(1000, 1000), minval=0, maxval=1)
    assert X.device.endswith("GPU:0")
    tf.matmul(X, X)
```
- `with tf.device("GPU:0")`: ejecución en GPU.
- `assert X.device.endswith("GPU:0")`: evalúa si el tensor está en determinado dispositivo.

---

## Modelamiento en TensorFlow (diap. 65)
Regresión logística en 4 pasos:

**1. Modelo**: $\hat y = \sigma(Xw + b) = \dfrac{1}{1 + e^{-(Xw+b)}}$
```python
def log_reg(w, b, X):
    return 1/(1+tf.math.exp(-(tf.matmul(X, w) + b)))
```

**2. Función de pérdida y optimizador** — entropía cruzada binaria (BCE):
$$\mathcal{L}_{BCE} = -\frac{1}{N}\sum_i \left[y_i \log \hat y_i + (1-y_i)\log(1-\hat y_i)\right]$$
```python
def bce(y_true, y_pred):
    return -tf.math.reduce_mean(y_true*tf.math.log(y_pred) +
            (1. - y_true) * tf.math.log(1. - y_pred))

opt = tf.keras.optimizers.SGD(learning_rate=0.5)
```

**3. Entrenamiento**:
```python
losses = []
for i in range(50):
    print('Epoch',i,end=' ')
    opt.minimize(loss=create_loss_fun,
                 var_list=[w, b])
    print('loss =',create_loss_fun().numpy())
    losses.append(create_loss_fun().numpy())
```
*(`create_loss_fun` no se define en la diapositiva; por analogía con `create_f_x`, sería una función sin argumentos que devuelve la pérdida BCE sobre los datos de entrenamiento.)*

**4. Evaluación**:
```python
log_reg(w, b, X_test)
```

---

## Modelamiento en Keras (diap. 66)
El mismo modelo con la API de alto nivel de Keras:

**1. Modelo**:
```python
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(
            units=1,
            input_shape=(2, ),
            activation=tf.keras.activations.sigmoid))
```

**2. Compilación** (optimizador + función de pérdida):
```python
model.compile(
            optimizer=tf.optimizers.SGD(learning_rate=0.5),
            loss=bce,
            )
```

**3. Entrenamiento**:
```python
model.fit(x=X_train,
          y=Y_train,
          epochs=50
          )
```

**4. Evaluación**:
```python
model.predict(X_test)
model.evaluate(X_test)
```

## TensorFlow vs Keras (diap. 67)
- Comparación lado a lado del mismo flujo (modelo → pérdida/optimizador → entrenamiento → predicción). En TF "puro" se escriben a mano el modelo, la pérdida y el bucle de entrenamiento con `opt.minimize`; en Keras basta con `Sequential` + `Dense`, `compile`, `fit` y `predict`.
- En esta diapositiva la evaluación de Keras aparece solo como `model.predict(X_test)`.

---

## Temas adicionales de TF (diap. 68)
- **Estimadores**: API de alto nivel de TF que simplifica la programación de ML. Encapsula las principales tareas: entrenamiento, evaluación, predicción.
- **Guardar y cargar modelos**: TF proporciona diferentes herramientas para guardar modelos entrenados.
- **API de conjuntos de datos**: facilita el manejo de grandes cantidades de datos, diferentes formatos y transformaciones complicadas.
- **`tf.keras.layers`**: API de alto nivel para construir redes neuronales; métodos para crear capas densas (totalmente conectadas) y capas convolucionales.

---

## Despedida y créditos (diap. 69, 77)
- Profesor: **Fabio Augusto González Osorio, PhD** — https://dis.unal.edu.co/~fgonza/ — Depto. de Ingeniería de Sistemas e Industrial, Facultad de Ingeniería, UNAL Sede Bogotá.
- Créditos (2023): Asistente docente Santiago Toledo Cortés, PhD (C); Coordinador de virtualización Edder Hernández Forero; Diagramador PPT Mario Andrés Rodríguez Triana; Diseño gráfico Clara Valeria Suárez Caballero y Milton R. Pachón Pinzón.

---

## Referencias (diap. 70)
- Pseudo1ntellectual (2008). *Perceptron Research from the 50's & 60's, clip* [Vídeo]. YouTube. https://www.youtube.com/watch?v=cNxadbrN_aI
- McMillan, R. (2013). *Google Hires Brains that Helped Supercharge Machine Learning*. WIRED. https://www.wired.com/2013/03/google-hinton/
- Metz, C. (2013). *Facebook Taps «Deep Learning» Giant for New AI Lab*. WIRED. https://www.wired.com/2013/12/facebook-yann-lecun/
- Novet, J. (2015). *Google says its speech recognition technology now has only an 8% word error rate*. VentureBeat. https://venturebeat.com/business/google-says-its-speech-recognition-technology-now-has-only-an-8-word-error-rate/
- **Silver, D., Huang, A., Maddison, C. et al. (2016). Mastering the game of Go with deep neural networks and tree search. *Nature* 529, 484–489.** https://doi.org/10.1038/nature16961
- Hertz, J. (2022). *IBM Touts SoC as Solution for Faster Deep Learning Training*. https://www.allaboutcircuits.com/news/ibm-touts-soc-as-solution-for-faster-deep-learning-training/
- Simonite, T. (2014). *Chinese Search Giant Baidu Hires Man Behind the "Google Brain"*. MIT Technology Review. https://www.technologyreview.com/2014/05/16/172839/chinese-search-giant-baidu-hires-man-behind-the-google-brain/
- S. (2015). *Here's How Deep Learning Will Accelerate Self-Driving Cars*. SemisMatter. http://www.semismatter.com/2015/03/03/7444/
- Science X (2022). *Integrating digital twins and deep learning for medical image analysis in the era of COVID-19*. https://medicalxpress.com/news/2022-11-digital-twins-deep-medical-image.html

### Obras/papers destacados entre los "Derechos de imágenes" (diap. 71–76)
- McCulloch, W. & Pitts, W. (1943). *A Logical Calculus of the Ideas Immanent in Nervous Activity*.
- Minsky, M. (1987). *Perceptrons* (Expanded ed.). MIT Press.
- Rumelhart, D., Hinton, G. & Williams, R. (1986). *Learning representations by back-propagating errors*. Nature 323, 533–536. https://doi.org/10.1038/323533a0
- Hinton, G., Osindero, S. & Teh, Y. (2006). *A fast learning algorithm for deep belief nets*.
- Krizhevsky, A., Sutskever, I. & Hinton, G. — *ImageNet Classification with Deep Convolutional Neural Networks* (AlexNet; slides CS231b Stanford).
- Vapnik, V. (2013). *The Nature of Statistical Learning Theory* (2.ª ed.). Springer.
- Montavon, G., Orr, G. & Müller, K. (2012). *Neural Networks: Tricks of the Trade* (LNCS 7700). Springer.
- Sun, Y. et al. (2020). *Summarizing CPU and GPU Design Trends with Product Data*. https://arxiv.org/pdf/1911.11313.pdf
- Ruder, S. / Kumar, S. (2016/2021). *Overview of various Optimizers in Neural Networks* (GIF de optimizadores): https://ruder.io/content/images/2016/09/saddle_point_evaluation_optimizers.gif
- Amer, M. (2022). *Large Language Models and Where to Use Them* (Partes 1 y 2), Cohere: https://txt.cohere.ai/llm-use-cases/ y https://txt.cohere.ai/llm-use-cases-p2/
- OpenAI (2022). *DALL·E: Creating Images from Text*.
- El resto son créditos de iconos/imágenes (Flaticon, Adobe Stock, Freepick, Wikimedia, NVIDIA, Papers with Code, etc.).

---

## Menciones de entregables
- Ninguna diapositiva del rango 39–77 menciona actividades, talleres, entregas ni fechas. La única fecha visible es el año de los créditos (2023, diap. 77).
