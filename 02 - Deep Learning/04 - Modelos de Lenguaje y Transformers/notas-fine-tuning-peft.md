# Fine-tuning eficiente de modelos de lenguaje (PEFT)

Notas de estudio sobre cuántos parámetros hay que entrenar para ajustar un modelo grande y qué se
actualiza realmente dentro de él. Punto de partida: la infografía *Finetuning: 5 Techniques* de Lan Chu
en LinkedIn. Las cifras se recalcularon y se contrastaron con los artículos originales; las
diferencias con la publicación están señaladas al final.

## La idea central

Los pesos de un modelo de lenguaje son, sobre todo, matrices. Ajustar el modelo es sumarle a cada
matriz un cambio:

$$W' = W + \Delta W$$

El *fine-tuning* completo aprende $\Delta W$ entero. Las técnicas PEFT (*parameter-efficient
fine-tuning*) parten de una observación: el cambio que aporta un conjunto de datos pequeño se puede
representar con muchos menos números que la matriz completa.

**Ejemplo de juguete** usado en todas las secciones: una matriz $W$ de 800 × 500.

## 1. Fine-tuning completo

- Se entrenan todos los pesos: 800 × 500 = **400.000 parámetros** por matriz.
- En un modelo de 70B se actualizan los 70.000 millones de parámetros: el 100 %.
- Costo: hay que guardar gradientes y estados del optimizador para cada peso, lo que multiplica la
  memoria necesaria.

## 2. LoRA

Se congela $W$ y se aprende $\Delta W$ como el producto de dos matrices delgadas de rango $r$:

$$\Delta W = B\,A, \qquad B \in \mathbb{R}^{800 \times r},\; A \in \mathbb{R}^{r \times 500}$$

- Con $r = 4$: $B$ tiene 3.200 parámetros y $A$ tiene 2.000, en total **5.200** por matriz, el 1,3 % de
  los 400.000 del ajuste completo.
- En Llama-2-70B, con $r = 64$ en todas las capas lineales de las 80 capas del modelo, LoRA entrena
  **≈ 828 millones** de parámetros, el **1,2 %** del modelo.
- Al terminar, $B\,A$ se puede sumar a $W$, así que la inferencia no se vuelve más lenta.

## 3. LoRA-FA

Igual que LoRA, pero también se congela $A$, que se deja con su inicialización aleatoria. Solo se
entrena $B$.

- Ejemplo de juguete: solo $B$, 800 × 4 = **3.200** parámetros por matriz.
- Llama-2-70B con $r = 64$: **≈ 430 millones**, el **0,6 %** del modelo.
- La ventaja principal no es solo entrenar menos parámetros: como $A$ y $W$ están congelados, no hay
  que guardar las activaciones de entrada completas, solo $XA$, que es mucho más pequeña. El artículo
  reporta hasta 1,4 veces menos memoria que LoRA.

## 4. QLoRA

Mismos adaptadores y mismo número de parámetros entrenables que LoRA. Lo que cambia es el modelo base
congelado, que se guarda en 4 bits en lugar de 16.

- Memoria del modelo base de 70B: 70 × 10⁹ × 2 bytes = **140 GB** en 16 bits, frente a 70 × 10⁹ × 0,5
  bytes = **35 GB** en 4 bits.
- Introduce el formato NF4, la doble cuantización y los optimizadores paginados. Gracias a esto se
  puede ajustar un modelo grande en una sola GPU.
- Lo que se cuantiza es el modelo base; los adaptadores se entrenan en mayor precisión.

## 5. TinyLoRA

Lleva la idea al extremo. El cambio se expresa como una combinación lineal de matrices aleatorias
fijas:

$$\Delta W = \sum_{i=1}^{N} v_i\,\lambda_i$$

- Los $\lambda_i$ son matrices aleatorias que no se entrenan; solo se aprende el vector $v$ de
  tamaño $N$.
- Con *weight tying*, un único $v$ se comparte entre todas las capas del modelo, de modo que $N$ puede
  bajar hasta 1.
- Resultado del artículo: **13 parámetros**, 26 bytes en bf16, llevaron a un modelo Qwen2.5 de 8B al
  **91 % en GSM8K**, entrenando con aprendizaje por refuerzo mediante GRPO.
- Hallazgo clave: con ajuste supervisado (SFT), el modelo necesita actualizaciones **100 a 1.000 veces
  más grandes** para el mismo desempeño. En este régimen, el aprendizaje por refuerzo es mucho más
  eficiente que el ajuste supervisado.
- Es un trabajo de investigación de 2026, sin revisión por pares, y su valor práctico todavía está en
  discusión.

## Resumen

| Técnica | Qué se entrena | Ejemplo 800 × 500 | Modelo de 70B |
|---|---|---:|---:|
| Fine-tuning completo | Todo $W$ | 400.000 | 70.000 M · 100 % |
| LoRA, r = 4 / 64 | $A$ y $B$ | 5.200 | ≈ 828 M · 1,2 % |
| LoRA-FA | Solo $B$ | 3.200 | ≈ 430 M · 0,6 % |
| QLoRA | $A$ y $B$, base en 4 bits | 5.200 | ≈ 828 M · base de 35 GB |
| TinyLoRA | Vector $v$ | $N$ | 13 en total, medido en un modelo de 8B |

## Cálculo de las cifras para 70B

Llama-2-70B tiene 80 capas, dimensión 8.192, capa intermedia de 28.672 y atención con 8 cabezas de
clave y valor de 128 dimensiones, por lo que $k$ y $v$ proyectan a 1.024.

```python
d, ffn, kv, capas, r = 8192, 28672, 1024, 80, 64
matrices = {  # (entrada, salida)
    "q": (d, d), "k": (d, kv), "v": (d, kv), "o": (d, d),
    "gate": (d, ffn), "up": (d, ffn), "down": (ffn, d),
}
lora = sum(r * (ent + sal) for ent, sal in matrices.values()) * capas  # A y B
lora_fa = sum(r * sal for ent, sal in matrices.values()) * capas      # solo B
print(lora, lora / 70e9)        # 828.375.040 -> 1,18 %
print(lora_fa, lora_fa / 70e9)  # 429.916.160 -> 0,61 %
```

## Diferencias con la publicación original

- **LoRA-FA:** el texto dice que se entrenan 2.000 parámetros por matriz, pero si se congela $A$ y se
  entrena $B$, son 800 × 4 = 3.200. La infografía sí muestra 3.200.
- **LoRA en 70B:** el texto da 910 M (1,3 %) porque usa un modelo de juguete con 175.000 matrices de
  800 × 500. Con la arquitectura real de Llama-2-70B y $r = 64$ son ≈ 828 M (1,2 %), la cifra de la
  infografía.
- **TinyLoRA y 70B:** el artículo no mide un modelo de 70B. Los 13 parámetros corresponden a Qwen2.5 de
  8B en GSM8K; aplicarlo a 70B es una extrapolación de la publicación.

## Para practicar

1. Con la biblioteca `peft` de Hugging Face, ajustar con LoRA un modelo pequeño de clasificación de
   texto en español, por ejemplo RoBERTuito para sentimiento, y comparar el macro-F1 y la memoria
   frente al ajuste completo.
2. Variar el rango $r$ (4, 8, 16, 64) y graficar desempeño contra parámetros entrenables.
3. Repetir el experimento cargando el modelo en 4 bits con `bitsandbytes`, que es el esquema de QLoRA.

## Referencias

- Hu, E. J. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685.
- Dettmers, T. et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. arXiv:2305.14314.
- Zhang, L. et al. (2023). *LoRA-FA: Memory-efficient Low-rank Adaptation for Large Language Models
  Fine-tuning*. arXiv:2308.03303.
- Morris, J. X., Mireshghallah, N., Ibrahim, M. y Mahloujifar, S. (2026). *Learning to Reason in 13
  Parameters*. arXiv:2602.04118.
- Touvron, H. et al. (2023). *Llama 2: Open Foundation and Fine-Tuned Chat Models*. arXiv:2307.09288.
