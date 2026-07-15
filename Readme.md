# Filtro "Portal" con las manos

Detecta el pulgar y el índice de cada mano (4 puntos), los usa como vértices
de un cuadrilátero y aplica un filtro visual solo dentro de esa región,
como en tu video de referencia. Al pellizcar con ambas manos (cerrar el
rectángulo) y volver a abrir, cambia al siguiente filtro.

## Instalación

```bash
pip install -r requirements.txt
```

> Nota: fijé `mediapipe==0.10.13` en requirements.txt porque en algunos
> entornos (según SO/arquitectura) versiones más nuevas de mediapipe no
> incluyen el módulo `solutions` (la API clásica de Hands). Si en tu
> máquina `mediapipe` más reciente sí trae `mediapipe.solutions`, puedes
> usar esa versión sin problema.

## Uso

```bash
python main.py
```

- Muestra ambas manos a la cámara con el pulgar y el índice extendidos:
  esos 4 dedos son las esquinas del rectángulo.
- Mueve o rota las manos: el rectángulo (y el filtro dentro de él) te sigue.
- Pellizca (junta pulgar e índice) con **ambas manos a la vez** y vuelve a
  abrir: pasa al siguiente filtro de la lista.
- `ESC` para salir.

## Filtros incluidos

1. Sepia
2. comic (puntos negro/blanco)
3. Pixelado
4. Bordes (Canny)
5. Blur
6. Cuadricula (rejilla de líneas horizontales/verticales)
7. Pop Art (posterizado naranja/rosa/negro/blanco)
8. Cromatico (aberración cromática RGB + líneas de escaneo tipo VHS)

## Personalizar

- **Agregar/quitar filtros**: edita las listas `FILTROS` y
  `NOMBRES_FILTROS` en `main.py`. Cada filtro es una función
  `imagen_bgr -> imagen_bgr` del mismo tamaño.
- **Sensibilidad del pellizco**: ajusta `UMBRAL_PELLIZCO` (más bajo =
  hay que juntar más los dedos para que cuente como pellizco).
- **Tamaño de los puntos del halftone / píxeles**: parámetros
  `tam_celda` / `tam_pixel` en `filtro_halftone` / `filtro_pixelado`.

## Cómo funciona (resumen técnico)

1. **MediaPipe Hands** detecta hasta 2 manos y da 21 landmarks por mano.
   Se usan los landmarks `4` (punta del pulgar) y `8` (punta del índice).
2. Con los 4 puntos (2 por mano) se arma un cuadrilátero, ordenando los
   puntos por ángulo respecto a su centroide para que el polígono no se
   auto-cruce aunque gires o muevas las manos.
3. Se crea una **máscara binaria** del polígono con
   `cv2.fillConvexPoly`, se aplica el filtro a una copia completa del
   frame, y se combinan frame original + frame filtrado usando la máscara
   (`np.where`), de forma que el filtro solo aparece dentro del rectángulo.
4. Para detectar el "cierre" del rectángulo se mide, por mano, la
   distancia pulgar-índice normalizada por el tamaño de la mano (distancia
   muñeca-nudillo medio), para que el umbral funcione sin importar qué tan
   cerca esté la mano de la cámara. Cuando **ambas manos** pasan de
   "abiertas" a "pellizcando" en el mismo frame, se avanza al siguiente
   filtro (con detección de flanco, para que no cambie de filtro en cada
   frame mientras mantienes el pellizco).
