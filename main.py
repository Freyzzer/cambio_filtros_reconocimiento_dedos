import cv2
import numpy as np
import math
import mediapipe as mp

mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# with mp.hands.Hands(
#     static_image_mode=True, 
#     max_num_hands=2, 
#     min_detection_confidence=0.5) as hands:

# Índices de landmarks que nos interesan (ver diagrama de MediaPipe Hands)
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_MCP = 9
 
# Umbral de "pellizco": distancia pulgar-índice / distancia muñeca-nudillo medio
UMBRAL_PELLIZCO = 0.42


# ---------------------------------------------------------------------------
# FILTROS
# Cada filtro recibe una imagen BGR (numpy array) y devuelve otra del mismo
# tamaño con el efecto aplicado. Se aplican sobre todo el frame y luego se
# recortan con la máscara del rectángulo, así que no importa que sean
# "costosos": solo se ven donde corresponde.
# ---------------------------------------------------------------------------

def filtro_sepia(img):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia = cv2.transform(img, kernel)
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)
    return sepia


def filtro_comic(img, niveles=6):
    suavizada = cv2.bilateralFilter(img, d=9, sigmaColor=200, sigmaSpace=200)
    div = 256 // niveles
    posterizada = (suavizada // div * div + div // 2).astype(np.uint8)

    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gris_suave = cv2.medianBlur(gris, 5)
    bordes = cv2.adaptiveThreshold(gris_suave, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 9, 6)
    bordes_bgr = cv2.cvtColor(bordes, cv2.COLOR_GRAY2BGR)

    return cv2.bitwise_and(posterizada, bordes_bgr)


def filtro_pixelado(img, tam_pixel=16):
    h, w = img.shape[:2]
    temp = cv2.resize(img, (max(1, w // tam_pixel), max(1, h // tam_pixel)),
                      interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)


def filtro_bordes(img):
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bordes = cv2.Canny(gris, 60, 150)
    return cv2.cvtColor(bordes, cv2.COLOR_GRAY2BGR)


def filtro_blur(img, k=25):
    """Desenfoque fuerte (gaussiano)."""
    k = k if k % 2 == 1 else k + 1  # el kernel debe ser impar
    return cv2.GaussianBlur(img, (k, k), 0)


def filtro_halftone_rojo(img, tam_celda=8,
    color_punto=(30, 0, 140), color_fondo=(220, 200, 245)):
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gris = clahe.apply(gris)

    h, w = gris.shape
    filas = max(1, h // tam_celda)
    columnas = max(1, w // tam_celda)
    pequena = cv2.resize(gris, (columnas, filas), interpolation=cv2.INTER_AREA)

    salida = np.full((h, w, 3), color_fondo, dtype=np.uint8)
    radio_max = tam_celda / 2 - 1

    for fila in range(filas):
        for col in range(columnas):
            brillo = int(pequena[fila, col])
            factor = ((255 - brillo) / 255) ** 0.7
            radio = int(factor * radio_max)
            if radio > 0:
                cy = fila * tam_celda + tam_celda // 2
                cx = col * tam_celda + tam_celda // 2
                cv2.circle(salida, (cx, cy), radio, color_punto, -1)
    return salida


def filtro_cuadricula(img, tam_celda=25, color_linea=(255, 255, 255), grosor=1):
    """Rejilla de líneas horizontales y verticales sobre la imagen."""
    salida = img.copy()
    h, w = salida.shape[:2]
    for x in range(0, w, tam_celda):
        cv2.line(salida, (x, 0), (x, h), color_linea, grosor, cv2.LINE_AA)
    for y in range(0, h, tam_celda):
        cv2.line(salida, (0, y), (w, y), color_linea, grosor, cv2.LINE_AA)
    return salida


def filtro_popart(img):
    suavizada = cv2.GaussianBlur(img, (5, 5), 0)
    gris = cv2.cvtColor(suavizada, cv2.COLOR_BGR2GRAY)

    salida = np.zeros((*gris.shape, 3), dtype=np.uint8)
    negro = (10, 10, 10)
    rosa = (140, 20, 210)  # BGR -> rosa/magenta
    naranja = (0, 140, 255)  # BGR -> naranja
    blanco = (255, 255, 255)

    salida[gris < 70] = negro
    salida[(gris >= 70) & (gris < 130)] = rosa
    salida[(gris >= 130) & (gris < 200)] = naranja
    salida[gris >= 200] = blanco
    return salida


def filtro_cromatico(img, desplazamiento=6):
    b, g, r = cv2.split(img)
    r = np.roll(r, desplazamiento, axis=1)
    b = np.roll(b, -desplazamiento, axis=1)
    salida = cv2.merge([b, g, r]).astype(np.float32)

    salida[::3, :, :] *= 0.45
    return np.clip(salida, 0, 255).astype(np.uint8)


# Lista de filtros disponibles
FILTROS = [filtro_sepia, filtro_comic, filtro_pixelado, filtro_bordes,
           filtro_blur, filtro_halftone_rojo, filtro_cuadricula,
           filtro_popart, filtro_cromatico]
NOMBRES_FILTROS = ["Sepia", "comic", "Pixelado", "Bordes",
                   "Blur", "Halftone Rojo", "Cuadricula",
                   "Pop Art", "Cromatico"]


# ---------------------------------------------------------------------------
# UTILIDADES GEOMÉTRICAS
# ---------------------------------------------------------------------------
 
def distancia(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
 
 
def ordenar_puntos(pts):
    """Ordena 4 puntos en sentido de un polígono (no cruzado) alrededor
    de su centroide, para poder dibujarlos como cuadrilátero válido aunque
    las manos se muevan o roten."""
    pts = np.array(pts, dtype=np.float32)
    centro = pts.mean(axis=0)
    angulos = np.arctan2(pts[:, 1] - centro[1], pts[:, 0] - centro[0])
    orden = np.argsort(angulos)
    return pts[orden]
 
 
def render_portal(frame, puntos, filtro_func, nombre_filtro):
    """Dibuja el cuadrilátero definido por 'puntos' y aplica el filtro
    únicamente dentro de esa región."""
    ordenados = ordenar_puntos(puntos).astype(np.int32)
 
    # Máscara binaria del polígono
    mascara = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mascara, ordenados, 255)
 
    # Aplicamos el filtro a una copia completa del frame (simple y robusto)
    filtrado = filtro_func(frame.copy())
    if filtrado.shape != frame.shape:
        filtrado = cv2.resize(filtrado, (frame.shape[1], frame.shape[0]))
 
    mascara_3c = cv2.merge([mascara, mascara, mascara])
    resultado = np.where(mascara_3c == 255, filtrado, frame)
 
    # Borde del rectángulo
    cv2.polylines(resultado, [ordenados], isClosed=True,
                   color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
 
    # Etiqueta con el nombre del filtro actual
    x, y = ordenados[0]
    cv2.putText(resultado, nombre_filtro, (int(x), max(20, int(y) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
 
    return resultado
 
 
def esta_pellizcando(landmarks_px):
    """landmarks_px: lista de (x,y) en píxeles para UNA mano (21 puntos)."""
    d_pellizco = distancia(landmarks_px[THUMB_TIP], landmarks_px[INDEX_TIP])
    escala = distancia(landmarks_px[WRIST], landmarks_px[MIDDLE_MCP])
    if escala == 0:
        return False
    return (d_pellizco / escala) < UMBRAL_PELLIZCO
 
 
# ---------------------------------------------------------------------------
# PROGRAMA PRINCIPAL
# ---------------------------------------------------------------------------
 
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return
 
    hands_detector = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
 
    filtro_index = 0
    rectangulo_cerrado_prev = False  # para detectar el flanco de "cierre"
 
    while True:
        ok, frame = cap.read()
        if not ok:
            break
 
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = hands_detector.process(rgb)
 
        puntos_portal = []
        manos_pellizcando = 0
        manos_detectadas = 0
 
        if resultados.multi_hand_landmarks:
            for hand_landmarks in resultados.multi_hand_landmarks:
                manos_detectadas += 1
                pts_px = [(lm.x * w, lm.y * h) for lm in hand_landmarks.landmark]
 
                puntos_portal.append(pts_px[THUMB_TIP])
                puntos_portal.append(pts_px[INDEX_TIP])
 
                if esta_pellizcando(pts_px):
                    manos_pellizcando += 1
 
                # (opcional) dibuja los landmarks de la mano, comenta si molesta
                # mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
 
        # Solo dibujamos el portal si detectamos las 2 manos (4 puntos)
        if manos_detectadas == 2 and len(puntos_portal) == 4:
            frame = render_portal(frame, puntos_portal,
                                   FILTROS[filtro_index],
                                   NOMBRES_FILTROS[filtro_index])
 
        # --- lógica de "cerrar para cambiar de filtro" ---
        rectangulo_cerrado_ahora = (manos_detectadas == 2 and manos_pellizcando == 2)
        if rectangulo_cerrado_ahora and not rectangulo_cerrado_prev:
            filtro_index = (filtro_index + 1) % len(FILTROS)
        rectangulo_cerrado_prev = rectangulo_cerrado_ahora
 
        cv2.imshow("Filtro Portal", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break
 
    cap.release()
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()