import pickle
import cv2
import mediapipe as mp
import numpy as np
from collections import deque, Counter
import time

#Cargar modelo
modelo_diccio = pickle.load(open('./modelo.p', 'rb'))
modelo = modelo_diccio.get('modelo', modelo_diccio.get('model'))
label_encoder = modelo_diccio.get('label_encoder', None)

#Diccionario de etiquetas
labels_diccio = {
    0:'A', 1:'B', 2:'C', 3:'D', 4:'F', 5:'I', 6:'K', 7:'L',
    8:'M', 9:'N', 10:'O', 11:'P', 12:'R', 13:'S', 14:'T', 15:'U', 16:'V', 17:'Y',
    18: 'G', 19: 'H', 20: 'J'
}

# Configuracion de camara
def abrir_camara():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("No se puede abrir la cámara (0/1). Cerrá apps que la usen.")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1366)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    cap.set(cv2.CAP_PROP_FPS, 30)
    for _ in range(15): cap.read()
    return cap

cap = abrir_camara()

# MediaPipe Hands (modo video)
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=4,              # puede haber mucha gente
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

#Calcular el area de mano mas grande
def caja_area(hand, W, H):
    xs = [int(lm.x * W) for lm in hand.landmark]
    ys = [int(lm.y * H) for lm in hand.landmark]
    x1, y1 = max(min(xs), 0), max(min(ys), 0)
    x2, y2 = min(max(xs), W-1), min(max(ys), H-1)
    return (x2 - x1 + 1) * (y2 - y1 + 1), (x1, y1, x2, y2)

#Asegurar que solo toma los datos de 1 mano asi no se cierra el programa
def features_42_min_only(hand):
    # misma normalización que entrenaste: (x - minx), (y - miny)
    xs = [lm.x for lm in hand.landmark]
    ys = [lm.y for lm in hand.landmark]
    minx, miny = min(xs), min(ys)
    feats = []
    for lm in hand.landmark:
        feats.append(lm.x - minx)
        feats.append(lm.y - miny)
    return np.array(feats, dtype=np.float32)

#Funcion para las letras de la palabra
def letra_desde_indice(idx):
    if label_encoder is not None and isinstance(idx, (np.integer, int)):
        try:
            return str(label_encoder.inverse_transform([idx])[0])
        except Exception:
            pass
    return labels_diccio.get(int(idx), str(idx))

# Suavizado por ventana
VENTANA = 12             # frames para mayoría
UMBRAL_MAYORIA = 0.6     # al menos 60% de la ventana
buffer_preds = deque(maxlen=VENTANA)

# Estado de interacción
capturando_letra = False
letra_actual = "-"
palabra = ""
ultimo_commit_ts = 0
COOLDOWN_COMMIT = 0.3

#Video
while True:
    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        if (cv2.waitKey(10) & 0xFF) == 27:
            break
        continue

    H, W = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(frame_rgb)

    letra_dominante = "-"
    conf_dominante  = 0.0
    caja_dibujar = None

    if res.multi_hand_landmarks:
        # Se elige una mano: la de mayor área de caja
        candidatos = []
        for hand in res.multi_hand_landmarks:
            area, caja = caja_area(hand, W, H)
            candidatos.append((area, caja, hand))
        candidatos.sort(reverse=True, key=lambda t: t[0])
        _, caja_dibujar, hand = candidatos[0]

        # Dibujar puntos de referencia de esa mano
        mp_draw.draw_landmarks(
            frame, hand, mp_hands.HAND_CONNECTIONS,
            mp_style.get_default_hand_landmarks_style(),
            mp_style.get_default_hand_connections_style()
        )

        # Vector de 42 features
        feats = features_42_min_only(hand)

        # Predicción del modelo
        try:
            pred_idx = int(modelo.predict([feats])[0])
            # abilidad opcional
            try:
                proba = modelo.predict_proba([feats])[0]
                conf = float(np.max(proba))
            except Exception:
                conf = 1.0  # si RF no tiene probabilidades, uso 1.0 simbólico

            # Solo se acumula si esta capturando
            if capturando_letra:
                buffer_preds.append(pred_idx)

            # Dominante en la ventana
            if len(buffer_preds) > 0:
                conteo = Counter(buffer_preds)
                idx_dom, count_dom = conteo.most_common(1)[0]
                letra_dominante = letra_desde_indice(idx_dom)
                conf_dominante  = count_dom / len(buffer_preds)
            else:
                letra_dominante = letra_desde_indice(pred_idx)
                conf_dominante  = conf

        except Exception:
            # Si algo falla
            letra_dominante = "-"
            conf_dominante  = 0.0

    # UI: recuadro, palabra y estado
    if caja_dibujar is not None:
        x1, y1, x2, y2 = caja_dibujar
        cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 30, 30), 2)

    cv2.putText(frame, f"Palabra: {palabra}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    estado = "CAPTURANDO" if capturando_letra else "Pausa"
    cv2.putText(frame, f"Estado: {estado}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if capturando_letra else (0, 0, 255), 2)
    cv2.putText(frame, f"Letra actual: {letra_dominante}  (Prob: {conf_dominante:.2f})", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Instrucciones
    cv2.putText(frame, "S: Iniciar/Pausa  ENTER: Confirmar  B: Borrar  C: Limpiar  F: Finalizar  ESC: Salir",
                (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF

    # Teclas
    if key == 27:  # ESC
        break
    elif key in (ord('s'), ord('S')):
        capturando_letra = not capturando_letra
        buffer_preds.clear()
    elif key == 13:  # ENTER
        now = time.time()
        if now - ultimo_commit_ts > COOLDOWN_COMMIT:
            # Confirmar letra solo si hay mayoría suficiente
            if len(buffer_preds) > 0:
                conteo = Counter(buffer_preds)
                idx_dom, count_dom = conteo.most_common(1)[0]
                estabilidad = count_dom / len(buffer_preds)
                if estabilidad >= UMBRAL_MAYORIA:
                    palabra += letra_desde_indice(idx_dom)
                    buffer_preds.clear()
            ultimo_commit_ts = now
    elif key in (ord('b'), ord('B')):
        if len(palabra) > 0:
            palabra = palabra[:-1]
    elif key in (ord('c'), ord('C')):
        palabra = ""
        buffer_preds.clear()
    elif key in (ord('f'), ord('F')):
        print(f"[FINALIZADA] Palabra: {palabra}")
        palabra = ""
        buffer_preds.clear()

cap.release()
hands.close()
cv2.destroyAllWindows()
