import os, re
import numpy as np
import cv2

#Verificar que exista ya la carpeta data
DATA_DIR = './data'
os.makedirs(DATA_DIR, exist_ok=True)

num_clases = 21
nuevas_por_clase = 100
clases_objetivo = None
nombre_fmt = "{:06d}.jpg"

#Configuracion de camara
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

for _ in range(15):
    cap.read()

def frame_valido(ret, frame, thr=8.0):
    return ret and frame is not None and frame.size > 0 and float(np.mean(frame)) > thr

#Verifico si existe la cantidad de carpetas iguales a numero de clases
for j in range(num_clases):
    os.makedirs(os.path.join(DATA_DIR, str(j)), exist_ok=True)

#Calculo cual es el siguiente indice en las fotos de cada carpeta
_pat = re.compile(r'^(\d+)\.(jpg|jpeg|png)$', re.IGNORECASE)
def proximo_indice(ruta_carpeta: str) -> int:
    nums = []
    for n in os.listdir(ruta_carpeta):
        m = _pat.match(n)
        if m:
            nums.append(int(m.group(1)))
    return (max(nums) + 1) if nums else 0

# Selección de clases a actualizar
todas = list(range(num_clases))
clases = [int(c) for c in (clases_objetivo if clases_objetivo is not None else todas)]

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
print(f"Clases a actualizar: {clases}")

for j in clases:
    ruta = os.path.join(DATA_DIR, str(j))
    os.makedirs(ruta, exist_ok=True)

    start_idx = proximo_indice(ruta)               # << continúa después del último archivo
    objetivo  = start_idx + nuevas_por_clase

    print(f'\nClase {j}: agrego {nuevas_por_clase} imágenes (desde índice {start_idx}).')
    # Pantalla de espera hasta 'Q'
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Presione "Q".',
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        if cv2.waitKey(25) == ord('q'):
            break

    #Empiezo a capturar los frames
    idx = start_idx
    while idx < objetivo:
        ret, frame = cap.read()
        if not frame_valido(ret, frame):
            # no muestrar ni guardar si el frame es inválido/negro
            if (cv2.waitKey(1) & 0xFF) == 27:
                cap.release(); cv2.destroyAllWindows()
                raise SystemExit("Cancelado por el usuario (ESC).")
            continue

        cv2.imshow('frame', frame)

        if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
            cap.release(); cv2.destroyAllWindows()
            raise SystemExit("Cancelado por el usuario (ESC).")

        out_path = os.path.join(ruta, nombre_fmt.format(idx))
        if cv2.imwrite(out_path, frame):
            idx += 1

print("\nAgregado de imágenes finalizado.")
cap.release()
cv2.destroyAllWindows()