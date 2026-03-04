import os
import numpy as np
import cv2

#Creacion de la carpeta data donde se guardaran las fotos
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

num_de_clases = 21
tam_dataset = 100
nombre_fmt = "{:06d}.jpg"

#Configuracion de la camara
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

#Ciclo para capturar los frames en el video
for j in range(num_de_clases):
    #Verifico que la carpeta y creo la carpeta para la seña en especifico (0,1,2,3,...)
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Recoleción de datos para clase {}'.format(j))

    #Pantalla de espera hasta que se apreta q
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Presione "Q"', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    #Capturo la cantidadd de frames equivalentes a tam_dataset
    contador = 0
    while contador < tam_dataset:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        out_path = os.path.join(DATA_DIR, str(j), nombre_fmt.format(contador))
        cv2.imwrite(out_path, frame)

        contador += 1

cap.release()
cv2.destroyAllWindows()