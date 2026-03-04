# Detector de Lengua de Señas Argentina (LSA) — Random Forest + MediaPipe

Proyecto en Python para **captura de dataset**, **extracción de features de mano**, **entrenamiento** con **RandomForestClassifier** e **inferencia en tiempo real** desde cámara para construir palabras letra por letra.

**Tecnologías:** Python, OpenCV (cv2), MediaPipe, NumPy, Pickle, Scikit-learn.

---

## Estructura del proyecto (orden de archivos)

1. `colectar_imgs.py`  
2. `colectar_nuevas_imgs.py`  
3. `crear_dataset.py`  
4. `limpiar_pickle.py`  
5. `entrenamiento_clasificador.py`  
6. `clasificador_inferencia_palabras.py`

---

## Descripción técnica (pipeline)

### 1) Captura de imágenes (dataset crudo)
- Crea `./data/` y una carpeta por clase (`0..20`).
- Por defecto: **21 clases** y **100 imágenes por clase**.
- Para cada clase, espera a que se presione **Q** y luego captura los frames.

**Salida:** `./data/<clase>/<imagen>.jpg`

---

### 2) Creación de dataset (features con MediaPipe)
- Procesa las imágenes de `./data/`.
- MediaPipe Hands detecta landmarks de la mano.
- Para cada muestra, genera **42 features**:
  - Para cada landmark: `(x - min(x))` y `(y - min(y))` (normalización por mínimos).

**Salida:** `data.pickle` con:
- `data`: lista de vectores de features
- `labels`: lista de etiquetas/clases

---

### 3) Limpieza del dataset (solo 1 mano)
- Filtra y conserva solo muestras con **1 mano** verificando `len(features) == 42`.
- Descarta labels de clases que no existan como carpeta en `./data/`.

**Salida:** `data_1mano.pickle`

---

### 4) Entrenamiento del clasificador
- Carga `data_1mano.pickle`.
- Split:
  - `test_size=0.2`
  - `shuffle=True`
  - `stratify=labels`
- Entrena `RandomForestClassifier()` (parámetros por defecto).
- Imprime accuracy.
- Guarda el modelo en `modelo.p` como:
  - `{'modelo': modelo}`

**Salida:** `modelo.p`

---

### 5) Inferencia en tiempo real (cámara)
- Abre cámara (intenta índice `0`, si falla usa `1`) y configura resolución alta.
- MediaPipe detecta hasta `max_num_hands=4`, pero el flujo utiliza **una mano**:
  - Selecciona la de **mayor área**.
- Genera features con el mismo esquema (42 features normalizados por mínimos).
- Predice clase con el modelo y aplica suavizado por ventana:
  - `VENTANA = 12` frames
  - `UMBRAL_MAYORIA = 0.6`
- Muestra en pantalla:
  - Palabra acumulada
  - Estado de captura
  - Letra dominante

---

## Requisitos

### Dependencias
- `opencv-python`
- `mediapipe`
- `numpy`
- `scikit-learn`

> `pickle`, `time` y `collections` son parte de la librería estándar.

### Archivos/outputs esperados
- `./data/` (dataset de imágenes)
- `data.pickle`
- `data_1mano.pickle`
- `modelo.p`

---

## Uso (paso a paso)

### 1) Capturar dataset inicial

python colectar_imgs.py

### 2) Agregar nuevas imágenes al dataset (sin pisar nombres)

python colectar_nuevas_imgs.py

### 3) Crear dataset de features

python crear_dataset.py

### 4) Limpiar dataset para 1 mano

python limpiar_pickle.py

### 5) Entrenar Random Forest

python entrenamiento_clasificador.py

### 6) Ejecutar inferencia y armar palabras

python clasificador_inferencia_palabras.py

---

## Controles (inferencia)

- S: Iniciar / Pausar captura (limpia el buffer).

- ENTER: Confirmar letra (solo si hay mayoría suficiente).

- B: Borrar última letra.

- C: Limpiar palabra y buffer.

- F: Finalizar (imprime palabra y reinicia).

- ESC: Salir.

---

## Mapeo de clases a letras
- El script de inferencia define un diccionario de etiquetas (índice → letra) con 21 clases: A, B, C, D, F, I, K, L, M, N, O, P, R, S, T, U, V, Y, G, H, J

---

## Notas técnicas

- Vector de entrada al modelo: 42 features (21 landmarks × 2 coords), normalizados por mínimos minx/miny.

- Si hay múltiples manos en escena, se toma solo una (la más grande por área) para mantener consistencia del input.

- El modelo se persiste como {'modelo': modelo} en modelo.p.
