# Detector-de-Lenguaje-de-Senas-Argentino
Detector de Lengua de Señas Argentina (LSA) en tiempo real. Captura imágenes con OpenCV, extrae 21 landmarks de mano con MediaPipe y genera 42 features (x,y normalizados). Entrena un RandomForestClassifier y permite inferencia en cámara con suavizado por ventana para “armar” palabras por teclado.
