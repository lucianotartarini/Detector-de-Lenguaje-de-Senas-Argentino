#Limpio el archivo data.pickle para que guarde solo los features de 42 y se pueda entrenar el modelo
import os, pickle
from collections import Counter

DATA_DIR = "./data"
clases_validas = {c for c in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, c))}

d = pickle.load(open("data.pickle", "rb"))
X_raw, y_raw = d["data"], d["labels"]

# Quedarse SOLO con muestras de 1 mano (len=42) y clases que aún existen
X, y = [], []
for v, lab in zip(X_raw, y_raw):
    if lab in clases_validas and hasattr(v, "__len__") and len(v) == 42:
        X.append(v); y.append(lab)

print("Antes:", Counter(len(v) if hasattr(v, "__len__") else -1 for v in X_raw))
print("Después:", Counter(len(v) for v in X), "| Muestras:", len(X), "| Clases:", len(set(y)))

pickle.dump({"data": X, "labels": y}, open("data_1mano.pickle", "wb"))
print("Guardado: data_1mano.pickle")
