import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Creo un diccionario con los datos del archivo .pickle limpio
data_diccio = pickle.load(open("./data_1mano.pickle", "rb"))

#Convierto a data y labels en un arreglo de numpy
data = np.asarray(data_diccio['data'])
labels = np.asarray(data_diccio['labels'])

#Variables de entrenamiento y testeo
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

modelo = RandomForestClassifier()

modelo.fit(x_train, y_train)

y_prediccion = modelo.predict(x_test)

score = accuracy_score(y_prediccion, y_test)

print('{}% de ejemplos fueron clasificados correctamente.'.format(score * 100))

f = open('modelo.p', 'wb')
pickle.dump({'modelo': modelo}, f)
f.close()