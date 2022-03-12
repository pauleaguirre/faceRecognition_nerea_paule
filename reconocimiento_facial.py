#!/usr/bin/env python
# coding: utf-8

# In[56]:

import base64
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
from elasticsearch import Elasticsearch
from PIL import Image
from io import BytesIO
import mtcnn
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

import cv2
import os



# ### Lectura del dataset

# In[57]:


df=pd.read_excel('informacion_imagenes.xlsx')
df.head(5)
df.shape


# ### Directorios

# In[58]:


path = "Imagenes/"
path_nuevo = "Imagenes_nuevas/"


# ### Lectura de imagenes y cambio de tamaño (guardarlas otra vez)

# In[59]:


for i in df.nombre_imagen.values: 
    img = Image.open(path + i)

    img = img.resize((2200, 3000), Image.BILINEAR) #Cambio de tamaño
    #print(img.height, img.width)
    img.save(path_nuevo + i)
    imgplot = plt.imshow(img)
    plt.show()


# ### Lectura imagenes nuevas (mismo tamaño)
# In[60]:


for i in df.nombre_imagen.values: 
    img = mpimg.imread(path_nuevo + i)
    imgplot = plt.imshow(img)
    plt.show()


# In[61]:

for i in df.nombre_imagen.values: 
    img = mpimg.imread(path_nuevo + i)
    print(type(img))

# ### Codificacion imagenes

# In[62]:

lista_codigos = []
for i in df.nombre_imagen.values: 
    with open(path_nuevo + i, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    encoded_string=encoded_string.decode("utf-8")
    #print(encoded_string)
    lista_codigos.append(encoded_string)


# In[63]:

len(lista_codigos) == len(df)


# In[64]:

df["codigo_imagen"] = lista_codigos
df.head()


# ### Crear cliente elasticsearch

# In[65]:


es = Elasticsearch([{'scheme': 'http', 'host': '192.168.56.103', 'port': 9200}])


# In[66]:


# Borramos el indice si ya está creado
es.indices.delete(index='index_datos', ignore=[400, 404])
es.indices.delete(index='index_modelos', ignore=[400, 404])
es.indices.delete(index='index_predicciones', ignore=[400, 404])


# ### Indexar imagenes

# In[67]:


#Creamos el diccionario
doc = {}
lista = []
for i in df.index: 
    doc[i] = {
        "nombre_imagen": df.nombre_imagen[i], 
        "Nombre": df.Nombre[i], 
        "Apellido": df.Apellido[i], 
        "Mascarilla": df.Mascarilla[i], 
        "codigo_imagen": df.codigo_imagen[i], 
        "modelo" : df.modelo[i]
    
    }
    lista.append(doc[i])


# In[68]:


#Indexamos los valores y creamos el indice
j = 0
for document in lista: 
    res = es.index(index = 'index_datos', id = j, document = document)
    j+= 1


# ### Modelo

# #### Bajamos de elasticsearch las fotos

# In[69]:


X_train = []
X_test = []

y_train = []
y_test = []

for i in df.index:
    res = es.get(index = 'index_datos', id = i)
    train_test = res['_source']['modelo']
    nombre = res['_source']['Nombre']

    if train_test == 'train':
        y_train.append(nombre)
        img = res['_source']['codigo_imagen'].encode()
        im = Image.open(BytesIO(base64.b64decode(img)))
        X_train.append(im)
    elif train_test == "test":
            y_test.append(nombre)
            img = res['_source']['codigo_imagen'].encode()
            im = Image.open(BytesIO(base64.b64decode(img)))
            X_test.append(im)


# In[70]:


# Convertimos a array y extraemos la cara
def extract_face(image, required_size=(200, 200)):
    # Convertir en array
    pixels = np.asarray(image)
    # Crear el detector de caras
    detector = mtcnn.MTCNN()
    # Detectar las caras en la imagen
    results = detector.detect_faces(pixels)
    # Extraer el delimmitador de la primera cara
    x1, y1, width, height = results[0]['box']
    # Gestionar pixels negativos
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # Extraer la cara
    face = pixels[y1:y2, x1:x2]
    # Redimensionar pixels al tamñao modelo y convertir en array
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array


# In[71]:

# Extraer la cara de la primera imagen
pixels = extract_face(X_train[0])
plt.imshow(pixels)
plt.show()
print(pixels.shape) #(200,200,3)


# In[72]:


#aplicamos la funcion sobre todas las fotos para extraer todas las caras
def load_face(datos):
    caras = list()
    # enumerate files
    # Para cada imagen extrae la cara y las junta en una lista llamada caras
    for image in datos:
        face = extract_face(image)
        caras.append(face)
    return caras


# In[74]:

X_train = load_face(X_train)
X_test = load_face(X_test)


# ### Modelo

# In[79]:


#Redimensionamos
x_train = np.array(X_train)
x_test = np.array(X_test)

x_train = x_train.reshape(len(X_train),-1)
x_test = x_test.reshape(len(X_test),-1)

x_train = x_train.flatten().reshape(len(X_train),-1)
x_test = x_test.flatten().reshape(len(X_test),-1)


# In[80]:


grid = GridSearchCV(LinearSVC(random_state= 0), {'C': [0.0001, 1.0, 2.0, 4.0, 8.0,100.0], "max_iter" : [2000, 50000]}, cv = 3, verbose = 3)
grid.fit(x_train, y_train)


# In[87]:

C = [0.0001, 0.0001, 0.0001,
     0.0001, 0.0001, 0.0001,
     1.0, 1.0,  1.0,
     1.0, 1.0,  1.0,
     2.0, 2.0, 2.0,
     2.0, 2.0, 2.0,
     4.0, 4.0, 4.0,
     4.0, 4.0, 4.0,  
     8.0, 8.0, 8.0,
     8.0, 8.0, 8.0,
     100,100,100,
     100,100,100]

kFold = [1, 2, 3, 
         1, 2, 3,
         1, 2, 3, 
         1, 2, 3,
         1, 2, 3, 
         1, 2, 3,
         1, 2, 3, 
         1, 2, 3,
         1, 2, 3, 
         1, 2, 3,
         1, 2, 3, 
         1, 2, 3]

max_iter = [2000, 2000, 2000, 
            50000, 50000, 50000, 
            2000, 2000, 2000, 
            50000, 50000, 50000, 
            2000, 2000, 2000, 
            50000, 50000, 50000, 
            2000, 2000, 2000, 
            50000, 50000, 50000, 
            2000, 2000, 2000, 
            50000, 50000, 50000, 
            2000, 2000, 2000, 
            50000, 50000, 50000]

accuracy= [0.625,0.688, 0.688, 
           0.625,0.688, 0.688, 
           0.625,0.688, 0.688, 
           0.625,0.688, 0.688, 
           0.625,0.688, 0.688, 
           0.625,0.688, 0.688, 
           0.625,0.688, 0.688, 
           0.625,0.688, 0.688, 
           0.625,0.688, 0.688, 
           0.625,0.688, 0.688, 
           0.625,0.688, 0.688, 
           0.625,0.688, 0.688] 



modelo = ["LinearSVC", "LinearSVC", "LinearSVC",  
          "LinearSVC", "LinearSVC", "LinearSVC",
          "LinearSVC", "LinearSVC", "LinearSVC", 
          "LinearSVC", "LinearSVC", "LinearSVC", 
          "LinearSVC", "LinearSVC", "LinearSVC", 
          "LinearSVC", "LinearSVC", "LinearSVC", 
          "LinearSVC", "LinearSVC", "LinearSVC",  
          "LinearSVC", "LinearSVC", "LinearSVC", 
          "LinearSVC", "LinearSVC", "LinearSVC",  
          "LinearSVC", "LinearSVC", "LinearSVC",
          "LinearSVC", "LinearSVC", "LinearSVC", 
          "LinearSVC", "LinearSVC", "LinearSVC"]


# In[88]:

parametros = pd.DataFrame(list(zip(modelo,C,kFold,max_iter,accuracy)), 
                  columns = ['modelo','valor_parametro', 'kFold','max_iter','accuracy'])
parametros.head()
##

# ### Subimos modelos al indice

# In[89]:

lista = []
doc = {}

for fila in parametros.index: 
    doc[fila] = {
        "modelo" : parametros.modelo[fila], 
        "C" : parametros.valor_parametro[fila], 
        "kFold" : parametros.kFold[fila],
        "max_iter": parametros.max_iter[fila],
        "accuracy" : parametros.accuracy[fila]
    }
    lista.append(doc[fila])


# In[90]:

#Indexamos los valores y creamos el indice
j = 0
for document in lista: 
    res = es.index(index = 'index_modelos', id = j, document = document)
    j+= 1


# ### Escoger mejor modelo

# In[83]:

# Best score
grid.best_score_


# In[84]:

# Best params
grid.best_params_


# In[85]:

model = grid.best_estimator_
model.fit(x_train, y_train)


# In[128]:

labels = model.predict(x_test)
accuracy_score(y_test, labels)


# ### Indexar mejor modelo en Elasticsearch

# In[134]:

modelo = ['LinearSVC(C=0.0001, max_iter=2000, random_state=0)',
'LinearSVC(C=0.0001, max_iter=2000, random_state=0)',
'LinearSVC(C=0.0001, max_iter=2000, random_state=0)',
'LinearSVC(C=0.0001, max_iter=2000, random_state=0)',
'LinearSVC(C=0.0001, max_iter=2000, random_state=0)',
'LinearSVC(C=0.0001, max_iter=2000, random_state=0)',
'LinearSVC(C=0.0001, max_iter=2000, random_state=0)',
'LinearSVC(C=0.0001, max_iter=2000, random_state=0)',
'LinearSVC(C=0.0001, max_iter=2000, random_state=0)',
'LinearSVC(C=0.0001, max_iter=2000, random_state=0)',
'LinearSVC(C=0.0001, max_iter=2000, random_state=0)',
'LinearSVC(C=0.0001, max_iter=2000, random_state=0)',
'LinearSVC(C=0.0001, max_iter=2000, random_state=0)',
'LinearSVC(C=0.0001, max_iter=2000, random_state=0)',
'LinearSVC(C=0.0001, max_iter=2000, random_state=0)',
'LinearSVC(C=0.0001, max_iter=2000, random_state=0)',
'LinearSVC(C=0.0001, max_iter=2000, random_state=0)',
'LinearSVC(C=0.0001, max_iter=2000, random_state=0)',
'LinearSVC(C=0.0001, max_iter=2000, random_state=0)',
'LinearSVC(C=0.0001, max_iter=2000, random_state=0)',
'LinearSVC(C=0.0001, max_iter=2000, random_state=0)',
'LinearSVC(C=0.0001, max_iter=2000, random_state=0)',
'LinearSVC(C=0.0001, max_iter=2000, random_state=0)',
'LinearSVC(C=0.0001, max_iter=2000, random_state=0)',
'LinearSVC(C=0.0001, max_iter=2000, random_state=0)',
'LinearSVC(C=0.0001, max_iter=2000, random_state=0)',
'LinearSVC(C=0.0001, max_iter=2000, random_state=0)',
'LinearSVC(C=0.0001, max_iter=2000, random_state=0)'
]
predicciones = ['Gorka', 'Alba', 'Gorka', 'Gorka', 'Gorka', 'Unax', 'Eneko',
       'Alba', 'Unax', 'Sara', 'Alba', 'Paule', 'Paule', 'Paule', 'Alba',
       'Alba', 'Paule', 'Paule', 'Paule', 'Alba', 'Paule', 'Paule',
       'Sara', 'Sara', 'Sara', 'Alba', 'Sara', 'Unax']

accuracy_predicciones = [0.4285, 0.4285, 0.4285, 0.4285, 0.4285,
                         0.4285, 0.4285, 0.4285, 0.4285, 0.4285,
                         0.4285, 0.4285, 0.4285, 0.4285, 0.4285,
                         0.4285, 0.4285, 0.4285, 0.4285, 0.4285,
                         0.4285, 0.4285, 0.4285, 0.4285, 0.4285,
                         0.4285, 0.4285, 0.4285]


# In[135]:


predicciones_finales = pd.DataFrame(list(zip(modelo,predicciones,y_test,accuracy_predicciones)), 
                  columns = ['modelo','predicciones', 'y_test', 'accuracy_predicciones'])
predicciones_finales.head(12)

# In[ ]:

lista = []
doc = {}

for fila in predicciones_finales.index: 
    doc[fila] = {
        "modelo" : predicciones_finales.modelo[fila], 
        "predicciones" : predicciones_finales.predicciones[fila], 
        "y_test" : predicciones_finales.y_test[fila], 
        "accuracy_pred" : predicciones_finales.accuracy_predicciones[fila]
    }
    lista.append(doc[fila])


# In[ ]:


#Indexamos los valores y creamos el indice
j = 0
for document in lista: 
    res = es.index(index = 'index_predicciones', id = j, document = document)
    j+= 1


# ### Foto desde webcam

# In[121]:


camara = cv2.VideoCapture(0)

leido, frame = camara.read()

if leido == True:
	cv2.imwrite("imagen.jpg", frame)
	print("Imagen tomada correctamente")
else:
	print("Error al acceder a la cámara")

"""
	Al final liberamos o soltamos la cámara
"""
camara.release()


# In[122]:


imagen = Image.open("imagen.jpg")
imagen.show()


# In[118]:


cara = extract_face(imagen)
plt.imshow(cara)


# In[119]:


foto = np.array(cara)
foto = foto.reshape(len(cara), -1)
foto = foto.flatten().reshape(len(cara), -1)


# In[133]:


prediccion = model.predict(foto.reshape(1,-1))
print('La prediccion para el rostro es: ', prediccion[0])

os.remove("imagen.jpg")
