# 1. Cargar las librerías necesarias y sino instalarlas 

se pueden instalar con el siguiente comando install.packages("libreria") desde Rstudio

las librerias son las siguietes:

1. rpart
2. random forest


# 2. Descomprimir el archivo dataset en la misma carpeta donde se encuentre en archivo R

# 3. Se pueden consultar los diccionarios de datos en la carpeta "Diccionarios"

# 4. Conformacion del archivo R

El archivo R esta conformado de chunks de los arboles de decisiones y los random forests ademas de sus respectivas predicciones



```

#DOMINIO
#P01I03 ¿Qué edad tenia al irse?	Escala
#P01I05 ¿En qué país se encuentra actualmente?	Nominal


library(rpart)
library(rpart.plot)

migracion <- read.csv('C:/Users/Andres/Desktop/Proyecto parte 2/migracion2.csv', sep = ',')


# Crear el árbol de decisión para la condición
arbol_condicion1 <- rpart(DOMINIO ~
                           
P01I03 +
P01I05,     
                         
                         
                         data = migracion, method = "class")

# Visualizar el árbol de decisión
rpart.plot(arbol_condicion1, type=2, extra=0, under = TRUE, fallen.leaves = TRUE, box.palette = "BuGn",
           main = "Predicción del dominio", cex = 1)


# Caso para predicción 1
migracion_pred <- data.frame(

  P01I05 = c(3001), #¿En qué país se encuentra actualmente?
  P01I03 = c(50) #¿Qué edad tenia al irse?
  
)

# Realizar predicción
result_condicion <- predict(arbol_condicion1, migracion_pred, type = "class")
result_condicion  # Dominio de Estudio

# Caso para predicción 2
migracion_pred <- data.frame(

  P01I05 = c(3003), #¿En qué país se encuentra actualmente?
  P01I03 = c(30) #¿Qué edad tenia al irse?
  
)

# Realizar predicción
result_condicion <- predict(arbol_condicion1, migracion_pred, type = "class")
result_condicion  # Dominio de Estudio

library(rpart)
library(rpart.plot)

vivienda <- read.csv('C:/Users/Andres/Desktop/Proyecto parte 2/vivienda.csv', sep = ',')


# Crear el árbol de decisión para la condición
arbol_condicion2 <- rpart(DOMINIO ~

P01A02 +
P01A03 +
P01A04 ,
                         
                         data = vivienda, method = "class")

# Visualizar el árbol de decisión
rpart.plot(arbol_condicion2, type=2, extra=0, under = TRUE, fallen.leaves = TRUE, box.palette = "BuGn",
           main = "Predicción del dominio", cex = 1)

# Caso de ejemplo para predicción
vivienda_pred <- data.frame(


P01A04 = c(5), #¿Cuál es el material predominante en el piso?
P01A03 = c(3), #¿Cuál es el material predominante en el techo?
P01A02 = c(5)  #¿Cuál es el material predominante en las paredes exteriores?
  
)

# Realizar predicción
result_condicion <- predict(arbol_condicion2, vivienda_pred, type = "class")
result_condicion  # Resultado DOMINIO

# Caso de ejemplo para predicción
vivienda_pred <- data.frame(


P01A04 = c(2), #¿Cuál es el material predominante en el piso?
P01A03 = c(2), #¿Cuál es el material predominante en el techo?
P01A02 = c(3)  #¿Cuál es el material predominante en las paredes exteriores?
  
)

# Realizar predicción
result_condicion <- predict(arbol_condicion2, vivienda_pred, type = "class")
result_condicion  # Resultado DOMINIO

library(rpart)
library(rpart.plot)

vivienda2 <- read.csv('C:/Users/Andres/Desktop/Proyecto parte 2/vivienda.csv', sep = ',')


# Crear el árbol de decisión para la condición
arbol_condicion3 <- rpart(DEPTO ~

P01A05A +
P01A05B +
P01A05C 
,
                         
                         data = vivienda2, method = "class")

# Visualizar el árbol de decisión
rpart.plot(arbol_condicion3, type=2, extra=0, under = TRUE, fallen.leaves = TRUE, box.palette = "BuGn",
           main = "Predicción del departamento", cex = 1)

# Caso de ejemplo para predicción
vivienda_pred2 <- data.frame(

P01A05B = c(2), # ¿Esta vivienda esta conectada a una red de drenajes?
P01A05C = c(2), # ¿Esta vivienda esta conectada a una red de distribución de energía eléctrica?
P01A05A = c(2)  # ¿Esta vivienda esta conectada a una red de distribución de agua?
  
)

# Realizar predicción
result_condicion <- predict(arbol_condicion3, vivienda_pred2, type = "class")
result_condicion  # Mostrar el resultado de la predicción

# Caso de ejemplo para predicción
vivienda_pred2 <- data.frame(

P01A05B = c(1), # ¿Esta vivienda esta conectada a una red de drenajes?
P01A05C = c(1), # ¿Esta vivienda esta conectada a una red de distribución de energía eléctrica?
P01A05A = c(1)  # ¿Esta vivienda esta conectada a una red de distribución de agua?
  
)

# Realizar predicción
result_condicion <- predict(arbol_condicion3, vivienda_pred2, type = "class")
result_condicion  # Mostrar el resultado de la predicción

# Caso de ejemplo para predicción
vivienda_pred2 <- data.frame(

P01A05B = c(2), # ¿Esta vivienda esta conectada a una red de drenajes?
P01A05C = c(1), # ¿Esta vivienda esta conectada a una red de distribución de energía eléctrica?
P01A05A = c(2)  # ¿Esta vivienda esta conectada a una red de distribución de agua?
  
)

# Realizar predicción
result_condicion <- predict(arbol_condicion3, vivienda_pred2, type = "class")
result_condicion  # Mostrar el resultado de la predicción

library(rpart)
library(rpart.plot)

vivienda3 <- read.csv('C:/Users/Andres/Desktop/Proyecto parte 2/vivienda.csv', sep = ',')


# Crear el árbol de decisión para la condición 
arbol_condicion4 <- rpart(POBREZA ~

P01D20A +
P01D20D +
P01D20E,
                         
                         data = vivienda3, method = "class")

# Visualizar el árbol de decisión
rpart.plot(arbol_condicion4, type=2, extra=0, under = TRUE, fallen.leaves = TRUE, box.palette = "BuGn",
           main = "Predicción de la pobreza", cex = 1)

# Caso de ejemplo para predicción
vivienda_pred3 <- data.frame(

P01D20A = c(2), #¿Tiene el hogar servicio de: plan residencial?
P01D20D = c(1), #¿Tiene el hogar servicio de: internet residencial?
P01D20E = c(1)  #¿Tiene el hogar servicio de: T.V. por cable?
  
)

# Realizar predicción
result_condicion <- predict(arbol_condicion4, vivienda_pred3, type = "class")
result_condicion  # Mostrar el resultado de la predicción


#INSTALAR RANDOM FOREST
install.packages("randomForest")

library(randomForest)

hogar <- read.csv('C:/Users/Andres/Desktop/Proyecto parte 2/vivienda.csv', sep=',')


hogar <- hogar[, c("POBREZA","P01H01",
"P01H02",
"P01H03",
"P01H04",
"P01H05",
"P01H06",
"P01H07",
"P01H08",
"P01H09",
"P01H10",
"P01H11",
"P01H12",
"P01H13",
"P01H14",
"P01H15",
"P01H16")]

hogar$POBREZA <- as.factor(hogar$POBREZA)

set.seed(100)
hogar <- hogar[sample(1:nrow(hogar)),]

index <-sample(1:nrow(hogar), 0.8*nrow(hogar))

train <- hogar[index,]
test <- hogar[-index,]

bosque <- randomForest(POBREZA ~

P01H01 +
P01H02 +
P01H03 +
P01H04 +
P01H05 +
P01H06 +
P01H07 +
P01H08 +
P01H09 +
P01H10 +
P01H11 +
P01H12 +
P01H13 +
P01H14 +
P01H15 +
P01H16,

                       data = train,
                       ntree = 200,
                       mtry = 10
                       )
prueba <- predict(bosque, test)

prueba

matriz <- table(test$POBREZA, prueba)

matriz

pre <- sum(diag(matriz)) / sum(matriz)
pre

plot(bosque)



#CASO 1#

dato_nuevo <- data.frame(
P01H01 = 2, # ¿Usted se preocupó de que los alimentos se acabaran en su hogar?
P01H02 = 2, # ¿En su hogar se quedaron sin alimentos?
P01H03 = 2, # ¿En su hogar dejaron de tener una alimentación saludable y balanceada?
P01H04 = 1, # ¿Usted o algún adulto en su hogar tuvo una alimentación basada en poca variedad de alimentos?
P01H05 = 2, # ¿Usted o algún adulto en su hogar dejó de desayunar, almorzar o cenar?
P01H06 = 1, # ¿Usted o algún adulto en su hogar comió menos de lo que debía?
P01H07 = 2, # ¿Usted o algún adulto sintió hambre pero no comió?
P01H08 = 2, # ¿Usted o algún adulto en su hogar comió solo una vez al día o dejó de comer todo un día?
P01H09 = 1, # ¿En su hogar viven personas menores de 18 años? 
P01H10 = 2, # ¿Algún menor de 18 años en su hogar dejó de tener una alimentacion saludable y balanceada? 
P01H11 = 2, # ¿Algún menor de 18 años en su hogar tuvo una alimentacion basada en poca variedad de alimentos?
P01H12 = 2, # ¿Algún menor de 18 años en su hogar dejo de desayunar, almorzar o cenar?
P01H13 = 2, # ¿Algún menor de 18 años en su hogar comió menos de lo que debía?
P01H14 = 2, # ¿Tuvieron que disminuir la cantidad servida en las comidas a algún menor de 18 años en su hogar? 
P01H15 = 2, # ¿Algún menor de 18 años en su hogar sintió hambre pero no comió?
P01H16 = 2  # ¿Algún menor de 18 años en su hogar solo comió una vez al día o dejó de comer todo un día?
)

prediccion <- predict(bosque, dato_nuevo)
prediccion

#CASO 2

dato_nuevo <- data.frame(
P01H01 = 1, # ¿Usted se preocupó de que los alimentos se acabaran en su hogar?
P01H02 = 1, # ¿En su hogar se quedaron sin alimentos?
P01H03 = 1, # ¿En su hogar dejaron de tener una alimentación saludable y balanceada?
P01H04 = 1, # ¿Usted o algún adulto en su hogar tuvo una alimentación basada en poca variedad de alimentos?
P01H05 = 2, # ¿Usted o algún adulto en su hogar dejó de desayunar, almorzar o cenar?
P01H06 = 1, # ¿Usted o algún adulto en su hogar comió menos de lo que debía?
P01H07 = 2, # ¿Usted o algún adulto sintió hambre pero no comió?
P01H08 = 2, # ¿Usted o algún adulto en su hogar comió solo una vez al día o dejó de comer todo un día?
P01H09 = 1, # ¿En su hogar viven personas menores de 18 años? 
P01H10 = 2, # ¿Algún menor de 18 años en su hogar dejó de tener una alimentacion saludable y balanceada? 
P01H11 = 2, # ¿Algún menor de 18 años en su hogar tuvo una alimentacion basada en poca variedad de alimentos?
P01H12 = 1, # ¿Algún menor de 18 años en su hogar dejo de desayunar, almorzar o cenar?
P01H13 = 1, # ¿Algún menor de 18 años en su hogar comió menos de lo que debía?
P01H14 = 1, # ¿Tuvieron que disminuir la cantidad servida en las comidas a algún menor de 18 años en su hogar? 
P01H15 = 2, # ¿Algún menor de 18 años en su hogar sintió hambre pero no comió?
P01H16 = 2  # ¿Algún menor de 18 años en su hogar solo comió una vez al día o dejó de comer todo un día?
)

prediccion <- predict(bosque, dato_nuevo)
prediccion

library(randomForest)

hogar <- read.csv('C:/Users/Andres/Desktop/Proyecto parte 2/vivienda.csv', sep=',')


hogar <- hogar[, c("POBREZA",
"P01D01",
"P01D02",
"P01D03",
"P01D04",
"P01D05")]

hogar$DEPTO <- as.factor(hogar$POBREZA)

set.seed(100)
hogar <- hogar[sample(1:nrow(hogar)),]

index <-sample(1:nrow(hogar), 0.8*nrow(hogar))

train <- hogar[index,]
test <- hogar[-index,]

bosque <- randomForest(DEPTO ~

P01D01 +
P01D02 +
P01D03 +
P01D04 +
P01D05,

                       data = train,
                       ntree = 200,
                       mtry = 10
                       )
prueba <- predict(bosque, test)

prueba

matriz <- table(test$POBREZA, prueba)

matriz

pre <- sum(diag(matriz)) / sum(matriz)
pre

plot(bosque)


#Prediccion 1

dato_nuevo <- data.frame(
P01D01 = 1,
P01D02 = 1,
P01D03 = 1,
P01D04 = 1,
P01D05 = 1
)

prediccion <- predict(bosque, dato_nuevo)
prediccion

#Prediccion 2

dato_nuevo <- data.frame(
P01D01 = 1,
P01D02 = 2,
P01D03 = 2,
P01D04 = 5,
P01D05 = 2
)

prediccion <- predict(bosque, dato_nuevo)
prediccion

```

Para las redes neuronales se pueden consultar en este link https://drive.google.com/file/d/1bu4Wh3Jk7NO7TOWQlJGPYZOW01GdVvr6/view?usp=sharing de google colab o mediante el notebook que se encuentra en el repositorio 

```
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from google.colab import drive
drive.mount("/content/drive")

# Carga del archivo CSV
data = pd.read_csv('/content/drive/My Drive/Mineria/vivienda.csv', encoding='latin1')

# Mezcla de datos y reinicio de índices
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# 0 = Rural y 1 = Urbana
data['AR'] = data['AREA'].apply(lambda x: 1 if x == 1 else 0)

# Selección de características y variable objetivo
X = data[['P02B01', 'P02B02A', 'P02B02B', 'P02B02C', 'P02B02D', 'P02B02E', 'P02B02F', 'P02B02G']]
y = data['AR']

# División del conjunto de datos
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Limpieza y conversión de datos
X_train = X_train.replace(' ', np.nan).dropna().astype('float32')
y_train = y_train[X_train.index].astype('float32')

X_test = X_test.replace(' ', np.nan).dropna().astype('float32')
y_test = y_test[X_test.index].astype('float32')

# Construcción del modelo
model = Sequential()
model.add(Dense(100, input_dim=8, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compilación y entrenamiento del modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=128, validation_data=(X_test, y_test))

# Evaluación del modelo
loss, acc = model.evaluate(X_test, y_test)
print(f"Precisión del modelo: {acc * 100:.2f}%")


# Predicción para nuevos datos
vivienda = np.array([[1, 2, 2, 2, 2, 2, 2, 2]])
p = model.predict(vivienda)

# Interpretación de la predicción
resultado = "Área Rural" if p < 0.5 else "Área Urbana"
print(f"Predicción: {resultado}")

# Predicción para nuevos datos
vivienda = np.array([[3, 2, 2, 2, 2, 2, 2, 2]])
p = model.predict(vivienda)

# Interpretación de la predicción
resultado = "Área Rural" if p < 0.5 else "Área Urbana"
print(f"Predicción: {resultado}")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from google.colab import drive
drive.mount("/content/drive")

# Carga del archivo CSV
data = pd.read_csv('/content/drive/My Drive/Mineria/vivienda.csv', encoding='latin1')

# Mezcla de datos y reinicio de índices
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# 0 = Rural y 1 = Urbana
data['AR'] = data['AREA'].apply(lambda x: 1 if x == 1 else 0)

# Selección de características y variable objetivo
X = data[['P01C01', 'P01C02A', 'P01C02B', 'P01C02C', 'P01C02D', 'P01C02E',
          'P01C02F', 'P01C02G', 'P01C02H', 'P01C02I', 'P01C02J',
          'P01C02K', 'P01C02L', 'P01C02M']]
y = data['AR']

# División del conjunto de datos
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Limpieza y conversión de datos
X_train = X_train.replace(' ', np.nan).dropna().astype('float32')
y_train = y_train[X_train.index].astype('float32')

X_test = X_test.replace(' ', np.nan).dropna().astype('float32')
y_test = y_test[X_test.index].astype('float32')

# Construcción del modelo
model = Sequential()
model.add(Dense(100, input_dim=14, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compilación y entrenamiento del modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=128, validation_data=(X_test, y_test))

# Evaluación del modelo
loss, acc = model.evaluate(X_test, y_test)
print(f"Precisión del modelo: {acc * 100:.2f}%")

# Predicción para nuevos datos
vivienda = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
p = model.predict(vivienda)

# Interpretación de la predicción
resultado = "Área Rural" if p < 0.5 else "Área Urbana"
print(f"Predicción: {resultado}")

```
