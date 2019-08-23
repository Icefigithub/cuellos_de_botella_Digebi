#############################################################################
#Algoritmo correspondiente a árboles de decisión para la ejecución presupuestaria de la Digebi
#############################################################################
# Cargar paquetes.  En caso de no tenerlos en las librerias de R instalarlos (install.packages("nombre del paquete"))
library(tidyverse)
library(rpart)
library(rpart.plot)
library(caret)
library(foreign)
library(randomForest)
library(Hmisc)
library(magrittr)

# Cargar la base de datos de interés. 
# Si se desea descargar la base de datos a su equipo de cómputo 
archivo <- download.file("https://raw.githubusercontent.com/Icefigithub/cuellos_de_botella_Digebi/master/ejecuciondigebi.csv", "ejecdigebi.csv")
ejecdigebi<-read.csv("~\\ejecdigebi.csv")
# Si se desea acceder a la base de datos desde el repositorio en línea
ejecdigebi <- read.csv("https://raw.githubusercontent.com/Icefigithub/cuellos_de_botella_Digebi/master/ejecuciondigebi.csv")
# Filtrar la base para tener en consideración la información de interés

# Seleccionar los años de interés y limpiar la base de datos respecto a los porcentajes de ejecución "DIV/0" (eliminándolos de la base).  Debe de tenerse en cuenta que el porcentaje de ejecución corresponde a la división del monto ejecutado dentro del monto vigente.  En caso de que ambos fuesen cero, entonces la división da como resultado infinito.  Estos resultados fueron sustituidos por menos uno

ejecdigebi<-filter (ejecdigebi,year>=2016, year<=2018,ejecucioncuanti!=-1)

# Seleccionar las variables de interés.   Debe de tenerse en cuenta que inicialmente se utilizó la variable relativa a fuente de financiamiento agregada, sin embargo, al eliminarla, la precisión y la concordancia de las predicciones del árbol de decisión estimado mejoraron
ejecdigebimuni<-select(ejecdigebi,  ejecucioncuali, grupo, municipio, programa) 

# Definir set de entrenamiento (80% de los registros de la base de datos)
set.seed(1234)
ejecdigebimuni_entrenamiento<-sample_frac(ejecdigebimuni, 0.8)

#definir set de prueba (20% de los registros de la base de datos)

ejecdigebimuni_prueba<-setdiff(ejecdigebimuni, ejecdigebimuni_entrenamiento)

# Para buscar el árbol óptimo (sin sub o sobre ajuste) se realizó una búsqueda recursiva de los hiper parámetros

# Inicialmente se utiliza una búsqueda recursiva mediante un grid search
hyper_grid <- expand.grid(minsplit = seq(2, 12, 1), maxdepth = seq(2, 12 , 1)) 
nrow(hyper_grid) # corresponde a 121 modelos

# Primero crear la lista de 121 modelos
modelos <- list()
for (i in 1:nrow(hyper_grid)) {
  
  # Luego obtener los hyper parámetros correspondientes a minsplit, maxdepth para cada modelo (modelo[i])
  minsplit <- hyper_grid$minsplit[i]
  maxdepth <- hyper_grid$maxdepth[i]
  
  # Entrenar cada uno de los 121 modelos con su respectivo minsplit y maxdepth y almacenarlos en la lista creada anteriormente
  modelos[[i]] <- rpart(formula= ejecucioncuali ~ ., data  = ejecdigebimuni_entrenamiento, method =  "class", control = list(minsplit = minsplit, maxdepth = maxdepth))}

# Para cada uno de los 121 modelos crear una función para obtener los valores mínimos de cp y sus errores de validación cruzada

# Función para obtener el valor óptimo de cp
cpoptimoi <- function(x) {
  min    <- which.min(x$cptable[, "xerror"])
  cp <- x$cptable[min, "CP"] 
}

# Función para obtener el error mínimo
errormin <- function(x) {
  min    <- which.min(x$cptable[, "xerror"])
  xerror <- x$cptable[min, "xerror"] 
}

# Obtener el minsplit y maxdepth vinculados al top 5 de cp y errores de validación cruzada mínimos
hyper_grid %>%
  mutate(
    cp    = purrr::map_dbl(modelos, cpoptimoi),
    error = purrr::map_dbl(modelos, errormin)
  ) %>%
  arrange(error) %>%
  top_n(-5, wt = error)

# Con los hiper parámetros obtenidos entrenar el árbol óptimo
arboloptimo <- rpart(
  formula = ejecucioncuali ~ .,  data    = ejecdigebimuni_entrenamiento, method  = "class", control = list(minsplit = 4, maxdepth = 8, cp = 0.01))

# Obtener la precisión y concordancia del árbol óptimo entrenado
prediccionoptima <- predict(arboloptimo, newdata = ejecdigebimuni_prueba, type = "class")
confusionMatrix(prediccionoptima, ejecdigebimuni_prueba[["ejecucioncuali"]])

# Determinar la importancia de las variables para predecir altas o bajas ejecuciones en la Digebi
importancia<-varImp(arboloptimo)

# Graficar el árbol óptimo
rpart.plot(arboloptimo, extra= 104)

################################################################################
#Algoritmo correspondiente a random forest para la ejecución presupuestaria de la Digebi
################################################################################

# Con la base de datos cargada anteriormente y la definición de datos de entrenamiento y prueba ya establecidos hacer una búsqueda recursiva del random forest más afinado

# Inicialmente establecer un random forest de línea base, para lo cual se establecen los parámetros siguientes
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
seed<-1234
set.seed(seed)
numvar<-ejecdigebimuni_entrenamiento[,2:4]
mtry <- sqrt(ncol(numvar))
tunegrid <- expand.grid(.mtry=mtry)
metric<-"Accuracy"
# Entrenar el random forests inicial
rf_lb <- train(ejecucioncuali~., data=ejecdigebimuni_entrenamiento, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control, na.action=na.roughfix)
print(rf_lb)
# Realizar una búsqueda recursiva del random forest con mayor precisión y concordancia, usando un grid search, luego comparar con el random forest de línea base

# Crear una serie de valores para el parámetro mtry y entrenar un random forest para cada uno de estos valores
afinacion <- expand.grid(.mtry=c(1:9))
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
rf_randomsearch <- train(ejecucioncuali~., data=ejecdigebimuni_entrenamiento, method="rf", metric=metric, tuneGrid=afinacion, trControl=control, na.action=na.roughfix)
print(rf_randomsearch)
plot(rf_randomsearch)
# Realizar una búsqueda recursiva del random forest con mayor precisión y concordancia, usando un random search, luego comparar con el random forest de línea base
afinacion <- expand.grid(.mtry=c(1:3))
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="random")
rf_randomsearch <- train(ejecucioncuali~., data=ejecdigebimuni_entrenamiento, method="rf", metric=metric, tuneGrid=afinacion, trControl=control, na.action=na.roughfix)
print(rf_randomsearch)
plot(rf_randomsearch)

# Realizar una búsqueda recursiva del random forest con mayor precisión y concordancia, usando un algoritmo propio, luego comparar con el random forest de línea base
set.seed(seed)
bestmtry <- tuneRF(numvar, ejecdigebimuni_entrenamiento$ejecucioncuali, stepFactor=1.5, improve=1e-5, ntree=500)
print(bestmtry)

# Se entrena un random forest con un mtry = 1.73 (correspondiente al random forest de línea basal) dado que este presenta mejor concordancia 
rf<-randomForest(ejecucioncuali~.,
                 data=ejecdigebimuni_entrenamiento,
                 importance=TRUE,
                 prOximity=TRUE,
                 ntree=500,
                 mytr= 1.732051,
                 na.action=na.roughfix)

# Del random forest anterior se realizan las predicciones y la matriz de confusión correspondiente
rfprediccionesrf <- predict(rf, ejecdigebimuni_prueba)
confusionMatrix(rfprediccionesrf, ejecdigebimuni_prueba[["ejecucioncuali"]])

# Del random forest entrenado se determina el número de árboles con el menor error OOB
which.min(rf$err.rate[,1])

# Con la determinación del número de árboles se procede a entrenar el random forest 
rfopt<-randomForest(ejecucioncuali~.,
                    data=ejecdigebimuni_entrenamiento,
                    importance=TRUE,
                    prOximity=TRUE,
                    ntree=1,
                    mytr= 1.732051,
                    na.action=na.roughfix)

# Del random forest óptimo se realizan las predicciones y la matriz de confusión correspondiente
rfprediccionesopt <- predict(rfopt, ejecdigebimuni_prueba)
confusionMatrix(rfprediccionesopt, ejecdigebimuni_prueba[["ejecucioncuali"]])

# Determinar la importancia de las variables para predecir altas o bajas ejecuciones en la Digebi
(VI_F=importance(rfopt))
varImpPlot(rfopt,type=2)