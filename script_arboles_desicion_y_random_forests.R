#############################################################################
#Algoritmo correspondiente a árboles de decisión para la ejecución presupuestaria de la Digebi
#############################################################################
# Cargar paquetes.  En caso de no tenerlos en las librerias de R instalarlos (install.packages(“nombre del paquete”))
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
archivo <- download.file ("https://raw.githubusercontent.com/Icefigithub/cuellos_de_botella_Digebi/master/ejecuciondigebi.csv", "ejecdigebi.csv")
ejecdigebi<-read.csv("~\\ejecdigebi.csv")
# Si se desea acceder a la base de datos desde el repositorio en línea
ejecdigebi <- read.csv ("https://raw.githubusercontent.com/Icefigithub/cuellos_de_botella_Digebi/master/ejecuciondigebi.csv")
# Filtrar la base para tener en consideración la información de interés

# Seleccionar los años de interés y limpiar la base de datos respecto a los porcentajes de ejecución “DIV/0” (eliminándolos de la base).  Debe de tenerse en cuenta que el porcentaje de ejecución corresponde a la división del monto ejecutado dentro del monto vigente.  En caso de que ambos fuesen cero, entonces la división da como resultado infinito.  Estos resultados fueron sustituidos por menos uno

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
#Algoritmo correspondiente a árboles de decisión para la ejecución presupuestaria de la Digebi con variables predictoras: programa; actividad u obra; vigente; departamento; municipio; fuente de financiamiento agregada; renglón y económico nivel 4
################################################################################
# Cargar la base de datos de interés. 
# Si se desea descargar la base de datos a su equipo de cómputo 
archivo <- download.file ("https://raw.githubusercontent.com/Icefigithub/cuellos_de_botella_Digebi/master/ejecuciondigebi.csv, "db.csv")
                          db<-read.csv("~\\db.csv")
                          # Si se desea acceder a la base de datos desde el repositorio en línea
                          db <- read.csv ("https://raw.githubusercontent.com/Icefigithub/cuellos_de_botella_Digebi/master/ejecuciondigebi.csv, "db.csv")
# Cargar paquetes.  En caso de no tenerlos en las librerias de R instalarlos (install.packages(“nombre del paquete”))
library(data.table)
library(doParallel)
library(bit64)
library(rpart) #para usar funcion de árbol de decisión
library(caret) #para hacer matrices de confusión
library(randomForest)
library(dummies)
library(RColorBrewer)
library(e1071)
library(ggplot2)
# Inicializar el generador de números aleatorios 
set.seed(4) 
# Generar una base de datos por cada año
db16<-db[db$Ano==2016,]
db17<-db[db$Ano==2017,]
db18<-db[db$Ano==2018,]

# Definir set de entrenamiento con el 70% de cada una de las bases de datos
porc<-0.7
corte<-sample(nrow(db16),nrow(db16)*porc)
db16_train<-db16[corte,]
corte<-sample(nrow(db17),nrow(db17)*porc)
db17_train<-db17[corte,]
corte<-sample(nrow(db18),nrow(db18)*porc)
db18_train<-db18[corte,]
# Definir set de prueba con el 30% de cada una de las bases de datos
db16_test<-db16[-corte,]
db17_test<-db17[-corte,]
db18_test<-db18[-corte,]
# Entrenar árboles con variables: programa; actividad u obra; vigente; departamento; municipio; fuente de financiamiento agregada; renglón; económico nivel 4
# Año 2016
modelo16a<-rpart(ej~.-Devengado-Ano, data=db16_train, method = "class")
modelo16a<-prune(modelo16a, cp=0.01) # podarlo con un parámetro de complejidad de 0.01
# Obtener la precisión y la concordancia del árbol correspondiente al año 2016
test16a<-predict(modelo16a, db16_test[,names(db16)!="ej"], type = "class") 
confusionMatrix(test16a,as.factor(db16_test$ej))
# Importancia de variables del árbol correspondiente al año 2016
modelo16a$variable.importance
# Año 2017
modelo17a<-rpart(ej~.-Devengado-Ano, data=db17_train, method = "class")
modelo17a<-prune(modelo17a, cp=0.01) # podarlo con un parámetro de complejidad de 0.01
# Obtener la precisión y la concordancia del árbol correspondiente al año 2017
test17a<-predict(modelo17a, db17_test[,names(db17)!="ej"], type = "class") 
confusionMatrix(test17a,as.factor(db17_test$ej))
# Importancia de variables del árbol correspondiente al año 2017
modelo17a$variable.importance
#Año 2018
modelo18a<-rpart(ej~.-Devengado-Ano, data=db18_train, method = "class")
modelo18a<-prune(modelo18a, cp=0.01) # podarlo con un parámetro de complejidad de 0.01
# Obtener la precisión y la concordancia del árbol correspondiente al año 2018
test18a<-predict(modelo18a, db18_test[,names(db18)!="ej"], type = "class") 
confusionMatrix(test18a,as.factor(db18_test$ej))
# Importancia de variables del árbol correspondiente al año 2018
modelo18a$variable.importance
################################################################################
#Algoritmo correspondiente a árboles de decisión para la ejecución presupuestaria de la Digebi con variables predictoras: programa; renglón; vigente y municipio
################################################################################
# Año 2016
modelo16a2<-rpart(ej~.-Devengado-Ano-Codigo_Economico_Nivel_4 -Codigo_Fuente_agregada-Codigo_Departamento-Codigo_Actividad_u_Obra, data=db16_train, method = "class")
modelo16a2<-prune(modelo16a2, cp=0.01) # podarlo con un parámetro de complejidad de 0.01
# Obtener la precisión y la concordancia del árbol correspondiente al año 2016
test16a2<-predict(modelo16a2, db16_test[,names(db16)!="ej"], type = "class") 
confusionMatrix(test16a2,as.factor(db16_test$ej))
# Importancia de variables del árbol correspondiente al año 2016
modelo16a2$variable.importance
# Año 2017
modelo17a2<-rpart(ej~.-Devengado-Ano-Codigo_Economico_Nivel_4 -Codigo_Fuente_agregada-Codigo_Departamento-Codigo_Actividad_u_Obra, data=db17_train, method = "class")
modelo17a2<-prune(modelo17a2, cp=0.01) # podarlo con un parámetro de complejidad de 0.01
# Obtener la precisión y la concordancia del árbol correspondiente al año 2017
test17a2<-predict(modelo17a2, db17_test[,names(db17)!="ej"], type = "class") 
confusionMatrix(test17a2,as.factor(db17_test$ej))
# Importancia de variables del árbol correspondiente al año 2017
modelo17a2$variable.importance
# Año 2018
modelo18a2<-rpart(ej~.-Devengado-Ano-Codigo_Economico_Nivel_4 -Codigo_Fuente_agregada-Codigo_Departamento-Codigo_Actividad_u_Obra, data=db18_train, method = "class")
modelo18a2<-prune(modelo18a2, cp=0.01) # podarlo con un parámetro de complejidad de 0.01
# Obtener la precisión y la concordancia del árbol correspondiente al año 2018
test18a2<-predict(modelo18a2, db18_test[,names(db18)!="ej"], type = "class") 
confusionMatrix(test18a2,as.factor(db18_test$ej))
# Importancia de variables del árbol correspondiente al año 2018
modelo18a2$variable.importance
################################################################################
#Algoritmo correspondiente a bosques aleatorios para la ejecución presupuestaria de la Digebi con variables predictoras: programa; renglón; vigente y municipio
################################################################################
# Determinar el número de árboles óptimo en función de la tasa de error de clasificación
# Año 2016
modelo16rf <- randomForest(as.factor(ej)~. -Devengado-Ano-Codigo_Economico_Nivel_4 -Codigo_Departamento, data = db16_train, importance = TRUE,ntree=100)
plot(modelo16rf) # Gráfico para determinar en qué número de árboles converge el OOB
# Bosque aleatorio con 50 árboles correspondiente al año 2016
modelo16rf <- randomForest(as.factor(ej)~. -Devengado-Ano-Codigo_Economico_Nivel_4 -Codigo_Departamento-Codigo_Actividad_u_Obra, data = db16_train, importance = TRUE,ntree=50)
# Obtener la precisión y la concordancia del bosque correspondiente al año 2016
test16rf<-predict(modelo16rf, db16_test[,names(db16)!="ej"], type = "class")
confusionMatrix(test16rf,as.factor(db16_test$ej))
# Importancia de las variables del bosque correspondiente al año 2016
knitr::kable(importance(modelo16rf), caption = 'forest 2016')
# Año 2017
modelo17rf <- randomForest(as.factor(ej)~. -Devengado-Ano-Codigo_Economico_Nivel_4 -Codigo_Actividad_u_Obra-Codigo_Departamento, data = db17_train, importance = TRUE, ntree=100)
plot(modelo17rf) # Gráfico para determinar en qué número de árboles converge el OOB
# Bosque aleatorio con 50 árboles correspondiente al año 2017
modelo17rf <- randomForest(as.factor(ej)~. -Devengado-Ano-Codigo_Economico_Nivel_4 -Codigo_Departamento-Codigo_Actividad_u_Obra, data = db17_train, importance = TRUE, ntree=50)
# Obtener la precisión y la concordancia del bosque correspondiente al año 2017
test17rf<-predict(modelo17rf, db17_test[,names(db17)!="ej"], type = "class")
confusionMatrix(test17rf,as.factor(db17_test$ej)) 
# Importancia de las variables del bosque correspondiente al año 2017
knitr::kable(importance(modelo17rf), caption = 'forest 2017')
# Año 2018
modelo18rf <- randomForest(as.factor(ej)~. -Devengado-Ano-Codigo_Economico_Nivel_4 -Codigo_Actividad_u_Obra-Codigo_Departamento, data = db18_train, importance = TRUE, ntree=100)
plot(modelo18rf) # Gráfico para determinar en qué número de árboles converge el OOB
# Bosque aleatorio con 50 árboles correspondiente al año 2018
modelo18rf <- randomForest(as.factor(ej)~. -Devengado-Ano-Codigo_Economico_Nivel_4 -Codigo_Departamento-Codigo_Actividad_u_Obra, data = db18_train, importance = TRUE, ntree=50)
# Obtener la precisión y la concordancia del bosque correspondiente al año 2018
test18rf<-predict(modelo18rf, db18_test[,names(db18)!="ej"], type = "class")
confusionMatrix(test18rf,as.factor(db18_test$ej)) 
# Importancia de las variables del bosque correspondiente al año 2018
knitr::kable(importance(modelo18rf), caption = 'forest 2018')
