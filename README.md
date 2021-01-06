# Tutorial Machine Learning: Regresión Lineal

## Introducción

Muchos data sets tienen una relación linear entre sus variables. En estos casos, podemos predecir una variable usando un valor conocido. A esto lo conocemos como la **línea de mejor ajuste**. Esta línea la podemos calcular usando la siguiente ecuación: 

![Ecuación regresión linear](https://latex.codecogs.com/gif.latex?y%20%3D%20mx%20&plus;%20b)

Gráficamente se ve de la siguiente manera:

![Regresión linear](https://github.com/XavierCarrera/Tutorial-Machine-Learning-Regresion-Lineal/blob/main/img/correlacion.png?raw=true)

Aquí la *x* es el predictor de la variable porque será usado para precedir a *y*. A *y* la conocemos como la respuesta a la variable. Esta técnica es la regresión linear, la cual es uno de los algoritmos clásicos de machine learning. 

El principio fundamental de la regresión linear es que un cambio en uno o más variables del predictor generarán un cambio en la respuesta de la variable. 

## Regresión Linear y Estadística

En su nucleo, machine learning es estadística inferencial ya que tomamos datos históricos para extrapolar y hacer predicciones de datos futuros. La estadística es de gran utilidad para encontrar la línea de mejor ajuste. Nuestros data sets son una colección de puntos distribuidos en un gráfico de dispersión, en los cuales tenemos promedios de *x*, *y*, sus desviaciones standard y coeficientes de correlación. 

Los promedios son simplemente la suma de *x* y *y* entre el número de elementos, en esta caso representado por *n*. 

Después de calcular los promedios (identificados como x̅ y y̅) podemos encontrar la desviación estándard con la siguiente fórmula: 

![Ecuación desviación estándard](https://wikimedia.org/api/rest_v1/media/math/render/svg/05100833069f1eb35275f27bf59467b30efb7517)

La desviación estándard no es otra cosa más que un forma de entender que tan lejos estan nuestros data points del promedio. Una desviación estándard pequeña significa que los data points están bastante cerca del promedio, mientras que una desviación estándard grande significa mayor dispersión. Por ejemplo:

![Gráfico desviación estándard](https://github.com/XavierCarrera/Tutorial-Machine-Learning-Regresion-Lineal/blob/main/img/distribucion_normal.png?raw=true)

Es importante recordar que la desviación estándard no mide precisión, sino variabilidad. Una desviación estándard grande no es indicador de que tengamos data points equivocados, sino que se encuentran más alejados del promedio.

Teniendo las desviaciones estándard de *x* y *y*, podemos calcular el coeficiente de correlación (identificado como *r*). Para calcularlo, tenemos que convertir *x* y *y* en unidades estándard. Usamos 1 ≤ *i*  ≤ *n* para poner *xi* y *yi* en la siguiente fórmula:

![Formula unidades estándard](https://github.com/XavierCarrera/Tutorial-Machine-Learning-Regresion-Lineal/blob/main/img/correlacion_1.png?raw=true)

Esta fórmula nos permite obtener las desviaciones estándard de *xi* y *yi* sobre la media. 

Hecho esto, podemos aplicar la fórmula para calcular el coeficiente de correlación tomando el promedio del producto:
 
![Formula coeficiente correlación](https://github.com/XavierCarrera/Tutorial-Machine-Learning-Regresion-Lineal/blob/main/img/correlaci%C3%B3n_2.png?raw=true)

Todo lo anterior puede parecer complicado. Lo que es importante recordar es que el resultado de esta fórmula va de -1 a 1 y nos indica que tan linearmente correlacionados están nuestros data points. Cuando *r* esta cerca de 0, existe poca correlación linear. Cuando *r* está cerca de 1 existe una correlación positiva. Al contrario, si *r* está cerca de -1 tenemos una correlación negativa.

Por ejemplo, imagina que queremos analizar una correlación sobre bajar de peso. Supongamos que tenemos el dato de cuantos libros leen un grupo de personas. Posiblemente el coeficiente de correlación será cercano a 0 porque leer libros no tendría porque ser indicador del peso de una persona. Sin embargo, el número de horas de ejercicio podria estar cercano a 1 (entre más ejercicio hace una persona, más baja de peso) y el número de calorias que consume podría estar cercano a -1 (entre más calorias consume una persona, menos baja de peso).

No te preocupes si todo esto parece complicado; solo tienes que recordar lo que cada uno de estos conceptos representa teóricamente. Los cálculos son hechos al final del día por las computadoras. 

Con toda esta información podemos calcular la línea de mejor ajuste. En unidades estándard se vería así: *y̅= rx̅*, donde *r* es el coeficiente de correlación. Esto se traduce a la siguiente ecuación:

![Formula linea mejor ajuste](https://github.com/XavierCarrera/Tutorial-Machine-Learning-Regresion-Lineal/blob/main/img/correlacion_3.png?raw=true)

La línea resultante no puede pasar por todos los data points, pero nos muestra una buena representación de todos estos. Con los valores de *xi* podemos calcular valores *yi* que son desconocidos en un principio. Sin embargo, esto solo es posible si existe una correlación linear (ya sea de 1 o -1). 

## Regresión Linear y Algebra
