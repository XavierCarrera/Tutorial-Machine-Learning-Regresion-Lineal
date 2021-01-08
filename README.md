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

### Promedios y Desviación Estándard

Los promedios son simplemente la suma de *x* y *y* entre el número de elementos, en esta caso representado por *n*. 

Después de calcular los promedios (identificados como *x̅* y *y̅*) podemos encontrar la desviación estándard con la siguiente fórmula: 

![Ecuación desviación estándard](https://wikimedia.org/api/rest_v1/media/math/render/svg/05100833069f1eb35275f27bf59467b30efb7517)

La desviación estándard no es otra cosa más que un forma de entender que tan lejos estan nuestros data points del promedio. Una desviación estándard pequeña significa que los data points están bastante cerca del promedio, mientras que una desviación estándard grande significa mayor dispersión. Por ejemplo:

![Gráfico desviación estándard](https://github.com/XavierCarrera/Tutorial-Machine-Learning-Regresion-Lineal/blob/main/img/distribucion_normal.png?raw=true)

Es importante recordar que la desviación estándard no mide precisión, sino variabilidad. Una desviación estándard grande no es indicador de que tengamos data points equivocados, sino que se encuentran más alejados del promedio.

### Coeficiente de Correlación

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

## Regresión Linear y Algebra Linear

La algebra linear es de gran ayuda cuando trabajamos con regresión linear porque podemos desarrollar una intuición sobre como se ve la línea de mejor ajuste. 

En esta parte vamos a hablar sobre otra forma de calcular la línea de mejor ajuste. Sin embargo, puedes elegir si usar un método estadístico o uno algebráico para este propósito. 

La fórmula que hemos visto anteriormente para calcular la línea de mejor ajuste también se le conoce como la línea de regresión de mínimos cuadrados, la cual indica que la suma de los cuadrados de las distancias verticales de cada data point hacía la línea de mejor ajuste es menor en comparación a cualquier otra línea.

Esto nos permite definir una función de error para cualquier línea *y = mx +b* que resulta en la suma de errores cuadráticos. Es decir, la suma del cuadrado de cada distancia vertical entre los puntos y la línea. La ecuación se ve de la siguiente manera:

![Suma de los Errores Cuadráticos](https://github.com/XavierCarrera/Tutorial-Machine-Learning-Regresion-Lineal/blob/main/img/suma_errores_cuadrados.png?raw=true)

Aunque lo anterior puede sonar complicado, solo tienes que recordar que la regresión de mínimos cuadrados es la distancia (o error) entre nuestros data points y nuestra línea. 

Por ejemplo, si tenemos una línea y cuatro puntos distanciados por valores 3, 5, 2 y 2. En esta caso solo tenemos que llevarlos al cuadrado. 

![Suma de errores cuadráticos](https://ds055uzetaobb.cloudfront.net/brioche/uploads/EoggUX0iRm-ssegraph2.png?width=1200)

Es decir, 9, 25, 4 y 4 respectívamente. Al final solo hay que sumarlos todos para que nos de un resultado de 42.

No olvides que esto es otra manera de calcular la línea de mejor ajuste. Suponiendo que tenemos un conjunto de puntos *(x1, y1),(x2,y2),...(xn,yn)*, tendríamos que derivar una formula para obtener la línea de mínimos cuadrados. Una primera aproximación sería poner nuestros puntos en términos de vectores. Es decir, *->x* es igual a:

     [m]
     [b]

Ahora definiriamos una matriz A en *n* × 2. Por 1 ≤ i ≤ n en la fila número *i* contedrá *xi* en la primera columna y 1 en la segunda.

Veamos ahora este vector de A:

    [*x1  1*]
    [*x2  1*]
    [  ...  ]
    [xn   1 ]

El vector *A ->x* contedrá valores que la línea representada por *->x* predecirá por cada valor *x* en nuestro data set. Es decir, calcular *A ->x* es nutrir el valor *x* de cada punto representado por *->x*. 

Como resultado, si definimos un vector *->b* para que esté en el elemento número *i* tendremos como resultado *yi*. De esta forma podemos encontrar la distancia vertical entre cada punto y el valor *y* predecido por restar *->b* a *A -> x*.

Así podemos encontrar la suma de errores cuadráticos al llevar al cuadrado individualmente cada valor dentro de *A -> x - -> b* y añadiéndolos todos. En fórmula la veríamos así:

    = ||*AX* - b||²

Por ejemplo, digamos que tenemos una serie de números (1, 0), (3,4), (2,3). Aquí *m* = 3 (número de datos) y *b* = 2 (promedio).

Empecemos armando la matriz que se vería así:

    [1 1]
    [3 1]
    [2 1]
    
 Podemos predecir los valores de *y* al predecir la linea de cada valor *x* calculando *Ax*:
 
       [1 1]
     = [3 1] [3]
       [2 1] [2]
        
       [1 ⋅ 3 + 1 ⋅ 2]    
     = [3 ⋅ 3 + 1 ⋅ 2]
       [2 ⋅ 3 + 1 ⋅ 2]
    
       [5]
     = [11]
       [8]
       
Ahora incluimos a *b* con los valores y:

      [0]
    = [4]
      [3]
      
Todo junto, podríamos caclular la suma de errores cuadráticos de la siguiente manera:

    = ||*AX* - b||²

      ||[5]    [0]||²
    = ||[11] - [4]||
      ||[8]    [3]||
      
      ||[5]||²
    = ||[7]||
      ||[5]||
      
    = 5² + 7² + 5²
    
    = 99
    
Hay que tomar en cuenta que al minimizar ||*A->x* - *->b*||, también estamos minimizano la distancia entre *A->x* y *->b* dado que minimizar la distancia a un cuadrado positivo minimiza también los valores.

Si has tenido problemas para entender esta última parte, no te preocupes. Repasar un poco de algebra lineal puede ayudarte a desarrollar un instinto para entender como se forman las líneas de mejor ajuste. 

De momento quédate con que esto nos ayudará despues a entender que tan bien nuestro modelo de machine learning predice datos desconocidos.

## Regresión en Alta Dimensionalidad

Aunque hasta el momento hemos trabajado solo con matrices bidimensionales, la realidad es que en la mayoría de los casos nos enfrentaremos a datasets multidimensionales. Sin embargo, la regresión de cuadrados mínimos es muy bueno para generalizar en alta dimensionalidad.

En esta caso, si tenemos variables predictoras [*x1, x2, ...., xp*] y una variable de respuesta *y*, entonces la ecuación linear que produce *y* se vería_

*y = m1x1 + m2x2 + ... + mp xp +b*

Lo que intentamos hacer acá es una fórmula para la línea de mejor ajuste para nuestra regresión lineal en alta dimensionalidad. Pero en este caso, no obtenemos una línea de mejor ajuste sino un hiperplano de mejor ajuste.

Podemos crear una matriz *A* que cuando se multiplica con *->x* resulta en un vector conteniendo el valor predecido de *y* de cada data point:

      [m1]
      [m2]
      [..]
      [mp]
      [b ]
   
Para producir alta dimensionalidad necesitamos solo añadir una columna más a cada variable predictor. En un dataset con *n* data points y variables predictoras *p*:


    [*x11 x12 ... x1p 1]
    [x21  x22 ... x2p 1]
    [ ...          ... ]
    [xn1  xn2 ... xnp 1]
    
De esta forma inicializamos el vector *->b* para contener valores y los valores *y* de cada data point. 
    
Resulta ser que de este punto en la derivación es exáctamente lo mismo que antes. Tenemos un vector *->x* por cada *A->x* esta tan cerca como sea posible de *b*. Esto lo podemos resolver con la siguiente ecuación:

![alta dimensionalidad](asdsad.as)

De esta forma, los elementos en *->x* tendrán el valor de coeficientes del hiperplano de mejor ajuste.

En este punto podemos encontrar el hiperplano de mejor ajuste para cualquier dataset, siempre que haya más datapoints que predictores. Pero también puede pasar que los data points sean muy predecibles pero no en una forma linear. En este caso, solo tenemos que añadir nuevos términos no-lineares a nuestra función y actualizar nuestras formulas. Lo anterior se hace añadiendo exponentes a nuestras variables predictoras, dando como resultado una regresión polinomial. 

Por ejemplo, si tuvieramos una variables predictora *x* y una variable a predecir *y* solo tendríamos que representar *y* para representar un polinomio de segundo grado de *x*. Es decir, en vez de representar una línea de mejor ajuste como:

*y = mx + b*

lo que haríamos es representarlo como un mejor ajuste polinomial:

*y = m1x² + m2x + b

En muchas formas, es lo mismo que crear otra variable predictora. Todo lo que hemos hecho es tomar cada punto en nuestro dataset y añadir otro valor x². Después de esto, podemos calcular el coeficiente como lo hacemos normalmente en una regresión linear de alta dimensionalidad.

## Limitaciones de Regresión Linear

Aunque la regresión linear es una herramienta poderosa de machine learning, solo puede ser utilizada si hay una relación linear obvia. Pero este no es el caso cuando si tenemos una variable a predecir que no obedece a una variable predictora.

### Outliers

Otro factor que puede limitar el uso de regresión linear son los outliers. Estos son elementos en nuestro dataset que son muy distintos al resto de nuestros data points. Un outlier bastante alejado de nuestro data set podría afectar la línea de mejor ajuste. 

![]()

Por lo general, los outliers son simplemente excluidos del resto de elementos que se encuentran distantes a ellos. Un método más complicado es modelar los datos y luego excluir data points que contribuyan de más a calcular el error. 

Sin embargo, no debemos olvidar que estos data points podrían tener información valiosa. Hay que tener cuidado si decidimos excluirlos. 

### Multicolinealidad

A multicolinealidad la conocemos cuando hay demasiadas variables que parecen estar fuertemente relacionadas. La mayor preocupación en este punto es que la multicolinealidad puede resultar en múltiples ecuaciones para encontrar el mejor ajuste y que su regresión de cuadrados mínimos produce resultados inestables. Es decir, podríamos tener problemas al extrapolar en situaciones en las que no hay tanta multicolinealidad.

Además, se vuelve más difícil medir el impacto de cada variable predictora en el valor a predecir. También podemos caer en una situación en la que con tantos valores predictores, nuestra regresión linear se especialice en un data set pero tenga problemas para predecir valores nuevos. Hay que recordar que el nombre del juego en este caso es **generalizar**.

### Heterocedasticidad

Otro caso que limita el uso de regresión linear es la popiedad de heterocedasticidad, la cual produce grandes diferencias en las desviaciones estándard. Esto puede causar que algunos data points tengán un peso desproporcionado al calcular su importancia en las regresiones.

![]()

### Sobreajuste

El sobreajuste es probablemente el problema más común cuando utilizamos regresión linear. En estos casos los errores entre los data points tienen un gran efecto en la ecuación de mejor ajuste. Como mencionamos antes, un modelo sobreajustado se desempeña bien en los datos usados para entrenar un modelo. Sin embargo, no será tan bueno para entrenarse con datos nuevos. Variables que no explican el fenómeno en cuestión puden tener demasiado peso, lo que puede dar resultados inesperados.

## Alternativas a la Regresión Linear

