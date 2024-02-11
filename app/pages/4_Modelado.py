import streamlit as st
import pandas as pd
import numpy as np
import pickle


st.set_page_config(layout="wide")

logo_url = 'app/imagenes/logo.png'
st.sidebar.image(logo_url)

st.title('Modelado')

tab1, tab2, tab3, tab4, tab5 = st.tabs(['Algoritmos Seleccionados', 'Particionado Interno', 'Técnicas de Balanceo de Datos', 'Pipeline', 'Métricas de Evaluación'])

with tab1:
    st.markdown('''## **:orange[Algoritmos Selecionados]**
De cara al diseño del modelo predictivo, se han seleccionado los siguientes algoritmos:''')
    
    with st.expander('+ :orange[Regresión Logística (Logistic Regression)]'):
        st.markdown('''<div style="text-align: justify;">
        
También conocida por los términos regresión logit, clasificación por máxima entropía o clasificación log-linear,  su funcionamiento se basa en el uso de la función logística que representa una función de distribución de probabilidad continua, cuyos valores de salida se encuentran en el rango (0,1) (rango abierto ya que los extremos nunca se alcanzan). El modelo no proporciona como salida directamente el valor de la clase, sino un determinado valor de probabilidad. En función del umbral que se establezca, se modificará la clasificación de las muestras entre las clases existentes.</div>''', unsafe_allow_html=True)
        
    with st.expander('+ :orange[Support Vector Classifier (SVC)]'):
        st.markdown('''<div style="text-align: justify;">
        
Basados en la búsqueda del hiperplano óptimo que será aquel que maximice el margen o distancia a las muestras (las más cercanas se denominan vectores soporte), el cual separá o dividirá el espacio en dos regiones (hiperplano positivo y negativo), cada una de las cuales estará asociada a cada una de las clases existentes, es decir, en función de en qué región se encuentre una muestra se le asignará una u otra clase de cara a la predicción.</div>''', unsafe_allow_html=True)
        
    with st.expander('+ :orange[Árboles de Decisión Simples]'):
        st.markdown('''<div style="text-align: justify;">
    
Concretamente se ha usado la implementación de CART (Classification and Regression Trees), presente en “sklearn.tree” que genera árboles binarios en base a los criterios de división seleccionados por el algoritmo para la clasificación de las muestras en las dos clases disponibles. Permite crear fronteras de división más complejas frente a algoritmos como Regresión Logística, pero ésto a su vez puede ser un problema ya que puede provocar situaciones de sobreajuste. Para evitarlas se pueden definir parámetros como profundidad máxima del árbol, número mínimo de muestras por hoja, etc.</div>''', unsafe_allow_html=True)
        
    with st.expander('+ :orange[Random Forest Classifier]'):
        st.markdown('''<div style="text-align: justify;">
        
Emplea mecanismos de bagging para la creación de subconjuntos de datos a partir del conjunto de datos original (boostrapped datasets). Cada uno de estos subconjuntos se clasifica de manera independiente usando un estimador de tipo árbol de decisión denominado “weak learner”, el cuál utiliza también a su vez un subconjunto distinto de atributos. La decisión final acerca de la pertenencia de cada muestra a una clase determinada se realiza promediando las decisiones establecidas por los distintos estimadores, otorgando el mismo peso a todos ellos (majority voting).</div>''', unsafe_allow_html=True)
        
    with st.expander('+ :orange[Adaboost]'):
        st.markdown('''<div style="text-align: justify;">
        
Emplea mecanismos de boosting. Se parte del conjunto de datos original sobre el cual se aplican árboles de decisión de un sólo nivel (stumps), usando en cada caso uno sólo de los atributos como criterio de clasificación. De entre todos los stumps de ese nivel se elige uno en función del índice Gini (el de mayor índice Gini o menor índice Gini Impurity), y se crea un nuevo conjunto de datos del mismo tamaño que el original de manera aleatoria, teniendo en cuenta que las muestras asociadas a los errores cometidos tendrán un peso mayor y por tanto mayor probabilidad de aparecer en el nuevo conjunto de datos. El proceso se repite y finalmente la decisión final acerca de la pertenencia de cada muestra a una clase determinada, se realiza promediando las decisiones establecidas por los distintos stumps, pero en este caso el peso de cada stump en la decisión final es diferente.</div>''', unsafe_allow_html=True)
        
    with st.expander('+ :orange[Redes neuronales (Perceptrón Multicapa)]'):
        st.markdown('''<div style="text-align: justify;">
        
Formadas por neuronas distribuidas en una capa de entrada, una o varias capas ocultas y una capa de salida. Cada una de las neuronas de una capa se conecta con cada una de las neuronas de la capa siguiente (fully connected), con un peso determinado wijk, siendo “i” el número de la neurona en la capa destino, “j” el número de la neurona en la capa origen y “k” el número de capa en la que nos encontramos. Para obtener el valor a la salida de cada neurona, se aplica una función de activación al sumatorio formado por el producto entre los anteriores pesos y los valores de salida de las neuronas anteriores, más un valor denominado sesgo o bias.

El cálculo de los pesos y sesgos se realiza mediante el proceso de retropropagación o backward propagation, cuyo objetivo consiste en minimizar una función de pérdidas que representa las diferencias entre los valores predichos y los reales.</div>''', unsafe_allow_html=True)

    st.markdown('''<div style="text-align: justify;">
    
En todos los casos a excepción de los modelos basados en redes neuronales, se ha hecho uso de GridSearchCV para la obtención de los mejores hiperparámetros de entre todos los proporcionados como entrada. Una vez obtenidos estos hiperparámetros se ha procedido a entrenar el modelo con el conjunto completo de datos de entrenamiento.</div>''', unsafe_allow_html=True)
    
with tab2:
    st.markdown('''## **:orange[Particionado Interno]**
<div style="text-align: justify;">
    
Como ocurría durante la realización del particionado externo (Hold-out), durante la creación de los folds para la realización de cross-validation o de particionado interno en particiones de entrenamiento y validación, **:orange[hay que asegurar que no se produce fuga de datos, es decir, garantizar que en las particiones de entrenamiento no existen datos asociados a instantes posteriores a los contenidos en las particiones de validación. Además dado el gran desbalance de muestras entre clases, hay que garantizar también la existencia de muestras de ambas clases en todas las particiones]**.

Por este motivo **:orange[se ha creado una función customizada para la creación de las distintas particiones, de manera que se asegure por un lado la coherencia temporal (no existencia de fuga de datos), y por otro la existencia de instancias de ambas clases, imitando para ello el comportamiento de "TimeSeriesSplit"]**. Para ello se calcula un reparto equitativo de muestras minoritarias y se van creando las particiones de manera secuencial. A continuación se puede observar gráficamente un ejemplo para la creación de cinco folds:</div>''', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col2:
        st.image('app/imagenes/folds.png')

with tab3:
    st.markdown('''## **:orange[Técnicas de Balanceo de Datos]**
<div style="text-align: justify;">

Dado el desbalance en lo que a número de muestras asociadas a cada clase respecta, **:orange[es necesario aplicar algún tipo de estrategia para conseguir un número equitativo de muestras en cada clase]**. Para ello se han aplicado las siguientes metodologías a cada uno de los algoritmos indicados en el punto anterior:</div>''', unsafe_allow_html=True)
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.write('''<div style="text-align: justify;">
        
**:orange[SMOTE (Syntetic Minority Oversampling Technique):]** se trata de una técnica que se encarga de **crear muestras artificiales de la clase minoritaria** a partir de las muestras existentes en el conjunto de datos que toma como origen.
            
El funcionamiento se puede ver de manera gráfica en la siguiente imagen:</div>''', unsafe_allow_html=True)
    
        with col2:
            st.write('''<div style="text-align: justify;">
            
**:orange[Random Oversampling:]** se trata de nuevo de una técnica de **“sobremuestreo” u “oversampling” de la clase minoritaria**, pero en este caso no se crean muestras artificiales, sino que de la partición original se eligen de forma aleatoria muestras existentes correspondientes a dicha clase que se incluyen repetidas veces en la partición final para conseguir el deseado balance en el número de muestras.</div>''', unsafe_allow_html=True)
    
        with col3:
            st.write('''<div style="text-align: justify;">
            
**:orange[Random Undersampling:]** a diferencia de los dos casos anteriores, esta técnica consiste en un **“submuestreo” o “undersampling” de las muestras asociadas a la clase mayoritaria**, manteniendo inalterado el número de muestras correspondientes a la clase minoritaria. Como consecuencia se reduce enórmemente el número de muestras útiles.</div>''', unsafe_allow_html=True)
        
        with col4:
            st.write('''<div style="text-align: justify;">
            
**:orange[Balance de pesos:]** se ha hecho uso del parámetro “class_weight”, que permite **calibrar los pesos asociados a cada clase, de forma que el algoritmo pondere de manera diferente los errores cometidos con las muestras de cada tipo de clase**, penalizando en mayor medida los errores cometidos con muestras de la clase minoritaria frente a los errores cometidos con muestras de la clase mayoritaria.</div>''', unsafe_allow_html=True)
    
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.image('app/imagenes/smote.png')
    
        with col2:
            st.image('app/imagenes/oversampling.png')
    
        with col3:
            st.image('app/imagenes/undersampling.png')
    
        with col4:
            st.write("")

with tab4:
    st.markdown('''## **:orange[Pipeline]**
<div style="text-align: justify;">
    
Un punto importante a destacar y en el que debemos centrarnos, es el referido a cuándo se debe realizar el proceso de balanceo de los datos en el caso de usar cross-validation. Si dicho proceso se aplica directamente sobre el conjunto de datos de entrenamiento sobre el cual a su vez se aplicará posteriormente una función para el reparto en folds, este proceso no garantiza que en las particiones finales de entrenamiento se mantenga dicho reparto equilibrado de muestras, por lo que no habría sido de utilidad.

**:orange[Para asegurar que se mantiene el buscado reparto equitativo de muestras en las particiones de entrenamiento, el proceso de balanceo debe realizarse directamente sobre la partición de entrenamiento de cada fold]**. Para ello se ha hecho uso de la librería “Imbalanced-learn”, en concreto de la utilidad “make_pipeline” que permite crear un estimador que sirva de entrada al proceso de entrenamiento concatenando para ello transformaciones, muestreos y el algoritmo concreto a usar. En este caso sólo se han concatenado el proceso de balanceo de datos (usando para ello cada una de las estrategias indicadas anteriormente), y el algoritmo de clasificación a usar.</div>''', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col2:
        st.image('app/imagenes/pipeline.png')

with tab5:
    st.markdown('''## **:orange[Métricas de Evaluación]**
<div style="text-align: justify;">
    
**:orange[En casos como el presente]** u otros parecidos (como pueden ser detección de fraude, anomalías, etc.), donde el volumen de muestras de la clase minoritaria es muy bajo comparado con el de muestras de la clase mayoritaria, lo que **:orange[se suele buscar es una métrica que maximice el número de verdaderos positivos, pero manteniendo un equilibrio con el número de falsos positivos]**, los cuales estarían representando falsas alarmas que deberían ser revisadas con los posibles costes asociados que ésto podría suponer.

**:orange[Una métrica que nos permite evaluar dicho equilibrio es “sklearn.metrics.average_precision_score”]**, que calcula una aproximación del área bajo la curva Precisión-Sensibilidad, métrica que en casos de datos desbalanceados suele considerarse más estable que el área bajo la curva ROC, que suele usarse en clasificación binaria de conjuntos de datos balanceados para evaluar la capacidad discriminativa del modelo entre clases en función del umbral de decisión establecido.

**:orange[Puede ocurrir que el nivel de penalización considerado para un falso positivo y un falso negativo sean diferentes]**, es decir, puede ocurrir que un falso negativo sea muy grave, pero no lo sea un falso positivo o falsa alarma, en cuyo caso nos **:orange[convendría usar otras métricas para potenciar minimizar por ejemplo la sensibilidad o recall del modelo para la clase minoritaria o modificar el umbral de decisión]**.

En este caso **:orange[consideraremos ambas situaciones como igualmente perjudiciales y usaremos como parámetro de evaluación para GridSearchCV el citado “average_precision_score”]**, para lo cual crearemos nuestra propia métrica haciendo uso de “sklearn.metrics.make_scorer”.</div>''', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col2:
        st.image('app/graficas/ROC_curve.png')
    with col3:
        st.image('app/graficas/PR_curve.png')
