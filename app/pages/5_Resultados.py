import streamlit as st
import pandas as pd
import numpy as np
import pickle


st.set_page_config(layout="wide")

logo_url = './imagenes/logo.png'
st.sidebar.image(logo_url)

st.title('Resultados')

tab1, tab2, tab3, tab4 = st.tabs(['Mejores Algoritmos', 'Matrices de Confusión', 'Predicciones', 'Interpretabilidad'])

with tab1:
    st.markdown('''## **:orange[Mejores Algoritmos]**
Durante el proceso de entrenamiento se han usado un total de sesenta y nueve modelos, correspondientes a todas las posibles combinaciones formadas por las:

+ Estrategias de imputación de valores ausentes.
+ Estrategias de balanceo de datos.
+ Algoritmos seleccionados.

Centrándonos en las estrategias de imputación de valores ausentes, los modelos con mejores resultados para cada una de ellas han sido:

+ Regresión Logística con Random Oversampling para eliminación de valores ausentes.
+ Perceptrón Multicapa con Random Oversampling para imputación de valores ausentes por valor medio del atributo.
+ Support Vector Classifier con balance de pesos para imputación de valores ausentes mediante KNN.

Las métricas obtenidas en cada caso han sido:''')
    
    with open('../modelos_entrenados/tabla_mejores_modelos.pkl', 'rb') as f:
        tabla_mejores_modelos = pickle.load(f)

    st.dataframe(tabla_mejores_modelos)

    st.write('Basándonos en el valor de la métrica seleccionada para evaluar los modelos ("average_precision_score”), el mejor modelo final sería Regresión Logística con Random Oversampling y eliminación de valores ausentes.')

with tab2:    
    st.markdown('''## **:orange[Matrices de confusión]**
    
Para cada uno de los mejores modelos, las matrices de confusión obtenidas han sido:''')

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image('./graficas/delete_LOGR_OVER.png')
    with col2:
        st.image('./graficas/media_MLP_OVER.png')
    with col3:
        st.image('./graficas/knn_SVM_BAL.png')

    st.markdown('''En ellas se puede ver claramente el trade-off entre fiabilidad a la hora de no pasar por alto ningún posible indicador de posible crisis futura (verdadero positivo en este caso), frente al nivel de falsos positivos generados.

Como se puede observar, el primer y tercer modelo aciertan todas las muestras de “pre-crisis” de la partición de test, mientras que el segundo falla en ocho de los casos. Sin embargo el segundo modelo tiene un nivel de falsos positivos bastante menor que los otros dos casos.

En este caso coincidiendo con el mayor valor de la métrica seleccionada, consideraremos el primer modelo como el mejor de los tres.''')

with tab3:
    st.markdown('''## **:orange[Predicciones]**
Si nos centramos en dicho modelo, podemos revisar en qué muestras se han producido los errores y aciertos (los años ausentes corresponden a aquellos asociados a crisis o post-crisis, que fueron eliminados al inicio para evitar la introducción de sesgo):''')

    col1, col2, col3 = st.columns(3)
    with col2:
        st.image('./graficas/errores_prediccion.png')
    
    st.markdown('''Como se puede observar muchos errores se localizan en el año 2020, año de la pandemia de COVID-19 y por tanto año extremadamente excepcional en muchos aspectos y entre ellos los económicos, por lo que tiene sentido que el modelo pueda detectar patrones de futura crisis. Dado que en el conjunto de datos de trabajo no hay información para los años posteriores, todas las muestras correspondientes a 2020 aparecen marcadas como “no crisis”, pero realmente esta catalogación es cuestionable, dado que sabemos que a nivel económico los años post-pandemia podrían considerarse como años de crisis económica, con lo que la predicción del modelo podría considerarse correcta.''')



with tab4:    
    st.markdown('''## **:orange[Interpretabilidad]**
    
Una de las ventajas del modelo de Regresión Logística es su interpretabilidad, ya que evaluando los coeficientes asociados a la función logística podemos determinar qué atributos son más o menos importantes y con qué tipo de relación (directa o inversa). 

Vamos a revisar si efectivamente los coeficientes asociados a cada uno de los atributos discriminativos (ordenados de mayor a menor importancia), concuerdan en su tipo de relación con el marco teórico expuesto:''')

    with st.expander('+ :orange[Diferencial relativo del Índice de Precios al Consumo (coeficiente=1,55):]'):
        st.markdown('Como vemos coincide con lo expresado en el apartado de marco teórico acerca de que escenarios de alta/hiper-inflación pueden constituir un indicador de posible crisis financiera.')

    with st.expander('+ :orange[Diferencial absoluto del estado de Cuenta Corriente del país escalado por el Producto Interior Bruto (coeficiente=-0,86):]'):
        st.markdown('De igual forma, a peor estado de la cuenta corriente del país, mayor indicador de problemas de salud financiera.')

    with st.expander('+ :orange[Ratio Tipos de Interes a Largo Plazo vs Corto Plazo (coeficiente=-0,43):]'):
        st.markdown('El signo negativo representaría una situación en la que los tipos de interés a corto plazo son superiores a los tipos a largo plazo y por tanto podrían indicar problemas económicos actuales.')

    with st.expander('+ :orange[Aumento de Crédito Global (coeficiente=0,34):]'):
        st.markdown('Coincidiría con el hecho de que mayores niveles de crédito indican mayor necesidad de endeudamiento y por tanto menor fortaleza económica.')

    with st.expander('+ :orange[Diferencial absoluto del Volumen de Crédito al sector privado escalado por el Producto Interior Bruto (coeficiente=0,24):]'):
        st.markdown('Al igual que en el punto anterior, mayores niveles de crédito indican mayor necesidad de endeudamiento y por tanto menor fortaleza económica.')

    with st.expander('+ :orange[Curva de Rendimiento Global (Global Yield Curve) (coeficiente=-0,15):]'):
        st.markdown('Según los expertos, una pendiente plana o descendente de la curva de rendimiento es posible indicador de recesión, lo cual coincide con el signo del coeficiente obtenido.')

    with st.expander('+ :orange[Diferencial absoluto del ratio Volumen de Crédito al sector privado - Tipo de Interés a largo plazo (coeficiente=-0,03):]'):
        st.markdown('Sería indicativo de reducciones en el volumen de crédito disponible para el sector privado por parte del sector bancario, lo cual coincidiría con un posible indicativo de recesión económica.')