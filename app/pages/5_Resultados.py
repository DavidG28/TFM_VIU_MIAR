import streamlit as st
import pandas as pd
import numpy as np
import pickle


st.set_page_config(layout="wide")

logo_url = 'app/imagenes/logo.png'
st.sidebar.image(logo_url)

st.title('Resultados')

tab1, tab2, tab3, tab4 = st.tabs(['Mejores Algoritmos', 'Matrices de Confusión', 'Predicciones', 'Interpretabilidad'])

with tab1:
    st.markdown('''## **:orange[Mejores Algoritmos]**
**:orange[Durante el proceso de entrenamiento se han usado un total de sesenta y nueve modelos]**, correspondientes a todas las posibles combinaciones formadas por las:

+ Estrategias de imputación de valores ausentes.
+ Estrategias de balanceo de datos.
+ Algoritmos seleccionados.

Haciendo foco en las estrategias de imputación de valores ausentes, **:orange[los modelos con mejores resultados]** para cada una de ellas han sido:

+ **:orange[Regresión Logística con Random Oversampling para eliminación de valores ausentes.]**
+ **:orange[Perceptrón Multicapa con Random Oversampling para imputación de valores ausentes por valor medio del atributo.]**
+ **:orange[Support Vector Classifier con balance de pesos para imputación de valores ausentes mediante KNN.]**

Las métricas obtenidas en cada caso han sido:''')
    
    with open('modelos_entrenados/tabla_mejores_modelos.pkl', 'rb') as f:
        tabla_mejores_modelos = pickle.load(f)

    st.dataframe(tabla_mejores_modelos)

    st.write('**:orange[En base al valor de la métrica seleccionada]** para evaluar los modelos ("average_precision_score”), **:orange[el mejor modelo final sería Regresión Logística con Random Oversampling y eliminación de valores ausentes]**.')

with tab2:    
    st.markdown('''## **:orange[Matrices de confusión]**
    
Para cada uno de los mejores modelos, las matrices de confusión obtenidas han sido:''')

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image('app/graficas/delete_LOGR_OVER.png')
    with col2:
        st.image('app/graficas/media_MLP_OVER.png')
    with col3:
        st.image('app/graficas/knn_SVM_BAL.png')

    st.markdown('''<div style="text-align: justify;">
    
En ellas se puede ver claramente el **:orange[trade-off entre fiabilidad a la hora de no pasar por alto ningún posible indicador de posible crisis futura (verdadero positivo en este caso), frente al nivel de falsos positivos generados]**.

Como se puede observar, el primer y tercer modelo aciertan todas las muestras de “pre-crisis” de la partición de test, mientras que el segundo falla en ocho de los casos. Sin embargo el segundo modelo tiene un nivel de falsos positivos bastante menor que los otros dos casos.

En este caso coincidiendo con el mayor valor de la métrica seleccionada, se considerará el primer modelo como el mejor de los tres.</div>''', unsafe_allow_html=True)

with tab3:
    st.markdown('''## **:orange[Predicciones]**
<div style="text-align: justify;">

Focalizando en dicho modelo, se puede revisar en qué muestras se han producido los errores (**:red[rojo]**) y aciertos (**:green[verde]**). Los años ausentes corresponden a aquellos asociados a crisis o post-crisis, que fueron eliminados al inicio para evitar la introducción de sesgo):</div>''', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col2:
        st.image('app/graficas/errores_prediccion.png')
    
    st.markdown('''<div style="text-align: justify;">
    
Como se puede observar **:orange[muchos errores se localizan en el año 2020, año de la pandemia de COVID-19]** y por tanto año extremadamente excepcional en muchos aspectos y entre ellos los económicos, por lo que tiene sentido que el modelo pueda detectar patrones de futura crisis. **:orange[Dado que en el conjunto de datos de trabajo no hay información para los años posteriores, todas las muestras correspondientes a 2020 aparecen marcadas como “no crisis”, pero realmente esta catalogación es cuestionable]**, dado que se sabe que a nivel económico los años post-pandemia podrían considerarse como años de crisis económica, **:orange[con lo que la predicción del modelo podría considerarse correcta]**.</div>''', unsafe_allow_html=True)



with tab4:    
    st.markdown('''## **:orange[Interpretabilidad]**
<div style="text-align: justify;">
    
**:orange[Una de las ventajas del modelo de Regresión Logística es su interpretabilidad]**, ya que evaluando los coeficientes asociados a la función logística podemos determinar qué atributos son más o menos importantes y con qué tipo de relación (directa o inversa). 

**:orange[Vamos a revisar si efectivamente los coeficientes asociados a cada uno de los atributos discriminativos]** (ordenados de mayor a menor importancia), **:orange[concuerdan en su tipo de relación con el marco teórico expuesto]**:</div>''', unsafe_allow_html=True)

    with st.expander('+ :orange[Diferencial relativo del Índice de Precios al Consumo (coeficiente=1,55):]'):
        st.markdown('''<div style="text-align: justify;">
        
Como vemos coincide con lo expresado en el apartado de marco teórico acerca de que escenarios de alta/hiper-inflación pueden constituir un indicador de posible crisis financiera.</div>''', unsafe_allow_html=True)

    with st.expander('+ :orange[Diferencial absoluto del estado de Cuenta Corriente del país escalado por el Producto Interior Bruto (coeficiente=-0,86):]'):
        st.markdown('''<div style="text-align: justify;">
        
De igual forma, a peor estado de la cuenta corriente del país, mayor indicador de problemas de salud financiera.</div>''', unsafe_allow_html=True)

    with st.expander('+ :orange[Ratio Tipos de Interes a Largo Plazo vs Corto Plazo (coeficiente=-0,43):]'):
        st.markdown('''<div style="text-align: justify;">
        
El signo negativo representaría una situación en la que los tipos de interés a corto plazo son superiores a los tipos a largo plazo y por tanto podrían indicar problemas económicos actuales.</div>''', unsafe_allow_html=True)

    with st.expander('+ :orange[Aumento de Crédito Global (coeficiente=0,34):]'):
        st.markdown('''<div style="text-align: justify;">
        
Coincidiría con el hecho de que mayores niveles de crédito indican mayor necesidad de endeudamiento y por tanto menor fortaleza económica.</div>''', unsafe_allow_html=True)

    with st.expander('+ :orange[Diferencial absoluto del Volumen de Crédito al sector privado escalado por el Producto Interior Bruto (coeficiente=0,24):]'):
        st.markdown('''<div style="text-align: justify;">
        
Al igual que en el punto anterior, mayores niveles de crédito indican mayor necesidad de endeudamiento y por tanto menor fortaleza económica.</div>''', unsafe_allow_html=True)

    with st.expander('+ :orange[Curva de Rendimiento Global (Global Yield Curve) (coeficiente=-0,15):]'):
        st.markdown('''<div style="text-align: justify;">
        
Según los expertos, una pendiente plana o descendente de la curva de rendimiento es posible indicador de recesión, lo cual coincide con el signo del coeficiente obtenido.</div>''', unsafe_allow_html=True)

    with st.expander('+ :orange[Diferencial absoluto del ratio Volumen de Crédito al sector privado - Tipo de Interés a largo plazo (coeficiente=-0,03):]'):
        st.markdown('''<div style="text-align: justify;">
    
Sería indicativo de reducciones en el volumen de crédito disponible para el sector privado por parte del sector bancario, lo cual coincidiría con un posible indicativo de recesión económica.</div>''', unsafe_allow_html=True)
