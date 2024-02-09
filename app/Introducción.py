import streamlit as st
import pandas as pd
import numpy as np


st.set_page_config(layout="wide")

logo_url = 'app/imagenes/logo.png'
st.sidebar.image(logo_url)

st.title('Introducción')

tab1, tab2, tab3 = st.tabs(['Objetivo y Definiciones', 'Puntos Importantes', 'Problemáticas'])

with tab1:
    st.markdown('''## **:orange[Objetivo]**
    
<div style="text-align: justify;">

#### La finalidad del presente trabajo se centra en el uso de modelos de Inteligencia Artificial para poder predecir con un máximo de dos años de antelación posibles situaciones o indicadores que con el paso del tiempo  podrían concluir en escenarios de crisis financiera.</div>''', unsafe_allow_html=True)

    st.markdown('''## **:orange[Definiciones]**

<div style="text-align: justify;">

El primer punto a abordar es definir qué entendemos por crisis financiera, algo que aunque pueda parecer sencillo, debido a todos los matices y escenarios que engloba no resulta una tarea simple. Tras realizar un análisis de la literatura especializada existente al respecto, **:orange[la mayoría de expertos parecen coincidir en que la definición más adecuada de dicho concepto sería la recogida en el artículo _“Is the crisis problem growing more severe?” (Bordo et al., 2000)_:]**

#### "Se define crisis financiera como episodios de volatilidad en los mercados financieros marcados por importantes problemas de falta de liquidez e insolvencia entre los distintos actores participantes y/o por falta de intervención oficial/gubernamental para limitar las consecuencias"</div>''', unsafe_allow_html=True)
    
with tab2:
    st.markdown('''## **:orange[Puntos Importantes]**
    
Más allá de la definición del concepto, se deberán tener en cuenta los siguientes puntos:

+ Ámbito geográfico y temporal que se desea contemplar.
+ Abordar ámbitos demasiado amplios podrían llevar a tener problemas para disponer de modelos suficientemente robustos.
+ Diferentes tipos de economías en las que se agrupan los países: economías avanzadas, emergentes, en vías de desarrollo, etc.
+ Diversidad de factores desencadenantes de este tipo de situaciones: normalmente observaremos una conjunción de varios de ellos y no un causante único.
+ Factores cambiantes a lo largo de la historia y entre los distintos tipos de economías: las causas o indicadores de crisis para un determinado periodo temporal o tipo de economía pueden no ser válidas para otros.''')

with tab3:

    st.markdown('''## **:orange[Problemáticas]**
    
Algunos de los problemas que han sido necesario abordar a lo largo del desarrollo del proyecto han sido:

+ Dificultad de acceso a datos fiables y completos.
+ Limitación temporal y geográfica de los mismos.
+ Bajo volumen de datos disponibles.
+ Tratamiento de volúmenes elevados de valores ausentes.
+ Desbalance de clases extremo: por suerte dado que las crisis económicas son eventos infrecuentes, el número de muestras asociadas a la clase minoritaria es extremadamente bajo en el conjunto de datos.

Estos puntos han hecho necesario abordar técnicas de:

+ Imputación de valores ausentes: se han comparado varias estrategias.
+ Balanceo del número de muestras de cara al entrenamiento de algoritmos: SMOTE, sobremuestreo, submuestreo.
+ Tipos de algoritmos a emplear.
+ Creación de funciones de particionado ad-Hoc.
+ Decisión sobre métricas válidas debido al desbalance de datos, etc.''')

