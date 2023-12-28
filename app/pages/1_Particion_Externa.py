import streamlit as st
import pandas as pd
import numpy as np

logo_url = './imagenes/logo.png'
st.sidebar.image(logo_url)

st.title('Partición de Datos Externa (Hold-Out)')
st.markdown('''* Estamos ante un problema de series temporales, se requiere mantener la coherencia temporal por lo que no se aplicará un shuffle a los datos.
* En nuestro caso asegurar que datos de entrenamiento no se localizan entre los datos de test, no sólo se limita a las instancias individuales en sí. Nuestro objetivo será predecir si en un cierto año y para un determinado país se está ante una posible situación de pre-crisis, por tanto tenemos que asegurar que no existe información relativa a años posteriores en nuestro conjunto de test ni para el país bajo estudio en concreto ni tampoco para otros paises (Hellwig, K.P. (2021). Predicting Fiscal Crises: A Machine Learning Approach).
* Particionaremos el dataset a partir de un año concreto manteniendo los valores más cercanos a una distribución de porcentajes igual al 80% para datos de entrenamiento y 20% para datos de prueba.''')

# Cargamos nuestro dataset
data = pd.read_csv('../datasets/DatasetLimpio.csv', sep=';', na_values='', decimal=',')

# Calculamos el número del registro a partir del cual particionar
pct_train = 0.8

num_paises = len(np.unique(data['Country']))
#st.write('Número Total de Países: ', num_paises)

num_anios = len(np.unique(data['Year']))
#st.write('Número Total de Años: ', num_anios)

anio_particion = np.min(data['Year']) + round(num_anios*pct_train)
st.write('**:orange[Año de Particionado:]** ', anio_particion)

train = data[data['Year'] < anio_particion]
test = data[data['Year'] >= anio_particion]

_, ocurrencias_train = np.unique(train['Crisis'], return_counts=True)
_, ocurrencias_test = np.unique(test['Crisis'], return_counts=True)

etiquetas = ['No Crisis: ', 'Pre-Crisis: ', 'Crisis/Post-Crisis: ']

st.write('**:orange[Muestras de Entrenamiento:]** (', round(train.shape[0] * 100 / data.shape[0], 2), '%) &emsp;', etiquetas[0], ocurrencias_train[0], ' &emsp; ',etiquetas[1], ocurrencias_train[1], ' &emsp; ', etiquetas[2], ocurrencias_train[2], unsafe_allow_html=True)
st.write('**:orange[Muestras de Test:]** (', round(test.shape[0] * 100 / data.shape[0], 2), '%) &emsp;', etiquetas[0], ocurrencias_test[0], ' &emsp; ', etiquetas[1], ocurrencias_test[1], ' &emsp; ', etiquetas[2], ocurrencias_test[2], unsafe_allow_html=True)

st.write('**:orange[Dataset de Entrenamiento:]**')
st.dataframe(train, height=180)
st.write('**:orange[Dataset de Test:]**')
st.dataframe(test, height=180)

