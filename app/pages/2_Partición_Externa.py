import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

logo_url = './imagenes/logo.png'
st.sidebar.image(logo_url)

st.title('Partición de Datos Externa (Hold-Out)')
st.markdown('''Se va a comenzar realizando la partición de datos externa para separar el conjunto de datos total en los subconjuntos de entrenamiento (train), y prueba (test).
Dado que estamos ante un problema de series temporales, se requiere mantener la coherencia temporal por lo que:

* No se aplicará shuffle a los datos.
* En este caso asegurar que datos de entrenamiento no se localizan entre los datos de test, no sólo se limita a las instancias individuales en sí, sino que dado que nuestro objetivo será predecir si en un cierto año y para un determinado país se está ante una posible situación de pre-crisis, también tenemos que asegurar que no existe información relativa a años posteriores en nuestro conjunto de test ni para el país bajo estudio en concreto ni tampoco para otros paises (*Hellwig, K.P. (2021). Predicting Fiscal Crises: A Machine Learning Approach*).
* Para cumplir lo anterior se particionará el dataset a partir de un año concreto manteniendo los valores más cercanos a una distribución de porcentajes igual al 80% para datos de entrenamiento y 20% para datos de prueba.''')

# Cargamos nuestro dataset
data = pd.read_csv('../datasets/finales/data_final.csv', sep=';', na_values='', decimal=',')

st.write(' Se tendrá que tener en cuenta que las instancias para cada año no son iguales en el dataset final para mantener un porcentaje 80%-20% adecuado:')

st.write('**:orange[Cálculos para Particionado:]**')
pct_train = 0.8

num_paises = len(np.unique(data['iso']))
st.write('&emsp;Número Total de Países: ', num_paises)

num_anios = len(np.unique(data['year']))
st.write('&emsp;Número Total de Años: ', num_anios)

instancias_test = data.shape[0] * (1 - pct_train)

anio_particion = int((data['year'].max() - (instancias_test/num_paises))) - 2
st.write('&emsp;Año de Particionado: ', anio_particion)

train = data[data['year'] < anio_particion]
test = data[data['year'] >= anio_particion]

_, ocurrencias_train = np.unique(train['crisisJST'], return_counts=True)
_, ocurrencias_test = np.unique(test['crisisJST'], return_counts=True)

etiquetas = ['No Crisis: ', 'Pre-Crisis: ']

st.write('**:orange[Muestras de Entrenamiento:]** (', round(train.shape[0] * 100 / data.shape[0], 2), '%) &emsp;', etiquetas[0], ocurrencias_train[0], ' &emsp; ',etiquetas[1], ocurrencias_train[1], unsafe_allow_html=True)

st.write('**:orange[Partición de Entrenamiento:]**')
st.dataframe(train, height=180)
st.write('**:orange[Muestras de Test:]** (', round(test.shape[0] * 100 / data.shape[0], 2), '%) &emsp;', etiquetas[0], ocurrencias_test[0], ' &emsp; ', etiquetas[1], ocurrencias_test[1], unsafe_allow_html=True)
st.write('**:orange[Partición de Test:]**')
st.dataframe(test, height=180)

