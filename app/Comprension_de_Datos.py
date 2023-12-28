import streamlit as st
import pandas as pd
import numpy as np

logo_url = './imagenes/logo.png'
st.sidebar.image(logo_url)

st.title('Comprensión de Datos')
st.write('### **:orange[Raw Dataset:]**')

# Cargamos nuestro dataset
data = pd.read_csv('../datasets/DatasetLimpio.csv', sep=';', na_values='', decimal=',')

# Creamos los filtros para interactividad
countries = list(data['Country'].drop_duplicates())
years = list(data['Year'].drop_duplicates())
country_choice = st.sidebar.multiselect('Filtro por país:', countries, [])
year_choice = st.sidebar.slider('Seleccione el rango de años',
                                min_value=np.min(data['Year']),
                                max_value=np.max(data['Year']),
                                value=[np.min(data['Year']),
                                np.max(data['Year'])],
                                step=1)
    
if country_choice==[]:
    filtered_df = data[(data['Year']>=year_choice[0]) & (data['Year']<=year_choice[1])]
else:
    filtered_df = data[(data['Country'].isin(country_choice)) &  (data['Year']>=year_choice[0]) & (data['Year']<=year_choice[1])]

# Mostramos nuestro dataset
st.dataframe(filtered_df, height=180, use_container_width=True)

# Mostramos métricas de interés
st.write('### **:orange[Métricas:]**')
st.write('&emsp;Número de Registros Seleccionados: ', filtered_df.shape[0],
         ' de ', data.shape[0],
         ' (', filtered_df['Country'].nunique(),
         ' de ', data['Country'].nunique(), ' países)')
st.write('&emsp;Número de Atributos: ', filtered_df.shape[1] - 1)
st.write('&emsp;Balance de Clases:&emsp;', 'No Crisis: ', round(filtered_df[filtered_df['Crisis'] == 0].shape[0] / filtered_df.shape[0] * 100, 2), '%&emsp;Pre-Crisis: ', round(filtered_df[filtered_df['Crisis'] == 1].shape[0] / filtered_df.shape[0] * 100, 2), '%&emsp;Crisis/Post-Crisis: ', round(filtered_df[filtered_df['Crisis'] == 2].shape[0] / filtered_df.shape[0] * 100, 2), '%', unsafe_allow_html=True)

num_ausentes = filtered_df.isnull().sum()
porcentaje = round(num_ausentes / filtered_df.shape[0] * 100, 2)
porcentaje.rename('Porcentaje de Valores Ausentes por Columna', inplace=True)
porcentaje.sort_values(ascending=False, inplace=True)
porcentaje_med = round(np.sum(porcentaje) / porcentaje.shape[0], 2)
st.write('&emsp;Porcentaje Medio de Valores Ausentes: ', porcentaje_med, '%')
st.write('&emsp;Valores Ausentes por Columna:')
st.dataframe(porcentaje, use_container_width=True, height=140)

st.markdown('''### **:orange[Conclusiones:]**   
Como puntos a descatar sobre nuestro dataset:

* Estamos ante un caso de clasificación multiclase.   
* Existe un acusado desbalance de instancias asociadas a cada clase.   
* El volumen de valores ausentes es elevado.
* El número de atributos disponible es elevado.
* Debemos trabajar con datos asociados a series temporales.''')