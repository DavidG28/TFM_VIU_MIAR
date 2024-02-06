import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

logo_url = 'app/imagenes/logo.png'
st.sidebar.image(logo_url)

st.title('Comprensión de Datos')
tab1, tab2 = st.tabs(['Raw Dataset', 'Dataset Final'])

with tab1:
    st.write('## **:orange[Raw Dataset:]**')
    
    # Cargamos nuestro dataset
    data = pd.read_csv('../datasets/JSTdatasetR6.csv', sep=';', na_values='', decimal=',')
    
    # Creamos los filtros para interactividad
    countries = list(data['country'].drop_duplicates())
    years = list(data['year'].drop_duplicates())
    country_choice = st.sidebar.multiselect('Filtro por país:', countries, [])
    year_choice = st.sidebar.slider('Seleccione el rango de años',
                                    min_value=np.min(data['year']),
                                    max_value=np.max(data['year']),
                                    value=[np.min(data['year']),
                                    np.max(data['year'])],
                                    step=1)
        
    if country_choice==[]:
        filtered_df = data[(data['year']>=year_choice[0]) & (data['year']<=year_choice[1])]
    else:
        filtered_df = data[(data['country'].isin(country_choice)) &  (data['year']>=year_choice[0]) & (data['year']<=year_choice[1])]
    
    # Mostramos nuestro dataset
    st.dataframe(filtered_df, height=180, use_container_width=True)
    
    # Mostramos métricas de interés
    st.write('#### **:orange[Métricas:]**')
    st.write('&emsp;Número de Registros Seleccionados: ', filtered_df.shape[0],
             ' de ', data.shape[0],
             ' (', filtered_df['country'].nunique(),
             ' de ', data['country'].nunique(), ' países)')
    st.write('&emsp;Número de Atributos: ', filtered_df.shape[1] - 1)
    st.write('&emsp;Balance de Clases:&emsp;', 'No Crisis: ', round(filtered_df[filtered_df['crisisJST'] == 0].shape[0] / filtered_df.shape[0] * 100, 2), '%&emsp;Crisis: ', round(filtered_df[filtered_df['crisisJST'] == 1].shape[0] / filtered_df.shape[0] * 100, 2), '%', unsafe_allow_html=True)
    
    num_ausentes = filtered_df.isnull().sum()
    porcentaje = round(num_ausentes / filtered_df.shape[0] * 100, 2)
    porcentaje.rename('Porcentaje de Valores Ausentes por Columna', inplace=True)
    porcentaje.sort_values(ascending=False, inplace=True)
    porcentaje_med = round(np.sum(porcentaje) / porcentaje.shape[0], 2)
    st.write('&emsp;Porcentaje Medio de Valores Ausentes: ', porcentaje_med, '%')
    st.write('&emsp;Valores Ausentes por Columna:')
    st.dataframe(porcentaje, use_container_width=True, height=140)

with tab2:
    st.write('## **:orange[Dataset Final:]**')
    
    st.write('A partir de los datos en bruto se creará el conjunto de datos de trabajo final. Para ello se seguirán los siguientes pasos:')
    st.markdown('''#### **:orange[Filtrado de datos:]**
    
* Se usará únicamente la información correspondiente a las instancias con un número total de valores ausentes inferior al 25% del total de atributos.
* Se usará únicamente la serie temporal contigua con un número de instancias útiles superior también al 25% del número total de países por año.''')

    # Obtenemos el número de muestras de países por año para el primer criterio
    # (número total de valores ausentes inferior al 30%)

    th_atributos = data.shape[1]*30/100
    datos_agrupados = data.groupby('year').apply(lambda x: (x.isnull().sum(axis=1) > th_atributos).sum())
    
    # Calculamos la serie de años final que cumple tanto el primer criterio como el segundo:
    th_anios = len(np.unique(data["iso"]))*30/100
    ultimo_anio_no_util = datos_agrupados[::-1].gt(th_anios).idxmax()
    
    st.write('En base a este criterio la serie temporal útil quedará reducida a los años', min(np.unique(data[data["year"]>ultimo_anio_no_util]["year"])), ' - ', max(np.unique(data[data["year"]>ultimo_anio_no_util]["year"])), '.')
    
    st.markdown('''#### **:orange[Creación de nuevos atributos:]**
    
A partir de los datos anteriores se procede a la elaboración de los siguientes nuevos atributos que serán los finalmente usados como base para el proceso de entrenamiento:''')
    
    with st.expander('+ :orange[Diferencial relativo del Índice de Precios al Consumo]'):
        st.markdown('Representa el cambio relativo entre el valor actual del Índice de Precios al Consumo (IPC o Consumer Prices Index CPI en inglés), y su valor anterior (dos años antes en nuestro caso). El motivo de selección de este parámetro se debe a que como hemos visto es frecuente que en situaciones de crisis financiera y situaciones de debilidad económica se produzcan escenarios de hiper-inflación que en muchas ocasiones se unen a escenarios de devaluación de la moneda nacional. Este tipo de escenarios de inflación elevada e hiper-inflación, podrían ser detectados mediante este parámetro ya que presentarían valores elevados del mismo.')
        
    with st.expander('+ :orange[Diferencial absoluto del Agregado Monetario Amplio (M3) escalado por el Producto Interior Bruto del país]'):
        st.markdown('En este caso se calculará la diferencia absoluta a dos años de dicho indicador escalado previamente por el Producto Interior Bruto. Según define el Banco Central Europeo (BCE), el Agregado Monetario Amplio integra entre otros la cantidad de moneda en circulación, los depósitos a la vista y depósitos a plazo. Según los expertos se trata de un indicador que también permite anticipar posibles escenarios de presión inflacionista y por tanto de problemas de salud financiera del país.')
        
    with st.expander('+ :orange[Diferencial relativo del Consumo Real per Cápita]'):
        st.markdown('De nuevo calcularemos el cambio relativo del indicador a dos años. Este indicador representa el volumen de bienes y servicios consumidos por persona y año y se considera como un buen indicador del nivel de salud de la economía.')
        
    with st.expander('+ :orange[Diferencial absoluto del volumen de Deuda Pública escalado por el Producto Interior Bruto]'):
        st.markdown('Como en casos anteriores calcularemos el cambio absoluto del indicador a dos años, indicador que como hemos visto puede ser signo de situaciones financieras adversas dado que en las crisis financieras los países pueden requerir aumentar su nivel de deuda acudiendo a financiación externa.')
        
    with st.expander('+ :orange[Diferencial absoluto del volumen de Inversión escalado por el Producto Interior Bruto]'):
        st.markdown('De nuevo diferencial calculado a dos años. Como hemos visto uno de los posibles motivos desencadenantes de situaciones de crisis es la pérdida de confianza de los inversores en el país con el correspondiente descenso del volumen de inversión.')
        
    with st.expander('+ :orange[Diferencial absoluto del estado de Cuenta Corriente del país escalado por el Producto Interior Bruto]'):
        st.markdown('Calculado nuevamente a dos años. Indicador que refleja el volumen de transacciones comerciales y de bienes y servicios transfronterizas o cross-border. Como hemos visto, situaciones de bloqueo comercial que podrían culminar en situaciones de crisis, quedarían reflejadas en este indicador.')
        
    with st.expander('+ :orange[Diferencial absoluto del Volumen de Crédito al sector privado escalado por el Producto Interior Bruto]'):
        st.markdown('Calculado a dos años. Como hemos visto volúmenes de crédito elevados podrían estar relacionados con la aparición de “burbújas” financieras que podrían desencadenar finalmente situaciones de crisis.')
        
    with st.expander('+ :orange[Diferencial absoluto del ratio Volumen de Crédito al sector privado - Tipo de Interés a largo plazo]'):
        st.markdown('Nuevamente calculado a un plazo de dos años. Elevados volúmenes de crédito con expectativas de altos tipos de interés a largo plazo pueden indicar problemas futuros de salud financiera del país.')
        
    with st.expander('+ :orange[Aumento de Crédito Global]'):
        st.markdown('Cálculo a dos años. En este caso se compara la evolución del nivel de crédito mundial excluyendo el país bajo estudio.')
        
    with st.expander('+ :orange[Curva de Rendimiento Global (Global Yield Curve)]'):
        st.markdown('La curva refleja la variación del rendimiento (tipo de interés), de la deuda pública emitida (bonos del estado), de manera global frente al país bajo estudio. Países en situación de mala salud financiera tendrán mayores problemas de financiación y deberán proporcionar mayores tipos de interés a los inversores para acceder a financiación externa.')
        
    with st.expander('+ :orange[Ratio Tipos de Interes a Largo Plazo vs Corto Plazo]'):
        st.markdown('Aumentos en los tipos de interés podrían indicar restricciones de acceso a la financiación privada, lo que podría ser indicativo de problemas de salud financiera.')
    
    # Cargamos nuestro dataset final
    data_final = pd.read_csv('../datasets/finales/data_final.csv', sep=';', na_values='', decimal=',')
    
    st.write('#### **:orange[Resultado Final:]**')
    st.dataframe(data_final, height=180, use_container_width=True)
    
    # Mostramos métricas de interés
    st.write('#### **:orange[Métricas:]**')
    st.write('&emsp;Número de Atributos: ', data_final.shape[1] - 1, unsafe_allow_html=True)
    st.write('&emsp;Balance de Clases:&emsp;', 'No Crisis: ', round(data_final[data_final['crisisJST'] == 0].shape[0] / data_final.shape[0] * 100, 2), '%&emsp;Pre-Crisis: ', round(data_final[data_final['crisisJST'] == 1].shape[0] / data_final.shape[0] * 100, 2), '%', unsafe_allow_html=True)
    
    num_ausentes = data_final.isnull().sum()
    porcentaje = round(num_ausentes / data_final.shape[0] * 100, 2)
    porcentaje.rename('Porcentaje de Valores Ausentes por Columna', inplace=True)
    porcentaje.sort_values(ascending=False, inplace=True)
    porcentaje_med = round(np.sum(porcentaje) / porcentaje.shape[0], 2)
    st.write('&emsp;Porcentaje Medio de Valores Ausentes: ', porcentaje_med, '%')
    st.write('&emsp;Valores Ausentes por Columna:')
    st.dataframe(porcentaje, use_container_width=True, height=140)
    
    st.markdown('''#### **:orange[Conclusiones Previas:]**

Como puntos a descatar sobre nuestro dataset:
* Existe un acusado desbalance de instancias asociadas a cada clase
* El volumen de valores ausentes era elevado en algunos atributos y hemos tenido que realizar un primer filtrado de datos brutos útiles.
* Debemos trabajar con datos asociados a series temporales.''')
    
        
        
