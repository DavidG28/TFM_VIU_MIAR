import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

logo_url = 'app/imagenes/logo.png'
st.sidebar.image(logo_url)

st.title('Comprensión de Datos')
tab1, tab2, tab3, tab4, tab5 = st.tabs(['Atributos Raw Dataset', 'Características Raw Dataset', 'Dataset Final', 'Métricas', 'Análisis Exploratorio'])

with tab1:
    st.write('## **:orange[Atributos Raw Dataset:]**')
        # Mostramos la descripción de los atributos:

    st.markdown('''La descripción de los diferentes atributos sería:
    
|     |Nombre de Atributo | Descripción|
| --- |----------- | ---------|
|1|year | Año|
|2|country | País|
|3|iso | Código ISO del país|
|4|ifs | Código numérico del país|
|5|pop | Población|
|6|rgdpmad | PIB real per cápita (PPP, 1990 International Dollar)|
|7|rgdbarro | PIB real per cápita (index, 2005=100)|
|8|rconsbarro | Consumo real per cápita|
|9|gdp | Producto Interior Bruto|
|10|iy | Inversión ponderada por PIB|
|11|cpi | Índice de precios al consumo (IPC)|
|12|ca | Estado de cuenta corriente|
|13|imports | Importaciones|
|14|exports | Exportaciones|
|15|narrowm | Agregado monetario estrecho (M1)|
|16|money | Agregado monetario amplio (M3)|
|17|stir | Tipo de interés a corto plazo|
|18|ltrate | Tipo de interés a largo plazo|
|19|hpnom | Precio de la vivienda|
|20|unemp | Tasa de desempleo|
|21|wage | Nivel salarial|
|22|debtgdp | Deuda pública ponderada por PIB|
|23|revenue | Ingresos públicos|
|24|expenditure | Gastos públicos|
|25|xrusd | Tipo de cambio vs Dólar|
|26|peg | Precio/Beneficio|
|27|peg_strict | Precio/Beneficio estricto|
|28|JSTtrilemmaIV | Cambios tasa base bruta|
|29|tloans | Créditos al sector privado|
|30|tmort | Créditos hipotecarios al sector privado|
|31|thh | Crédito a los hogares|
|32|tbus | Crédito a los negocios|
|33|bdebt | Deuda corporativa|
|34|peg_type | Precio/Beneficio tipo|
|35|peg_base | Precio/Beneficio base|
|36|eq_tr | Rentabilidad total del capital|
|37|housing_tr | Rentabilidad total de la vivienda|
|38|bond_tr | Rentabilidad del bono del estado|
|39|bill_rate | Tarifa de facturación|
|40|rent_ipolated | Rentabilidad de la vivienda interpolado|
|41|housing_capgain_ipolated | Plusvalía de vivienda interpolado|
|42|housing_capgain | Plusvalía de vivienda|
|43|housing_rent_rtn | Retorno rentabilidad alquiler de vivienda|
|44|housing_rent_yd | Rentabilidad alquiler de vivienda|
|45|eq_capgain | Ganancia por acciones|
|46|eq_dp | Ganancia por dividendo|
|47|eq_capgain_interp | Ganancia por acciones interpolado|
|48|eq_tr_interp | Ganancia total por acciones interpolado|
|49|eq_dp_interp | Ganancia por dividendo interpolado|
|50|bond_rate | Tipo bono del estado|
|51|eq_div_rtn | Retorno por dividendo|
|52|capital_tr | Retorno de riqueza|
|53|risky_tr | Retorno por activos de riesgo|
|54|safe_tr | Retorno por activos seguros|
|55|lev | Ratio de capital bancario|
|56|ltd | Ratio préstamos/depósitos bancarios|
|57|noncore | Ratio de financiación no básica bancaria|''')

    atributos = pd.dataFrame(
{'Nombre de Atributo' : ['year', 'country', 'iso', 'ifs', 'pop', 'rgdpmad', 'rgdbarro', 'rconsbarro', 'gdp', 'iy', 'cpi', 'ca', 'imports', 'exports', 'narrowm', 'money', 
                        'stir', 'ltrate', 'hpnom', 'unemp', 'wage' 'debtgdp', 'revenue', 'expenditure', 'xrusd', 'peg', 'peg_strict', 'JSTtrilemmaIV', 'tloans', 'tmort',
                        'thh', 'tbus', 'bdebt', 'peg_type', 'peg_base', 'eq_tr', 'housing_tr', 'bond_tr', 'bill_rate', 'rent_ipolated', 'housing_capgain_ipolated',
                        'housing_capgain', 'housing_rent_rtn', 'housing_rent_yd', 'eq_capgain', 'eq_dp', 'eq_capgain_interp', 'eq_tr_interp', 'eq_dp_interp', 'bond_rate',
                        'eq_div_rtn', 'capital_tr', 'risky_tr', 'safe_tr', 'lev', 'ltd', 'noncore'], 
 'Descripción' : ['Año', 'País', 'Código ISO del país', 'Código numérico del país', 'Población', 'PIB real per cápita (PPP, 1990 International Dollar)',
                  'PIB real per cápita (index, 2005=100)', 'Consumo real per cápita', 'Producto Interior Bruto', 'Inversión ponderada por PIB',
                  'Índice de precios al consumo (IPC)', 'Estado de cuenta corriente', 'Importaciones', 'Exportaciones', 'Agregado monetario estrecho (M1)',
                  'Agregado monetario amplio (M3)', 'Tipo de interés a corto plazo', 'Tipo de interés a largo plazo', 'Precio de la vivienda', 'Tasa de desempleo',
                  'Nivel salarial', 'Deuda pública ponderada por PIB', 'Ingresos públicos', 'Gastos públicos', 'Tipo de cambio vs Dólar', 'Precio/Beneficio',
                  'Precio/Beneficio estricto', 'Cambios tasa base bruta', 'Créditos al sector privado', 'Créditos hipotecarios al sector privado',
                  'Crédito a los hogares', 'Crédito a los negocios', 'Deuda corporativa', 'Precio/Beneficio tipo', 'Precio/Beneficio base',
                  'Rentabilidad total del capital', 'Rentabilidad total de la vivienda', 'Rentabilidad del bono del estado', 'Tarifa de facturación',
                  'Rentabilidad de la vivienda interpolado', 'Plusvalía de vivienda interpolado', 'Plusvalía de vivienda',
                  'Retorno rentabilidad alquiler de vivienda', 'Rentabilidad alquiler de vivienda', 'Ganancia por acciones', 'Ganancia por dividendo',
                  'Ganancia por acciones interpolado', 'Ganancia total por acciones interpolado', 'Ganancia por dividendo interpolado', 'Tipo bono del estado',
                  'Retorno por dividendo', 'Retorno de riqueza', 'Retorno por activos de riesgo', 'Retorno por activos seguros', 'Ratio de capital bancario',
                  'Ratio préstamos/depósitos bancarios', 'Ratio de financiación no básica bancaria']})

 st.dataframe(atributos)
 
with tab2:
    st.write('## **:orange[Características Raw Dataset:]**')
    
    # Cargamos nuestro dataset
    data = pd.read_csv('datasets/JSTdatasetR6.csv', sep=';', na_values='', decimal=',')
    
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
    porcentaje.rename('Porcentaje de Valores Ausentes por Atributo', inplace=True)
    porcentaje.sort_values(ascending=False, inplace=True)
    porcentaje_med = round(np.sum(porcentaje) / porcentaje.shape[0], 2)
    st.write('&emsp;Porcentaje Medio de Valores Ausentes: ', porcentaje_med, '%')
    st.write('&emsp;Valores Ausentes por Atributo:')
    st.dataframe(porcentaje, use_container_width=True, height=140)
    col1, col2, col3 = st.columns(3)
    with col2:
        st.image('app/graficas/comprension_bruto_ausentes_atributo.png')
    st.write('&emsp;Valores Ausentes por Año:')
    col1, col2, col3 = st.columns(3)
    with col2:
        st.image('app/graficas/comprension_bruto_ausentes_anio.png')

with tab3:
    st.write('## **:orange[Dataset Final:]**')
    
    st.write('A partir de los datos en bruto se creará el conjunto de datos de trabajo final. Para ello se seguirán los siguientes pasos:')
    st.markdown('''#### **:orange[Filtrado de datos:]**
    
* Se usará únicamente la información correspondiente a las instancias con un número total de valores ausentes inferior al 30% del total de atributos.
* Se usará únicamente la serie temporal contigua con un número de instancias útiles superior también al 30% del número total de países por año.''')

    # Obtenemos el número de muestras de países por año para el primer criterio
    # (número total de valores ausentes inferior al 30%)

    th_atributos = data.shape[1]*30/100
    datos_agrupados = data.groupby('year').apply(lambda x: (x.isnull().sum(axis=1) > th_atributos).sum())
    
    # Calculamos la serie de años final que cumple tanto el primer criterio como el segundo:
    th_anios = len(np.unique(data["iso"]))*30/100
    ultimo_anio_no_util = datos_agrupados[::-1].gt(th_anios).idxmax()
    
    st.write('En base a este criterio la serie temporal útil quedará reducida a los años', min(np.unique(data[data["year"]>ultimo_anio_no_util]["year"])), ' - ', max(np.unique(data[data["year"]>ultimo_anio_no_util]["year"])), '.')
    
    st.markdown('''#### **:orange[Creación de nuevos atributos:]**
    
<div style="text-align: justify;">

A partir de los datos anteriores se procede a la elaboración de los siguientes nuevos atributos que serán los finalmente usados como base para el proceso de entrenamiento. Se usarán indicadores relativos calculados a partir de los datos macroeconómicos, ya que por lo general ofrecen mayor información que los valores absolutos en sí.
    
A modo de ejemplo: el precio de la vivienda a lo largo del tiempo ha ido creciendo, pero ¿podríamos decir de forma categórica si el precio actual es alto o no sin tomar ningún otro indicador como referencia? En principio sería complicado, pero si por ejemplo calculamos un crecimiento interanual de dicho precio en un 100%, sí podríamos indicar que se ha producido un acusado aumento de los precios.</div>''', unsafe_allow_html=True)
    
    with st.expander('+ :orange[Diferencial relativo del Índice de Precios al Consumo]'):
        st.markdown('''<div style="text-align: justify;">
        
Representa el cambio relativo entre el valor actual del Índice de Precios al Consumo (IPC o Consumer Prices Index CPI en inglés), y su valor anterior (dos años antes en nuestro caso). El motivo de selección de este parámetro se debe a que como hemos visto es frecuente que en situaciones de crisis financiera y situaciones de debilidad económica se produzcan escenarios de hiper-inflación que en muchas ocasiones se unen a escenarios de devaluación de la moneda nacional. Este tipo de escenarios de inflación elevada e hiper-inflación, podrían ser detectados mediante este parámetro ya que presentarían valores elevados del mismo.''', unsafe_allow_html=True)
        
    with st.expander('+ :orange[Diferencial absoluto del Agregado Monetario Amplio (M3) escalado por el Producto Interior Bruto del país]'):
        st.markdown('''<div style="text-align: justify;">
    
En este caso se calculará la diferencia absoluta a dos años de dicho indicador escalado previamente por el Producto Interior Bruto. Según define el Banco Central Europeo (BCE), el Agregado Monetario Amplio integra entre otros la cantidad de moneda en circulación, los depósitos a la vista y depósitos a plazo. Según los expertos se trata de un indicador que también permite anticipar posibles escenarios de presión inflacionista y por tanto de problemas de salud financiera del país.''', unsafe_allow_html=True)
        
    with st.expander('+ :orange[Diferencial relativo del Consumo Real per Cápita]'):
        st.markdown('''<div style="text-align: justify;">
    
De nuevo calcularemos el cambio relativo del indicador a dos años. Este indicador representa el volumen de bienes y servicios consumidos por persona y año y se considera como un buen indicador del nivel de salud de la economía.''', unsafe_allow_html=True)
        
    with st.expander('+ :orange[Diferencial absoluto del volumen de Deuda Pública escalado por el Producto Interior Bruto]'):
        st.markdown('''<div style="text-align: justify;">
    
Como en casos anteriores calcularemos el cambio absoluto del indicador a dos años, indicador que como hemos visto puede ser signo de situaciones financieras adversas dado que en las crisis financieras los países pueden requerir aumentar su nivel de deuda acudiendo a financiación externa.''', unsafe_allow_html=True)
        
    with st.expander('+ :orange[Diferencial absoluto del volumen de Inversión escalado por el Producto Interior Bruto]'):
        st.markdown('''<div style="text-align: justify;">
    
De nuevo diferencial calculado a dos años. Como hemos visto uno de los posibles motivos desencadenantes de situaciones de crisis es la pérdida de confianza de los inversores en el país con el correspondiente descenso del volumen de inversión.''', unsafe_allow_html=True)
        
    with st.expander('+ :orange[Diferencial absoluto del estado de Cuenta Corriente del país escalado por el Producto Interior Bruto]'):
        st.markdown('''<div style="text-align: justify;">
    
Calculado nuevamente a dos años. Indicador que refleja el volumen de transacciones comerciales y de bienes y servicios transfronterizas o cross-border. Como hemos visto, situaciones de bloqueo comercial que podrían culminar en situaciones de crisis, quedarían reflejadas en este indicador.''', unsafe_allow_html=True)
        
    with st.expander('+ :orange[Diferencial absoluto del Volumen de Crédito al sector privado escalado por el Producto Interior Bruto]'):
        st.markdown('''<div style="text-align: justify;">
    
Calculado a dos años. Como hemos visto volúmenes de crédito elevados podrían estar relacionados con la aparición de “burbújas” financieras que podrían desencadenar finalmente situaciones de crisis.''', unsafe_allow_html=True)
        
    with st.expander('+ :orange[Diferencial absoluto del ratio Volumen de Crédito al sector privado - Tipo de Interés a largo plazo]'):
        st.markdown('''<div style="text-align: justify;">
        
Nuevamente calculado a un plazo de dos años. Elevados volúmenes de crédito con expectativas de altos tipos de interés a largo plazo pueden indicar problemas futuros de salud financiera del país.''', unsafe_allow_html=True)
        
    with st.expander('+ :orange[Aumento de Crédito Global]'):
        st.markdown('''<div style="text-align: justify;">
        
Cálculo a dos años. En este caso se compara la evolución del nivel de crédito mundial excluyendo el país bajo estudio.''', unsafe_allow_html=True)
        
    with st.expander('+ :orange[Curva de Rendimiento Global (Global Yield Curve)]'):
        st.markdown('''<div style="text-align: justify;">
        
La curva refleja la variación del rendimiento (tipo de interés), de la deuda pública emitida (bonos del estado), de manera global frente al país bajo estudio. Países en situación de mala salud financiera tendrán mayores problemas de financiación y deberán proporcionar mayores tipos de interés a los inversores para acceder a financiación externa.''', unsafe_allow_html=True)
        
    with st.expander('+ :orange[Ratio Tipos de Interes a Largo Plazo vs Corto Plazo]'):
        st.markdown('''<div style="text-align: justify;">
        
Aumentos en los tipos de interés podrían indicar restricciones de acceso a la financiación privada, lo que podría ser indicativo de problemas de salud financiera.''', unsafe_allow_html=True)
    
with tab4:
    st.write('## **:orange[Métricas:]**')

        # Cargamos nuestro dataset final
    data_final = pd.read_csv('datasets/finales/data_final.csv', sep=';', na_values='', decimal=',')
    
    st.write('#### **:orange[Resultado Final:]**')
    st.dataframe(data_final, height=180, use_container_width=True)
    
    # Mostramos métricas de interés
    st.write('&emsp;Número de Atributos: ', data_final.shape[1] - 1, unsafe_allow_html=True)
    st.write('&emsp;Balance de Clases:&emsp;', 'No Crisis: ', round(data_final[data_final['crisisJST'] == 0].shape[0] / data_final.shape[0] * 100, 2), '%&emsp;Pre-Crisis: ', round(data_final[data_final['crisisJST'] == 1].shape[0] / data_final.shape[0] * 100, 2), '%', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col2:
        st.image('app/graficas/comprension_final_instancias_clase.png')
    
    num_ausentes = data_final.isnull().sum()
    porcentaje = round(num_ausentes / data_final.shape[0] * 100, 2)
    porcentaje.rename('Porcentaje de Valores Ausentes por Atributo', inplace=True)
    porcentaje.sort_values(ascending=False, inplace=True)
    porcentaje_med = round(np.sum(porcentaje) / porcentaje.shape[0], 2)
    st.write('&emsp;Porcentaje Medio de Valores Ausentes: ', porcentaje_med, '%')
    st.write('&emsp;Valores Ausentes por Atributo:')
    st.dataframe(porcentaje, use_container_width=True, height=140)
    col1, col2, col3 = st.columns(3)
    with col2:
        st.image('app/graficas/comprension_final_porcentaje_ausentes_atributo.png')
    st.write('&emsp;Valores Ausentes por Año:')
    col1, col2, col3 = st.columns(3)
    with col2:
        st.image('app/graficas/comprension_final_ausentes_anio.png')
    st.write('&emsp;Valores Ausentes por País:')
    col1, col2, col3 = st.columns(3)
    with col2:
        st.image('app/graficas/comprension_final_ausentes_pais.png')

with tab5:
    st.write('## **:orange[Análisis Exploratorio:]**')
    st.write('#### **:orange[Distribución de valores de atributos por país:]**')
    st.markdown('''<div style="text-align: justify;">
    
Por lo general los rangos de los atributos para los distintos países son bastante parecidos. Destacar el cierto sesgo que introduce la variación del PIB de Japón al inicio de la serie temporal, lo que coincide con el lanzamiento en 1945 por parte de Estados Unidos de las bombas atómicas de Hiroshima y Nagasaki, lo que explica dicha variación abrupta dadas las consecuencias económicas.</div>''', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image('app/graficas/comprension_final_valores_atributos_pais_0.png')
        st.image('app/graficas/comprension_final_valores_atributos_pais_4.png')
        st.image('app/graficas/comprension_final_valores_atributos_pais_8.png')
    with col2:
        st.image('app/graficas/comprension_final_valores_atributos_pais_1.png')
        st.image('app/graficas/comprension_final_valores_atributos_pais_5.png')
        st.image('app/graficas/comprension_final_valores_atributos_pais_9.png')
    with col3:
        st.image('app/graficas/comprension_final_valores_atributos_pais_2.png')
        st.image('app/graficas/comprension_final_valores_atributos_pais_6.png')
        st.image('app/graficas/comprension_final_valores_atributos_pais_10.png')
    with col4:
        st.image('app/graficas/comprension_final_valores_atributos_pais_3.png')
        st.image('app/graficas/comprension_final_valores_atributos_pais_7.png')

    st.write('#### **:orange[Distribución de valores de atributos por clase:]**')
    st.markdown('''<div style="text-align: justify;">
    
+ Se observa que no existe una clara distinción en lo que a distribución de valores de los distintos atributos según la clase se refiere, a excepción de 'tloans_pib_dif_abs2' y 'tdbtserv_pib_dif_abs2' de manera muy débil y principalmente de 'global_loan2' y 'global_drate'.
    
+ Incluso si nos limitamos a revisar el comportamiento de los atributos para las muestras asociadas a la clase 'pre-crisis' y su desviación típica, se observa que las variaciones en el rango histórico y geográfico han sido pequeñas. Ésto indicaría en parte que en el ámbito bajo estudio los indicadores de posible crisis serán bastante estables y concentrados.</div>''', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image('app/graficas/comprension_final_distribucion_atributos_clase_0.png')
        st.image('app/graficas/comprension_final_distribucion_atributos_clase_4.png')
        st.image('app/graficas/comprension_final_distribucion_atributos_clase_8.png')
    with col2:
        st.image('app/graficas/comprension_final_distribucion_atributos_clase_1.png')
        st.image('app/graficas/comprension_final_distribucion_atributos_clase_5.png')
        st.image('app/graficas/comprension_final_distribucion_atributos_clase_9.png')
    with col3:
        st.image('app/graficas/comprension_final_distribucion_atributos_clase_2.png')
        st.image('app/graficas/comprension_final_distribucion_atributos_clase_6.png')
        st.image('app/graficas/comprension_final_distribucion_atributos_clase_10.png')
    with col4:
        st.image('app/graficas/comprension_final_distribucion_atributos_clase_3.png')
        st.image('app/graficas/comprension_final_distribucion_atributos_clase_7.png')

    st.write('#### **:orange[Relaciones bivariable con clase:]**')
    st.markdown('''<div style="text-align: justify;">
    
Usando dos variables, se empiezan a observar ciertas diferenciaciones según la clase como podría ser el caso de la combinación ('pdebt_pib_dif_abs2', 'tdbtserv_pib_dif_abs2'), donde las muestras asociadas a 'pre-crisis' paracen más concentradas en una determinada zona del espacio, aunque siguen con poca diferenciación con respecto a las muestras de 'no crisis':</div>''', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col2:
        st.image('app/graficas/comprension_final_pairplot.png')

    st.markdown('''<div style="text-align: justify;">
    
El análisis de datos se complementará posteriormente durante la etapa de preprocesamiento de datos y más en concreto en la fase de selección de atributos, en la cual se procederá a analizar qué atributos son discriminativos con respecto a la clase, se presentarán sus diagramas de caja-bigotes y se calculará el grado de correlación entre ellos para decidir qué atributos se usan finalmente durante la fase de entrenamiento de modelos.</div>''', unsafe_allow_html=True)
