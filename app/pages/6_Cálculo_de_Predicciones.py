import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(layout="wide")

logo_url = 'app/imagenes/logo.png'
st.sidebar.image(logo_url)

st.title('Cálculo de Predicciones')

# Cargamos nuestro dataset
data = pd.read_csv('datasets/JSTdatasetR6.csv', sep=';', na_values='', decimal=',')

# Creamos los filtros para interactividad y contenedores de widgets
if 'filtro_year' not in st.session_state:
        st.session_state.filtro_year = None
if 'filtro_country' not in st.session_state:
        st.session_state.filtro_country = None

def cambia_pais():
    st.session_state.filtro_country = True

def cambia_anio():
    st.session_state.filtro_year = True


countries = list(data['country'].drop_duplicates())
years = list(data['year'].drop_duplicates())
country_choice = st.sidebar.selectbox('Filtro por país:', countries, index=None, on_change=cambia_pais)
year_choice = st.sidebar.selectbox('Seleccione el año', range(1946,2021), index=None, on_change=cambia_anio)
vacio1 = st.sidebar.empty()
st.session_state.parar = None


# Funciones auxiliares    

def recargar():
    st.session_state.disable_boton=True
    st.session_state.expand1=True
    st.session_state.expand2=False
    st.session_state.expand3=False
    st.session_state.estado = 0
    st.session_state.filtro_year = None
    st.session_state.filtro_country = None
    country_choice=None
    year_choice=None



def aplica_filtros():
    filtered_df1 = data[(data['country']==country_choice) & (data['year']==year_choice-2)]
    filtered_df2 = data[(data['country']==country_choice) & (data['year']==year_choice)]
    st.session_state.country1 = str(country_choice)
    st.session_state.iso1 = filtered_df1['iso'].iloc[0]
    st.session_state.year1 = year_choice-2
    st.session_state.ltrate1=filtered_df1['ltrate'].iloc[0]
    st.session_state.tloans1=filtered_df1['tloans'].iloc[0]
    st.session_state.thh1=filtered_df1['thh'].iloc[0]
    st.session_state.tmort1=filtered_df1['tmort'].iloc[0]
    st.session_state.stir1=filtered_df1['stir'].iloc[0]
    st.session_state.ca1=filtered_df1['ca'].iloc[0]
    st.session_state.debtgdp1=filtered_df1['debtgdp'].iloc[0]
    st.session_state.cpi1=filtered_df1['cpi'].iloc[0]
    st.session_state.gdp1=filtered_df1['gdp'].iloc[0]
    st.session_state.hpnom1=filtered_df1['hpnom'].iloc[0]
    st.session_state.iy1=filtered_df1['iy'].iloc[0]
    st.session_state.rconsbarro1=filtered_df1['rconsbarro'].iloc[0]
    st.session_state.narrowm1=filtered_df1['narrowm'].iloc[0]
    st.session_state.country2 = str(country_choice)
    st.session_state.year2 = year_choice
    st.session_state.ltrate2=filtered_df2['ltrate'].iloc[0]
    st.session_state.tloans2=filtered_df2['tloans'].iloc[0]
    st.session_state.thh2=filtered_df2['thh'].iloc[0]
    st.session_state.tmort2=filtered_df2['tmort'].iloc[0]
    st.session_state.stir2=filtered_df2['stir'].iloc[0]
    st.session_state.ca2=filtered_df2['ca'].iloc[0]
    st.session_state.debtgdp2=filtered_df2['debtgdp'].iloc[0]
    st.session_state.cpi2=filtered_df2['cpi'].iloc[0]
    st.session_state.gdp2=filtered_df2['gdp'].iloc[0]
    st.session_state.hpnom2=filtered_df2['hpnom'].iloc[0]
    st.session_state.iy2=filtered_df2['iy'].iloc[0]
    st.session_state.rconsbarro2=filtered_df2['rconsbarro'].iloc[0]
    st.session_state.narrowm2=filtered_df2['narrowm'].iloc[0]
    if st.session_state.parar:
            vacio1.button('Reiniciar', on_click=recargar, type="primary")
    elif st.session_state.estado == 0:
        vacio1.button('Calcular Atributos', on_click=calcular_atributos, type="primary")
        st.session_state.estado = 1
    elif st.session_state.estado == 1:
        vacio1.button('Calcular Predicción', on_click=calcular_prediccion, type="primary")
        st.session_state.estado = 2
    else:
        vacio1.button('Reiniciar', on_click=recargar, type="primary")


def calcular_atributos():
    st.session_state.expand1=False
    st.session_state.expand2=True
    vacio1.empty()
    data_aux = pd.read_csv('datasets/finales/data_final.csv', sep=';', na_values='', decimal=',')
    data_aux = data_aux[(data_aux['iso']==st.session_state.iso1) & (data_aux['year']==int(st.session_state.year2))]
    if len(data_aux) > 0:
        st.session_state.iso=data_aux['iso'].iloc[0]
        st.session_state.inv_pib_dif_abs2=data_aux['inv_pib_dif_abs2'].iloc[0]
        st.session_state.year=data_aux['year'].iloc[0]
        st.session_state.ca_pib_dif_abs2=data_aux['ca_pib_dif_abs2'].iloc[0]
        st.session_state.pre_crisis=data_aux['crisisJST'].iloc[0]
        st.session_state.tloans_pib_dif_abs2=data_aux['tloans_pib_dif_abs2'].iloc[0]
        st.session_state.cpi_dif_por2=data_aux['cpi_dif_por2'].iloc[0]
        st.session_state.tdbtserv_pib_dif_abs2=data_aux['tdbtserv_pib_dif_abs2'].iloc[0]
        st.session_state.money_pib_dif_abs2=data_aux['money_pib_dif_abs2'].iloc[0]
        st.session_state.global_loan2=data_aux['global_loan2'].iloc[0]
        st.session_state.rconsbarro_dif_por2=data_aux['rconsbarro_dif_por2'].iloc[0]
        st.session_state.global_drate=data_aux['global_drate'].iloc[0]
        st.session_state.pdebt_pib_dif_abs2=data_aux['pdebt_pib_dif_abs2'].iloc[0]
        st.session_state.drate=data_aux['drate'].iloc[0]
    else:
        st.warning('El año seleccionado pertenece a una situación de crisis o post-crisis', icon="⚠️")
        st.session_state.parar=True
    


def calcular_prediccion():
    st.session_state.expand2=False
    st.session_state.expand3=True
    
    vacio1.empty()
    data_aux = pd.read_csv('datasets/finales/data_final.csv', sep=';', na_values='', decimal=',')
    data_aux = data_aux[(data_aux['iso']==st.session_state.iso1) & (data_aux['year']==int(st.session_state.year2))]

    # Atributos finales
    with open('preprocesadores/atributos_discriminativos_deleted.pkl', 'rb') as f:
        atributos_discriminativos_deleted = pickle.load(f)

    # Estandarizador
    with open('preprocesadores/estandarizador.pkl', 'rb') as f:
        estandarizador = pickle.load(f)
    
    X = data_aux.iloc[:, :11].to_numpy(na_value=np.nan)
    X_std = estandarizador.transform(X)
    columnas = list(data_aux.iloc[:, :11].columns)
    discriminativos = np.array([0 if i not in atributos_discriminativos_deleted else 1 for i in columnas])
    id_no_disc = np.where(discriminativos==0)
    X_std = np.delete(X_std, id_no_disc[0], axis=1)
    y = data_aux['crisisJST'].to_numpy()
    
    # Mejor modelo final
    with open('modelos_entrenados/mejor_modelo_entrenado.pkl', 'rb') as f:
        mejor_modelo_entrenado = pickle.load(f)
    resultado = int(mejor_modelo_entrenado["logisticregression"].predict(X_std)[0])    
    probabilidad = np.max(mejor_modelo_entrenado["logisticregression"].predict_proba(X_std))
    if resultado == 0:
        st.session_state.prediccion = 'No Crisis'
    else:
        st.session_state.prediccion = 'Pre-Crisis'
    if int(y[0]) == 0:
        st.session_state.real = 'No Crisis'
    else:
        st.session_state.real = 'Pre-Crisis'
    st.session_state.probabilidad = probabilidad



if st.session_state.filtro_country is not None and st.session_state.filtro_year is not None and country_choice is not None and year_choice is not None:
    if 'estado' not in st.session_state:
        st.session_state.estado = 0
    if st.session_state.estado == 0:
        st.session_state.expand1=True
    aplica_filtros()

if country_choice is None and year_choice is None:
    st.session_state.disable_boton=True
    st.session_state.expand1=False
    st.session_state.expand2=False
    st.session_state.expand3=False    
    
# Creamos contenedores
expander1 = st.expander("## **:orange[Indicadores Macroeconómicos:]**",expanded=st.session_state.expand1)
expander2 = st.expander("## **:orange[Atributos Diferenciales:]**",expanded=st.session_state.expand2)
expander3 = st.expander("## **:orange[Predicción:]**",expanded=st.session_state.expand3)

with expander1:
    st.write("Año Previo")
    with st.container():
        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
        with col1:
            country1 = st.text_input('País:', key='country1')
            tloans1 = st.number_input('Créd. Sect. Privado', key='tloans1', value=None)
        with col2:
            year1 = st.number_input('Año:', key='year1', value=None, step=1)
            thh1 = st.number_input('Cred. Hogares', key='thh1', value=None)
        with col3:
            ltrate1 = st.number_input('Interés L. P.', key='ltrate1', value=None)
            tmort1 = st.number_input('Cred. Hipotecarios', key='tmort1', value=None)
        with col4:
            stir1 = st.number_input('Interés C. P.', key='stir1', value=None)
            ca1 = st.number_input('Cuenta Corriente', key='ca1', value=None)
        with col5:
            debtgdp1 = st.number_input('Deuda Pública', key='debtgdp1', value=None)
            cpi1 = st.number_input('IPC', key='cpi1', value=None)
        with col6:
            gdp1 = st.number_input('PIB', key='gdp1', value=None)
            hpnom1 = st.number_input('Precio Vivienda', key='hpnom1', value=None)
        with col7:
            iy1 = st.number_input('Inversión', key='iy1', value=None)
            rconsbarro1 = st.number_input('Consumo per Cápita', key='rconsbarro1', value=None)
        with col8:
            narrowm1 = st.number_input('Agregado M1', key='narrowm1', value=None)
    
    st.write("Año Seleccionado")
    with st.container():
        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
        with col1:
            country2 = st.text_input('País:', key='country2')
            tloans2 = st.number_input('Créd. Sect. Privado', key='tloans2', value=None)
        with col2:
            year2 = st.number_input('Año:', key='year2', value=None, step=1)
            thh2 = st.number_input('Cred. Hogares', key='thh2', value=None)
        with col3:
            ltrate2 = st.number_input('Interés L. P.', key='ltrate2', value=None)
            tmort2 = st.number_input('Cred. Hipotecarios', key='tmort2', value=None)
        with col4:
            stir2 = st.number_input('Interés C. P.', key='stir2', value=None)
            ca2 = st.number_input('Cuenta Corriente', key='ca2', value=None)
        with col5:
            debtgdp2 = st.number_input('Deuda Pública', key='debtgdp2', value=None)
            cpi2 = st.number_input('IPC', key='cpi2', value=None)
        with col6:
            gdp2 = st.number_input('PIB', key='gdp2', value=None)
            hpnom2 = st.number_input('Precio Vivienda', key='hpnom2', value=None)
        with col7:
            iy2 = st.number_input('Inversión', key='iy2', value=None)
            rconsbarro2 = st.number_input('Consumo per Cápita', key='rconsbarro2', value=None)
        with col8:
            narrowm2 = st.number_input('Agregado M1', key='narrowm2', value=None)
            
        
with expander2:
    with st.container():
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            iso = st.text_input('ISO:', key='iso')
            inv_pib_dif_abs2 = st.number_input('Dif. Abs. Volumen Inversión', key='inv_pib_dif_abs2', value=None)
            rconsbarro_dif_por2 = st.number_input('Dif. Rel. Consumo Real per Cápita', key='rconsbarro_dif_por2', value=None)
        with col2:
            year = st.number_input('Año:', key='year', value=None, step=1)
            ca_pib_dif_abs2 = st.number_input('Dif. Abs. Cuenta Corriente', key='ca_pib_dif_abs2', value=None)
            pdebt_pib_dif_abs2 = st.number_input('Dif. Abs. Deuda Pública', key='pdebt_pib_dif_abs2', value=None)
        with col3:
            tloans_pib_dif_abs2 = st.number_input('Dif. Abs. Volumen Crédito Privado', key='tloans_pib_dif_abs2', value=None)
            global_drate = st.number_input('Curva Rendimiento Global', key='global_drate', value=None)
            drate = st.number_input('Tipos Interes Largo vs Corto Plazo', key='drate', value=None)
        with col4:
            cpi_dif_por2 = st.number_input('Dif. Rel. IPC', key='cpi_dif_por2', value=None)
            tdbtserv_pib_dif_abs2 = st.number_input('Dif. Abs. Crédito - Tipo Interés', key='tdbtserv_pib_dif_abs2', value=None)
        with col5:
            money_pib_dif_abs2 = st.number_input('Dif. Abs. M3', key='money_pib_dif_abs2', value=None)
            global_loan2 = st.number_input('Aumento Crédito Global', key='global_loan2', value=None)

with expander3:
    prediccion = st.text_input('Predicción:', key='prediccion')
    probabilidad = st.number_input('Probabilidad Asociada a la Predicción:', key='probabilidad')
    real = st.text_input('Valor Real:', key='real')
            
            
     
