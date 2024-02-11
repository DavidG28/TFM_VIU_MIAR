import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from scipy.stats import kstest
from scipy.stats import ttest_ind, mannwhitneyu
from scipy import stats

st.set_page_config(layout="wide")

logo_url = 'app/imagenes/logo.png'
st.sidebar.image(logo_url)

st.title('Workflow Preprocesamiento de Datos')

tab1, tab2, tab3, tab4 = st.tabs(['Estandarización', 'Imputación de Valores Ausentes', 'Eliminación de Outliers', 'Selección de Atributos'])

# Cargamos nuestro dataset de entrenamiento
data_train = pd.read_csv('particiones/train.csv', sep=';', na_values='', decimal=',')

# Cargamos nuestro dataset de test
data_test = pd.read_csv('particiones/test.csv', sep=';', na_values='', decimal=',')

def separa_X_y(data):
    X = data.iloc[:, :11].to_numpy(na_value=np.nan)
    y = data['crisisJST'].to_numpy()
    
    return X, y

def estandarizar(X_train, X_test):
    standardizer = StandardScaler()
    X_train_std = standardizer.fit_transform(X_train)
    X_test_std = standardizer.transform(X_test)
    
    return X_train_std, X_test_std

def eliminacion_missing_value_ratio(X_train, X_test):

    # Eliminamos los atributos con volumen de valores ausentes > 30%    
    # Presentamos las dimensiones de datos originales
    
    num_ausentes = np.sum(np.isnan(X_train),axis=0)
    porcentaje = np.round(num_ausentes / X_train.shape[0] * 100, 2)
    mask = porcentaje>30
    porcentaje = porcentaje[porcentaje>30]

    if len(porcentaje) == 0:
        minimo = 0
        maximo = 0
    else:
        maximo = np.max(porcentaje)
        minimo = np.min(porcentaje)
   
    return len(porcentaje), maximo, minimo

def elimina_ausentes(X, y, X_test, y_test):
    
    # Eliminación de registros: borraremos todos aquellos registros para los que alguno de los valores de los atributos no esté presente
    X_aux = X.copy()
    mask = np.isnan(X_aux).any(axis=1)
    registros_borrar = np.where(mask==True)

    X_aux = np.delete(X_aux, registros_borrar, axis=0)

    # Tenemos que borrar también las etiquetas
    y_aux = y.copy()
    y_aux = np.delete(y_aux, registros_borrar, axis=0)

    # Tenemos que borrar también las muestras de test
    
    X_test_aux = X_test.copy()
    mask = np.isnan(X_test_aux).any(axis=1)
    registros_borrar = np.where(mask==True)
    X_test_aux = np.delete(X_test_aux, registros_borrar, axis=0)

    y_test_aux = y_test.copy()
    y_test_aux = np.delete(y_test_aux, registros_borrar, axis=0)

    return X_aux, y_aux, X_test_aux, y_test_aux

def imputa_ausentes_media(X_train, X_test):

    # Imputación de valores ausentes por valor medio
    X_train_media = X_train.copy()
    X_test_media = X_test.copy()

    imp = SimpleImputer(strategy='mean') # Utilizamos la estrategia de la media
    X_train_media = imp.fit_transform(X_train_media)
    X_test_media = imp.transform(X_test_media)

    return X_train_media, X_test_media

def imputa_ausentes_knn(X_train, X_test):

    # Imputación de valores ausentes mediante KNN
    X_train_knn = X_train.copy()
    X_test_knn = X_test.copy()

    imp = KNNImputer()
    X_train_knn = imp.fit_transform(X_train_knn)
    X_test_knn = imp.transform(X_test_knn)


    return X_train_knn, X_test_knn

def evalua_atributos_discriminativos(X_train, y_train, X_test, atributos_origen):
    alpha=0.01
    h_norm=np.zeros(X_train.shape[1])

    nocrisis_data= X_train[y_train==0]
    crisis_data= X_train[y_train==1]

    h=np.zeros(X_train.shape[1])
    h_disc=np.zeros(X_train.shape[1])
    
    for i in range(0,X_train.shape[1]):
        _, p_value=kstest(X_train[:,i],'norm')
        if p_value<=alpha:
            h_norm[i]=0
        else:
            h_norm[i]=1

        if h_norm[i]==0:
            _,p_value=mannwhitneyu(crisis_data[:,i], nocrisis_data[:,i])
        else:
            _,p_value=ttest_ind(crisis_data[:,i], nocrisis_data[:,i])
    
        if p_value<=alpha:
            h_disc[i]=1
        else:
            h_disc[i]=0

    id_no_disc = np.where(h_disc==0)

    X_train_disc = X_train.copy()
    X_test_disc = X_test.copy()
    
    X_train_disc = np.delete(X_train_disc, id_no_disc[0], axis=1)
    X_test_disc = np.delete(X_test_disc, id_no_disc[0], axis=1)
    atributos_discriminativos = [atributos_origen[i] for i,_ in enumerate(atributos_origen) if h_disc[i]==1]

    return X_train_disc, X_test_disc, atributos_discriminativos

# Separamos datos X e y
X_train, y_train = separa_X_y(data_train)
X_test, y_test = separa_X_y(data_test)

with tab1:
    st.markdown('''## **:orange[Estandarización de Datos]**
El primer paso del preprocesamiento consiste en la estandarización de los datos asociados a las particiones de entrenamiento y test creadas en la fase anterior:''')

    # Estandarizamos
    X_train_std, X_test_std = estandarizar(X_train, X_test)
    st.write('#### **:orange[Datos de Entrenamiento]**')
    st.write('**:orange[Originales:]**')
    st.dataframe(X_train, use_container_width=True, height=140)
    st.write('**:orange[Estandarizados:]**')
    st.dataframe(X_train_std, use_container_width=True, height=140)

    st.write('#### **:orange[Datos de Test]**')
    st.write('**:orange[Originales:]**')
    st.dataframe(X_test, use_container_width=True, height=140)
    st.write('**:orange[Estandarizados:]**')
    st.dataframe(X_test_std, use_container_width=True, height=140)

with tab2:    
    st.markdown('## **:orange[Tratamiento de Valores Ausentes]**')
    st.markdown('''#### **Eliminación de Atributos por Missing Value Ratio**
<div style="text-align: justify;">

Dentro de este punto, el primer paso será **:orange[eliminar aquellas características que superen un Missing Value Ratio del 30%, dado que si se supera dicho umbral significaría que la información aportada por dicho atributo sería muy escasa]**:</div>''', unsafe_allow_html=True)

    atributos_eliminados, max_porcentaje, min_porcentaje = eliminacion_missing_value_ratio(X_train_std, X_test_std)
    # atributos_eliminados = 53
    # max_porcentaje = 87.08
    # min_porcentaje = 30.38
    st.write('&emsp;Número de Atributos eliminados: ', atributos_eliminados, unsafe_allow_html=True)
    st.write('&emsp;Porcentaje Máximo: ', max_porcentaje, '%', unsafe_allow_html=True)
    st.write('&emsp;Porcentaje Mínimo: ', min_porcentaje, '%', unsafe_allow_html=True)

    st.markdown('''#### **Estrategias de Imputación de Valores Ausentes**
<div style="text-align: justify;">

**:orange[De cara a poder realizar una comparativa]** posterior en base a las métricas de los distintos algoritmos, **:orange[se han planteado tres estrategias de imputación de valores ausentes]**:</div>''', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        X_train_std_deleted, y_train_deleted, X_test_std_deleted, y_test_deleted = elimina_ausentes(X_train_std, y_train, X_test_std, y_test)
        st.write('''##### **:orange[Eliminación de instancias con valores ausentes:]**''', unsafe_allow_html=True)
        st.write('&emsp;Tamaño Original Partición: ', X_train_std.shape, unsafe_allow_html=True)
        st.write('&emsp;Tamaño Final Partición: ', X_train_std_deleted.shape, unsafe_allow_html=True)
    with col2:
        st.write('''##### **:orange[Imputación de valores ausentes con valor medio:]**''')
        st.write('&emsp;Tamaño Original Partición: ', X_train_std.shape, unsafe_allow_html=True)
        st.write('&emsp;Tamaño Final Partición: ', X_train_std.shape, unsafe_allow_html=True)
    with col3:
        st.write('''##### **:orange[Imputación de valores ausentes mediante algoritmo KNN:]**''')
        st.write('&emsp;Tamaño Original Partición: ', X_train_std.shape, unsafe_allow_html=True)
        st.write('&emsp;Tamaño Final Partición: ', X_train_std.shape, unsafe_allow_html=True)

with tab3:
    st.markdown('''## **:orange[Eliminación de Outliers]**
<div style="text-align: justify;">
    
Aunque por lo general como parte del proceso de procesamiento de los datos siempre se suele aplicar una etapa de eliminación de valores atípicos o outliers, **:orange[en el presente caso se ha optado por no aplicarla dado que podría ser contraproducente]** y provocar pérdidas de información o patrones valiosos en lugar de las mejoras esperadas.

Los motivos principales de este comportamiento negativo serían dos:

* Por un lado hay que tener en cuenta que se está trabajando con datos correspondientes a distintos países y por tanto los rangos asociados a los datos macroeconómicos (a pesar de haber usado parámetros estandarizadores), podrían ser diferentes sin que eso supusiera la existencia de outliers. En este caso al menos deberíamos buscar valores atípicos limitando los datos a los correspondientes a cada país en lugar de hacer una búsqueda sobre el conjunto completo de datos disponibles.

* Sin embargo el punto más grave de proceder con la realización de una eliminación de outliers sería el siguiente: como sabemos uno de los posibles indicadores predictivos de **:orange[una posible situación futura de crisis financiera es un escenario de hiper-inflación. Este escenario se traduciría en un valor atípico del indicador macroeconómico del Índice de Precios al Consumo. Si como parte de nuestro proceso de procesamiento de los datos eliminamos estos valores atípicos, estaríamos evitando que el modelo predictivo pudiera detectar este tipo de patrones]** para la predicción de las futuras crisis financieras. Ésto mismos podría ocurrir con el resto de atributos empleados.</div>''', unsafe_allow_html=True)

with tab4:
    st.write('## **:orange[Selección de Atributos]**')

    st.write('#### **Atributos Discriminativos:**')
    st.markdown('''<div style="text-align: justify;">
    
El siguiente proceso a realizar se centra en **:orange[eliminar aquellos atributos que no sean discriminativos con respecto de la clase]**. Para ello se han seguido los siguientes pasos:

+ Para cada atributo se realiza un estudio para **:orange[determinar si sigue o no una distribución normal mediante la prueba de Kolmogorov-Smirnov]**.
+ En caso de **:orange[no seguir una distribución normal, se realiza una comparativa de las medianas]** de los atributos para las muestras asociadas a cada clase **:orange[(prueba U de Mann-Whitney)]**.
+ En caso de seguir una **:orange[distribución normal, se realiza una comparativa de las medias]** de los atributos para las muestras asociadas a cada clase **:orange[(prueba T de Student)]**.</div>''', unsafe_allow_html=True)

    with open('preprocesadores/nuevos_atributos.pkl', 'rb') as f:
        nuevos_atributos = pickle.load(f)

    X_train_std_disc_deleted, X_test_std_disc_deleted, atributos_discriminativos_deleted = evalua_atributos_discriminativos(X_train_std_deleted, y_train_deleted, X_test_std_deleted, nuevos_atributos)

    X_train_std_media, X_test_std_media = imputa_ausentes_media(X_train_std, X_test_std)
    y_train_media = y_train.copy()
    y_test_media = y_test.copy()
    X_train_std_disc_media, X_test_std_disc_media, atributos_discriminativos_media = evalua_atributos_discriminativos(X_train_std_media, y_train_media, X_test_std_media, nuevos_atributos)

    X_train_std_knn, X_test_std_knn = imputa_ausentes_knn(X_train_std, X_test_std)
    y_train_knn = y_train.copy()
    y_test_knn = y_test.copy()

    X_train_std_disc_knn, X_test_std_disc_knn, atributos_discriminativos_knn = evalua_atributos_discriminativos(X_train_std_knn, y_train_knn, X_test_std_knn, nuevos_atributos)

    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write('##### **:orange[Eliminación de instancias con valores ausentes:]**')
            st.write('&emsp;Nº Atributos Discriminativos: ', X_train_std_disc_deleted.shape[1])
            st.image('app/graficas/disc_deleted.png')
            st.write(atributos_discriminativos_deleted)
            
        with col2:
            st.write('''##### **:orange[Valores ausentes con valor medio:]**
    &nbsp;''', unsafe_allow_html=True)
            st.write('&emsp;Nº Atributos Discriminativos: ', X_train_std_disc_media.shape[1])
            st.image('app/graficas/disc_media.png')
            st.write(atributos_discriminativos_media)
            
        with col3:
            st.write('''##### **:orange[Valores ausentes con KNN:]**
    &nbsp;''', unsafe_allow_html=True)
            st.write('&emsp;Nº Atributos Discriminativos: ', X_train_std_disc_knn.shape[1])
            st.image('app/graficas/disc_knn.png')
            st.write(atributos_discriminativos_knn)
    

    st.write('#### **Coeficientes de Correlación:**')
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write('##### **:orange[Eliminación de instancias con valores ausentes:]**')
            st.image('app/graficas/corr_deleted.png')
            st.write('&emsp;&emsp;No hay atributos con alta correlación')
        with col2:
            st.write('''##### **:orange[Valores ausentes con valor medio:]**
    &nbsp;''', unsafe_allow_html=True)
            st.image('app/graficas/corr_media.png')
            st.write('&emsp;&emsp;No hay atributos con alta correlación')
        with col3:
            st.write('''##### **:orange[Valores ausentes con KNN:]**
    &nbsp;''', unsafe_allow_html=True)
            st.image('app/graficas/corr_knn.png')
            st.write('&emsp;&emsp;No hay atributos con alta correlación')
