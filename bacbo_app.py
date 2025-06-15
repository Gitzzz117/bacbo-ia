import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.title("Predicción de Bac Bo con IA")

# Función para convertir resultados a números
def resultado_a_numero(r):
    if r == 'a':
        return 0
    elif r == 'r':
        return 1
    else:  # empate 'e'
        return 2

# Cargar datos existentes o crear dataframe vacío
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=['resultado'])

# Agregar nuevo resultado
nuevo_resultado = st.text_input("Ingresa resultado (a=azul, r=rojo, e=empate):").lower()

if st.button("Agregar resultado"):
    if nuevo_resultado in ['a', 'r', 'e']:
        st.session_state.data = st.session_state.data.append({'resultado': nuevo_resultado}, ignore_index=True)
        st.success(f"Resultado '{nuevo_resultado}' agregado.")
    else:
        st.error("Por favor ingresa solo 'a', 'r' o 'e'.")

st.write("Resultados acumulados:")
st.write(st.session_state.data)

# Preparar datos para entrenar modelo
if len(st.session_state.data) >= 10:
    # Convertir resultados a números
    st.session_state.data['resultado_num'] = st.session_state.data['resultado'].apply(resultado_a_numero)

    # Crear características: resultados anteriores (lag 1)
    st.session_state.data['prev_result'] = st.session_state.data['resultado_num'].shift(1)
    data_clean = st.session_state.data.dropna()

    X = data_clean[['prev_result']]
    y = data_clean['resultado_num']

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar modelo Random Forest
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Predecir y evaluar
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"Precisión del modelo: {acc:.2f}")

    # Predecir siguiente resultado según último dato ingresado
    ultimo = st.session_state.data['resultado_num'].iloc[-1]
    prediccion = model.predict([[ultimo]])
    colores = {0: 'Azul', 1: 'Rojo', 2: 'Empate'}
    st.write(f"Predicción para el siguiente resultado: {colores[prediccion[0]]}")

else:
    st.write("Agrega al menos 10 resultados para entrenar el modelo.")