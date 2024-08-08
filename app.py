import pandas as pd
import streamlit as st
import joblib

# Carregando o modelo treinado.
model = joblib.load('modelo_diabetes.pkl')

# Carregando uma amostra dos dados.
dataset = pd.read_csv('diabetes.csv')

# Título
st.title("Data App - Predição de Diabetes")

# Subtítulo
st.markdown("Este é um Data App utilizado para exibir a solução de Machine Learning para o problema de predição de diabetes.")

st.sidebar.subheader("Defina os atributos do paciente para predição de diabetes")

# Mapeando dados do usuário para cada atributo
pregnancies = st.sidebar.number_input("Número de Gravidezes", value=dataset["Pregnancies"].mean())
glucose = st.sidebar.number_input("Glicose", value=dataset["Glucose"].mean())
blood_pressure = st.sidebar.number_input("Pressão Sanguínea", value=dataset["BloodPressure"].mean())
skin_thickness = st.sidebar.number_input("Espessura da Pele", value=dataset["SkinThickness"].mean())
insulin = st.sidebar.number_input("Insulina", value=dataset["Insulin"].mean())
bmi = st.sidebar.number_input("Índice de Massa Corporal (BMI)", value=dataset["BMI"].mean())
dpf = st.sidebar.number_input("Histórico de Diabetes na Família (Diabetes Pedigree Function)", value=dataset["DiabetesPedigreeFunction"].mean())
age = st.sidebar.number_input("Idade", value=dataset["Age"].mean())

# Inserindo um botão na tela
btn_predict = st.sidebar.button("Realizar Predição")

# Verifica se o botão foi acionado
if btn_predict:
    data_teste = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [dpf],
        'Age': [age]
    })
    
    # Imprime os dados de teste
    st.write("Dados de entrada para predição:", data_teste)

    # Realiza a predição
    result = model.predict(data_teste)
    
    st.subheader("A previsão de diabetes para o paciente é:")
    result_text = "Diabetes Positivo" if result[0] == 1 else "Diabetes Negativo"
    
    st.write(result_text)
