from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import sqlite3
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
import joblib
import subprocess


def eliminar_outlier(df, colname):
    media = df[colname].mean()  # media
    std = df[colname].std()     # desvio padrao

    corte_min = media - 2.5 * std
    corte_max = media + 2.5 * std

    list_index_outlier = []
    for i in range(len(df)):
        if ((df.iloc[i][colname] < corte_min) or (df.iloc[i][colname] > corte_max)):
            list_index_outlier.append(i)

    df = df.drop(df.index[list_index_outlier])

    return df


default_args = {'owner': 'airflow'}

path_db_producao = "bancoDiabetes.db"
path_temp_csv = "diabetes.csv"
path_transformed_csv = "diabetes_transformed.csv"
path_model_pkl = "modelo_diabetes.pkl"

dag = DAG(
    dag_id='data_pipeline_diabetes',
    default_args=default_args,
    schedule_interval='@daily',
    start_date=days_ago(2)
)


def _extract():
    # Conectando ao banco de dados de produção
    connect_db = sqlite3.connect(path_db_producao)
    
    # Selecionando os dados
    dataset_df = pd.read_sql_query(r"""
        SELECT *
        FROM diabetes;  # colocar o nome da tabela aqui
        """, 
        connect_db
    )
    
    # Exportando os dados para um arquivo CSV temporário
    dataset_df.to_csv(path_temp_csv, index=False)

    # Fechando a conexão com o banco de dados
    connect_db.close()


def _transform():
    dataset_df = pd.read_csv(path_temp_csv)

    # Verificando e tratando valores nulos
    dataset_df.fillna(dataset_df.mean(), inplace=True)

    # Padronizando os dados
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(dataset_df.drop(['Outcome'], axis=1))
    scaled_df = pd.DataFrame(scaled_features, columns=dataset_df.columns[:-1])
    scaled_df['Outcome'] = dataset_df['Outcome']

    # Chamando a função do ml
    numeric_cols = dataset_df.select_dtypes(include=["number"])
    for col in numeric_cols:
        dataset_df = eliminar_outlier(dataset_df, col)

    X = dataset_df.drop(['Outcome'], axis=1)
    y = dataset_df['Outcome']

    # Padronizar as features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dividir os dados em conjunto de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.80, random_state=0)

    # Treinar o modelo de regressão linear
    lm = LinearRegression()
    modelo_linear = lm.fit(X_train, y_train)

    # Fazer previsões
    predicoes = modelo_linear.predict(X_test)

    # Salvando os dados transformados em um novo arquivo CSV
    scaled_df.to_csv(path_transformed_csv, index=False)

    # Salvando o modelo treinado em um arquivo .pkl
    joblib.dump(modelo_linear, path_model_pkl)


def _load():
    try:
        # Adicionar os arquivos ao índice
        subprocess.run(['git', 'add', path_transformed_csv], check=True)
        subprocess.run(['git', 'add', path_model_pkl], check=True)

        # Fazer o commit dos arquivos
        commit_message = 'Adicionando novos arquivos CSV transformados e modelo PKL'
        subprocess.run(['git', 'commit', '-m', commit_message], check=True)

        # Fazer o push das mudanças para o repositório remoto
        subprocess.run(['git', 'push'], check=True)

        print("Arquivos commitados e enviados com sucesso.")
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar o comando: {e}")


extract_task = PythonOperator(task_id="extract", python_callable=_extract, dag=dag)
transform_task = PythonOperator(task_id="transform", python_callable=_transform, dag=dag)
load_task = PythonOperator(task_id="load", python_callable=_load, dag=dag)

# Definindo a ordem de execução das tarefas
extract_task >> transform_task >> load_task
