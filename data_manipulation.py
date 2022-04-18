import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# modelagem
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import LabelEncoder
from math import sqrt

# outros
import gc
import warnings
from pylab import rcParams
from functools import reduce
from tqdm import tqdm #Barra de progresso


def check_duplicate_lines(data_frame):
    if data_frame.shape[0]  !=data_frame.drop_duplicates().shape[0]:
        duplicate_lines = dfata_frame.shape[0]-data_frame.drop_duplicates().shape[0]
        return f'{duplicated_lines} duplicate lines'
    return '0 duplicate lines'


def check_missing(df):
    """
    Returna uma Series com o percentual de missing data em cada coluna.
    """
    import pandas
    if isinstance(df, pandas.core.frame.DataFrame):
        missing = (((df.isnull().sum()/df.shape[0])*100).round(2)).sort_values(ascending = False)
        return missing
    return -1


def show_percentage_missing(df):
    import matplotlib.pyplot as plt
    
    "Mostra o percentual de missing em cada coluna graficamente."
    missing = check_missing(df)
    plt.figure(figsize = (10, 15))
    plt.barh(y = missing.index, width = missing.values, color = 'darkgray', height = 0.7, align = 'edge')
    plt.xlabel('% of missing values', size = 10)
    plt.ylabel('Columns', size = 10)
    plt.title('Missing Values', fontdict = {'color':'gray', 'weight':'bold', 'size': 12})
    plt.grid(alpha = 0.5)
    plt.show()


def reduce_mem_usage(df):
    """
    Redução do uso de Memória RAM do DF
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    #for col in tqdm([x for x in df.columns if 'NU_NOTA_' not in x]):
    for col in tqdm([x for x in df.columns]):
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df



# *********** Modelos ****************

    
    
# *********** Modelo Lasso ***********
def print_importancias_lasso(df, coef):
    '''
    importância das variáveis explicativas do modelo lasso
    '''
    for e in sorted (list(zip(list(df), coef)), key = lambda e: -abs(e[1])):
        if e[1] != 0:
            print('\t{}, {:.3f}'.format(e[0], e[1]))


def trata_predicoes(valor):
    '''
    garante que valores das notas estarão sempre entre 0 e 1000
    '''
    if valor < 0:
        return 0
    elif valor > 1000:
        return 1000
    else:
        return valor


def rmse_score(true, pred):
    '''
    rmse score
    '''
    return (sqrt(mean_squared_error(true, pred)))


def algoritmo_lasso(df, lista_targets, lista_vars_explicativas):
    lista_df_submissao = []
    lista_erros_treino = []
    lista_erros_teste = []
    # faz um modelo por target (nota da prova)
    for target in lista_targets:
        print ('***************')
        print (target)
        print ('***************')
        # Separa dados em treino (80%) e teste (20%) - Hold out split
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=369)

        # treino
        X_train = df_train[lista_vars_explicativas]
        y_train = df_train[target].copy()

        # teste
        X_test = df_test[lista_vars_explicativas]
        y_test = df_test[target].copy()

        # submit
        X_submit = df[lista_vars_explicativas]

        # modelo Lasso
        clf = Lasso(alpha=1.0, max_iter = 10000)
        clf.fit(X_train, y_train)

        # importâncias das variáveis explicativas
        print_importancias_lasso(X_train, clf.coef_)

        # Erro de treino
        y_pred = [trata_predicoes(valor) for valor in clf.predict(X_train)]
        erro = rmse_score(y_train, y_pred)
        print ('Erro de treino:', erro)
        lista_erros_treino.append(erro)

        # Erro de teste
        y_pred = [trata_predicoes(valor) for valor in clf.predict(X_test)]
        erro = rmse_score(y_test, y_pred)
        print ('Erro de teste:', erro)
        lista_erros_teste.append(erro)

        # Previsão para o dataset de submissão
        df_submit_results = df[['INSCRICAO']].copy()
        df_submit_results[target] = [trata_predicoes(valor) for valor in clf.predict(X_submit)]
        lista_df_submissao.append(df_submit_results)

        # limpa memória
        del df_train, df_test, X_train,  X_submit, X_test, y_train, y_test
        gc.collect()
    return lista_df_submissao, lista_erros_treino, lista_erros_teste

