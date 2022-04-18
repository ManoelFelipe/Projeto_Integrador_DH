import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import geopandas as gpd
import geobr
from scipy import stats
from scipy.stats import pearsonr

def feature_plot(feature, data):
    # adjusts subplots and plot size
    fig, ax = plt.subplots(1, 3, figsize = (15, 5))
    fig.suptitle(f'Univariate analysis for {feature}')
    # insert histogram in axis 0
    sns.histplot(data[feature], kde = True, ax = ax[0])
    ax[0].set_xlabel('Distribution of '+feature)
    # insert boxplot in axis 1
    sns.boxplot(y = data[feature], ax = ax[1])
    # insert violinplot in axis 2
    sns.violinplot(x = data[feature], ax = ax[2])
    # adjust spacing between subplots
    fig.tight_layout(pad = 3)


def univariate_analysis(features: list, data = pd.DataFrame):
    for feature in features:
        feature_plot(feature, data)

        
def estatistica_descritiva_por_estado(df, metrica):
    "Calcula alguma estatística descritiva para as notas do Enem por estado."
    df_estados = geobr.read_state(year = 2020)
    # provas do dataset de base
    provas = ['CIENCIAS_NATUREZA', 'HUMANAS', 'LINGUAGENS', 'MATEMATICA', 'REDACAO']
    # obtém os resultados por estado conforme medida estatística inserida
    df = df.groupby(by = 'SG_UF', as_index = False)[provas].agg(metrica)
    # geolocalização
    df = gpd.GeoDataFrame(pd.merge(
    df,
    df_estados,
    left_on = 'SG_UF',
    right_on = 'abbrev_state',
    how = 'inner'))

    return df


def plot_mapa_estado(df, estatistica_descritiva = np.mean, title = '', cmap = 'BuPu'):
    '''
    gera mapa heatmap para o Brasil populado com a estatística descritiva de interesse
    '''
    # cria o DataFrame conforme estatística descritiva definida
    df = estatistica_descritiva_por_estado(df=df, metrica = estatistica_descritiva)
    # labels para o pllot
    labels_provas = ['Ciências da Natureza', 'Ciências Humanas', 'Linguagens', 'Matemática', 'Redação']
    # colunas referentes a prova
    provas = ['CIENCIAS_NATUREZA', 'HUMANAS', 'LINGUAGENS', 'MATEMATICA', 'REDACAO']
    # cria a figura
    fig, ax = plt.subplots(1, 5, figsize = (20, 20))
    # itera na lista de provas e cria o mapa
    for index, prova in enumerate(provas):
        df.plot(
            column = prova,
            cmap = cmap,
            edgecolor = 'lightgray',
            lw = 0.3,
            ax = ax[index],
            legend=True,
            legend_kwds = {'shrink': 0.08}
            )
        # remove marcações dos eixos
        ax[index].axis('off')
        # labels
        ax[index].set_title(labels_provas[index], fontsize  = 10)

        fig.suptitle(title, y = 0.6 , weight = 'bold')
        fig.tight_layout(pad = 2);

        
def percentile(n):
    '''
    retorna percentil
    '''
    def percentile_(x):
        return x.quantile(n)
    percentile_.__name__ = 'percentile_{:2.0f}'.format(n*100)
    return percentile_


# função para gerar um histgrama a partir de uma variável do data frame
def gerar_histograma(data_frame,
                     variavel, 
                     bins = 30,
                     color = 'red',
                     xlabel = 'Variável',
                     ylabel = 'Frequência',
                     titulo = 'Histograma',
                     fontsize = 15,
                     fontweight = 'bold',
                     figsize = (8,5)
                    ):
    fig, ax = plt.subplots(figsize = figsize)
    ax.hist(data_frame[variavel], bins = bins,
            color = color)
    ax.set(xlabel = xlabel, ylabel = ylabel)
    ax.set_title(titulo, fontsize = fontsize,
                 fontweight = fontweight
                 );
    # Hide grid lines
    ax.grid(False)


# função para gerar um boxplot
def box_plot(data, title, xlabel, ylabel, figsize = (12, 5)):
    fig, ax = plt.subplots(figsize = figsize)
    sns.boxplot(data = data, ax = ax)
    ax.set(title = title, xlabel = xlabel, ylabel = ylabel)
    # Hide grid lines
    ax.grid(False)
    

# função para gerar um boxplot baseado em algum filtro
# o argumento order serve para alterar a ordem dos elementos no eixo x
def boxplot_por_filtro(df, filtro, order = None):
    'Gera um boxplot com filtro para o eixo x e a variável no eixo y.'
    # provas = ['MATEMATICA','CIENCIAS_NATUREZA', 'LINGUAGENS', 'HUMANAS','REDACAO']
    provas = ['Matemática']
    filtro_tratado = ' '.join(filtro.split('_')).capitalize()
    # Hide grid lines
    #ax.grid(False)
    #for prova in provas+['MEDIA']:
    for prova in provas+['Média_Geral']:
        prova_nome_minusculo = prova.lower()
        fig, ax = plt.subplots(figsize = (15, 5))
        sns.boxplot(x = filtro, y = prova, data = df, ax = ax,
                   order = order)
        ax.set(
            xlabel = filtro_tratado, 
               ylabel = f'Nota em {prova_nome_minusculo}', 
               title = f'Nota em {prova_nome_minusculo} filtrada por {filtro_tratado}')
        ax.yaxis.grid(False)
        ax.xaxis.grid(False)


def map_cor(cor):
    '''
    Mapeia cor/raca de acordo com o metadados fornecido
    '''
    if cor == 0:
        return 'Não informado'
    elif cor == 1:
        return 'Branca'
    elif cor == 2:
        return 'Preta'
    elif cor == 3:
        return 'Parda'
    elif cor == 4:
        return 'Amarela'
    elif cor == 5:
        return 'Indígena'
    else:
        return ''

    
def gerar_painel_barra(data_frame,
                      var,
                      hue,
                      title = '',
                      title_subplot_1 = '',
                      title_subplot_2 = '',
                      legend_subplot_2 = '',
                      xlabel = 'Quantidade',
                      ylabel = '',
                      figsize = (12, 6)
                      ):
    '''
    Gera gráfico de barras
    '''
    fig, ax = plt.subplots(1, 2, figsize = figsize)
    sns.countplot(data = data_frame,
                y = var,
                ax = ax[0])
    sns.countplot(data = data_frame,
                y = var,
                hue = hue,
                ax = ax[1])
    ax[0].set(ylabel = ylabel, xlabel = xlabel, title = title_subplot_1)
    ax[1].set(ylabel = ylabel, xlabel = xlabel, title = title_subplot_2)
    ax[1].legend(title = legend_subplot_2)
    fig.suptitle(title)
    fig.tight_layout(pad = 4)
    

def inscritos_estado(df):
    # Calcula a quantidade de inscritos em cada estado (amostra)
    # Através dessa análise, podemos concluir que a maria dos inscritos é de São Paulo
    # df_estados = geobr.read_state(year = 2020)
    df_municipio = geobr.read_municipality(code_muni="SP", year=2020)
    df_inscritos_por_estado = df.groupby(by = 'Município_Prova')[['Município_Prova']].count()\
    .rename(columns = {'Município_Prova': 'Quantidade_inscritos'})\
    .reset_index()\
    .sort_values(by = 'Quantidade_inscritos', ascending = False)

    # inscritos por estado com geolocalização
    df_inscritos_por_estado_spatial_data = gpd.GeoDataFrame(pd.merge(
        df_inscritos_por_estado,
        df_municipio,
        left_on = 'Município_Prova',
        right_on = 'name_muni',
        how = 'inner'
    ))

    fig, ax = plt.subplots(figsize = (10, 8))
    df_inscritos_por_estado_spatial_data.plot(
        column = 'Quantidade_inscritos',
        cmap = 'Reds',
        legend = True,
        legend_kwds= {
            "label": "Quantidade de inscritos",
            "orientation": "horizontal",
            "shrink": 0.5,
        },
        edgecolor = 'lightgray',
        lw = 0.2,
        ax = ax
    )

    ax.set_title('Quantidade de inscritos no Enem por estado')
    ax.axis('off');
    

def quant_insc_muni(df):    
    # cria um vetor para armazenar os 30 municípios com maior número de inscritos
    municipios = df['Município_Prova'].value_counts()[:30]

    # gera um gráfico de barras para mapear a quantidade de inscritos por municípios
    fig, ax = plt.subplots(figsize = (15, 4))
    ax.bar(x = municipios.index,
           height = municipios.values,
           color = 'red', edgecolor = 'darkred',
           linewidth = 1
          )

    # Hide grid lines
    ax.grid(False)
    ax.set_xticklabels(labels = municipios.index,rotation = 90);
    ax.set_ylim(0, municipios.values[0]*1.1)
    ax.set(xlabel = 'Municípios', ylabel = 'Quantidades de Inscritos')
    ax.set_title('Quantidade de inscritos por municípios', 
                 pad = 10, fontsize = 15, fontweight = 'bold');

    
def plot_dist_notas(df):
    # Define uma lista das provas que serão analisadas
    provas = ['MATEMATICA','CIENCIAS_NATUREZA', 'LINGUAGENS', 'HUMANAS','REDACAO']
    provas_ = provas+['MEDIA']
    index = 0
    xlabels = ['Matemática', 'Ciências da Natureza', 'Linguagens',
               'Ciências Humanas', 'Redação', 'Média'
              ]
    fig, ax = plt.subplots(2, 3, figsize = (15, 8))

    # Plotando as notas em histogramas
    for linha in [0, 1]:
        for coluna in [0, 1, 2]:
            sns.distplot(x = df[provas_[index]], ax = ax[linha, coluna])
            ax[linha, coluna].set(xlabel = provas_[index])    

            index+=1

    fig.tight_layout(pad = 4)
    # Incluindo o Titulo na Figura
    fig.suptitle('Distribuição das notas', fontsize=22, color='#404040', fontweight=600);


def histo_nota_uf(df):
    # Analisando Residencia x Notas
    #Analise_Target = df[['SG_UF_RESIDENCIA', 'Ciências da Natureza', 'Ciências Humanas', 'Linguagens', 'Matemática', 'Redação']]
    Analise_Target = df[['SG_UF', 'CIENCIAS_NATUREZA', 'HUMANAS', 'LINGUAGENS', 'MATEMATICA', 'REDACAO']]

    # Criando o relátorio
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))

    # Plotando as notas em histogramas
    sns.histplot(data=Analise_Target, x='CIENCIAS_NATUREZA', color='skyblue', bins=100, ax=axs[0, 0])
    sns.histplot(data=Analise_Target, x='HUMANAS', color='olive', bins=100, ax=axs[0, 1])
    sns.histplot(data=Analise_Target, x='LINGUAGENS', color='gold', bins=100, ax=axs[1, 0])
    sns.histplot(data=Analise_Target, x='MATEMATICA', color='teal', bins=100, ax=axs[1, 1])
    sns.histplot(data=Analise_Target, x='REDACAO', color='olive', bins=50, ax=axs[2, 0])
    axs[2, 1].set_axis_off()

    # Incluindo o Titulo na Figura
    plt.suptitle('Distribuição das notas para cada prova', fontsize=22, color='#404040', fontweight=600);
    


def map_estado_civil(estado_civil):
    '''
    mapeia estado civil de acordo com o metadados fornecido
    '''
    if estado_civil == 0:
        return 'Não informado'
    elif estado_civil == 1:
        return 'Solteiro(a)'
    elif estado_civil == 2:
        return 'Casado(a)/Mora com companheiro(a)'
    elif estado_civil == 3:
        return 'Divorciado(a)/Desquitado(a)/Separado(a)'
    elif estado_civil == 4:
        return 'Viúvo(a)'
    else:
        return ''


def estado_civil(df):
    # Estado civil
    df['MAP_ESTADO_CIVIL'] = df['Estado_Civil'].apply(map_estado_civil)

    fig, ax = plt.subplots(1,2, figsize = (12, 6))
    sns.countplot(data = df,
                 y = 'MAP_ESTADO_CIVIL', ax = ax[0])
    sns.countplot(data = df,
                 y = 'MAP_ESTADO_CIVIL', hue = 'Gênero', ax = ax[1])
    ax[0].set(ylabel = 'Estado Civil', xlabel = 'Quantidade',title = 'Estado civil')
    ax[1].set(ylabel = 'Estado Civil', xlabel = 'Quantidade',title = 'Estado civil por gênero')
    ax[1].legend(title = 'Gênero')
    fig.suptitle('Estado civil dos inscritos')
    fig.tight_layout(pad = 4);
    

def renda_nota(df):
    # Correlação entre renda (variável categórica) e a nota em Matemática
    # Através dessa análise é possível concluir que a renda familiar
    # está muito correlacionada (correlação de ~42%) com a nota do Enem

    dict_map_renda = {
        'A': '0) 0',
        'B': '1) 998',
        'C': '2) 1.497',
        'D': '3) 1.996',
        'E': '4) 2.495',
        'F': '5) 2.994',
        'G': '6) 3.992',
        'H': '7) 4.990',
        'I': '8) 5.988',
        'J': '9) 6.986',
        'K': '10) 7.984',
        'L': '11) 8.982',
        'M': '12) 9.980',
        'N': '13) 11.976',
        'O': '14) 14.970',
        'P': '15) 19.960',
        'Q': '16) > 19.960'
    }

    plt.figure(figsize=(20, 10))
    plt.title('Renda familiar vs. Nota na prova de matemática do Enem', fontsize=18)
    plt.plot(df.groupby(df['Renda_mensal_familiar'].map(dict_map_renda))['MATEMATICA'].mean(),
                color='blue',
                alpha=0.5)
    plt.show()


    dict_map_renda = {
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3,
        'E': 4,
        'F': 5,
        'G': 6,
        'H': 7,
        'I': 8,
        'J': 9,
        'K': 10,
        'L': 11,
        'M': 12,
        'N': 13,
        'O': 14,
        'P': 15,
        'Q': 16
    }

    df_2 = df.dropna(how = 'any', subset = 'Renda_mensal_familiar')
    df_corr = df_2.query("MATEMATICA == MATEMATICA")
    corr, _ = pearsonr(df_corr['Renda_mensal_familiar'].map(dict_map_renda), df_corr['MATEMATICA'])
    print ("Correlação:", corr)



def media_mediana_estado(df):
    # Analisando Residência x Notas na prova do Enem
    Analise_UF = df[['SG_UF', 'CIENCIAS_NATUREZA', 'HUMANAS', 'LINGUAGENS', 'MATEMATICA', 'REDACAO']]

    # Tamanho da Imagem
    fig, ax = plt.subplots(figsize=(20, 10))

    # Cor de fundo
    Cor_Fundo = "#F5F4EF"
    ax.set_facecolor(Cor_Fundo)
    fig.set_facecolor(Cor_Fundo)

    # Estilo do gráfico
    plt.style.use('seaborn-darkgrid')

    # Posição do Plot
    plt.subplot(2, 1, 1)

    # Plot
    plt.plot(Analise_UF.groupby( by='SG_UF').median(), linewidth=4, alpha=0.7)

    # Titulo
    plt.title('Mediana das provas por Estado', loc='left', fontsize=14, fontweight=0)

    # Labels
    plt.xlabel('Estados', fontsize=14 )
    plt.ylabel('Nota de 0 - 1000', fontsize=14)

    # Legenda
    plt.legend( ['CIENCIAS_NATUREZA', 'HUMANAS', 'LINGUAGENS', 'MATEMATICA', 'REDACAO'], 
               ncol=5, fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True )

    # Posição do Plot
    plt.subplot( 2, 1, 2 )

    # Plot
    plt.plot( Analise_UF.groupby( by='SG_UF').mean(), linewidth=4, alpha=0.7 )

    # Titulo
    plt.title('Média das provas por Estado', loc='left', fontsize=14, fontweight=0)

    # Labels
    plt.xlabel('Estados', fontsize=14)
    plt.ylabel('Nota de 0 - 1000', fontsize=14)

    # Legenda
    plt.legend( ['CIENCIAS_NATUREZA', 'HUMANAS', 'LINGUAGENS', 'MATEMATICA', 'REDACAO'], 
               ncol=5, fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True )

    # Ajustando distancias dos gráficos no relatorio
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.91, wspace=0.2, hspace=0.4)

    # Incluindo o Titulo na Figura
    plt.suptitle('Média/Media por Estado do participante', 
                 fontsize=22, color='#404040', fontfamily='sans-serif', fontweight=600 );
                 #fontsize=22, color='#404040', fontfamily='KyivType Sans', fontweight=600 );
                

def cor_nota(df):
    brancos = df.query("COR == 1 and MATEMATICA == MATEMATICA")['MATEMATICA']
    pretos = df.query("COR == 2 and MATEMATICA == MATEMATICA")['MATEMATICA']

    print (f'Nota média em matemática de inscritos brancos: {round(brancos.mean(), 2)}')
    print (f'Nota média em matemática de inscritos pretos: {round(pretos.mean(), 2)}')
    print (f'Percentil 99 da nota em matemática de inscritos brancos: {round(np.quantile(brancos, 0.99), 2)}')
    print (f'Percentil 99 da nota em matemática de inscritos pretos: {round(np.quantile(pretos, 0.99), 2)}')
    # Teste de hipótese
    # Como o p-valor é baixíssimo, devemos rejeitar a hipótese nula (de que não existe correlação entre a cor do inscrito e a sua nota de Matemática no Enem)
    # Desse modo, é possível concluir que a cor está correlacionada com a nota de Matemática do Enem
    print(f'Kolmogorov-Smirnov test: {stats.ks_2samp(brancos, pretos)}')

    plt.hist(brancos, bins=30, label='Branco', histtype='stepfilled', color='green')
    plt.hist(pretos, bins=30, label='Preto', alpha=.7, histtype='stepfilled', color='red')
    plt.xlabel('Nota em Matemática')
    plt.ylabel('Quantidade')
    plt.axvline(x=brancos.mean(), color='limegreen')
    plt.axvline(x=pretos.mean(), color='darkred')
    plt.title('Distribuição das notas de Matemática dos inscritos brancos e pretos')
    plt.grid(b=None)
    plt.legend()
    plt.show()

    # Boxplot das notas de Matemática de brancos e pretos
    dict_notas = {'brancos': brancos.values.tolist(),
                  'pretos': pretos.values.tolist()}
    fig, ax = plt.subplots(figsize = (10, 10))
    ax.boxplot(dict_notas.values())
    ax.set_xticklabels(dict_notas.keys())
    plt.title('Boxplot das notas de Matemática dos inscritos brancos e pretos')
    plt.grid(b=None)
    plt.show()
    