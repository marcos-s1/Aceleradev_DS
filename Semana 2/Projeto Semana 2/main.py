#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[131]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[87]:


black_friday.head()


# In[132]:


df = black_friday.copy()


# In[133]:


df_info = pd.DataFrame({'Colunas': df.columns, 'types': black_friday.dtypes,
                             'NA #': df.isna().sum(),
                             'NA %': (df.isna().sum() / df.shape[0]) * 100})
df_info


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[135]:


def q1():
    return tuple(black_friday.shape)
q1()


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[91]:


def q2():
    return df[ (df['Gender'] == 'F') & (df['Age'] == '26-35') ].shape[0]
q2()


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[92]:


def q3():
    return df['User_ID'].nunique()
q3()


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[93]:


def q4():
    types = df.dtypes
    return types.nunique()
q4()


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[94]:


def q5():
    return df[df.isna().any(axis=1)].shape[0] / df.shape[0]

q5()


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[107]:


def q6():
    return int(df_info['NA #'].max())
q6()


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[96]:


def q7():
    return df['Product_Category_3'].value_counts().idxmax()
q7()


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[111]:


def q8():
    dados = df['Purchase']
    df['Purchase_normalizado']=(dados - dados.min()) / (dados.max() - dados.min())
    return float(df['Purchase_normalizado'].mean())

q8()


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[106]:


def q9():
    dados = df['Purchase']
    df['Purchase_padronizado'] = (dados - dados.mean()) / dados.std()
    return len( df[ abs(df['Purchase_padronizado']) <= 1])
q9()


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[129]:


def q10():
    product_category_2 = list(df['Product_Category_2'].isnull())
    product_category_3 = list(df['Product_Category_3'].isnull())
    
    return bool(True if x == y else False for (x,y) in zip(product_category_2, product_category_3))
    
q10()

