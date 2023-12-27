from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from statistics import mean, stdev
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

#Realiza previsão de preço e tendencia da ação MXRF11 baseado nos ultimos 90 dias


batch_size = 32
epochs = 100
timesteps = 90

#Carrega e trata base de dados
def CarregaDados(caminho):
    dados = pd.read_csv(caminho)
    dados = dados.dropna()
    dados = dados.drop(['Date'], axis=1)

    normalizador = MinMaxScaler()
    normalizador_previsao = MinMaxScaler()

    dados_previsao = dados['Open']
    dados_previsao = np.asarray(dados_previsao)
    dados_previsao = np.expand_dims(dados_previsao, axis=1)

    normalizador_previsao.fit_transform(dados_previsao)
    joblib.dump(normalizador_previsao,'normalizador_previsao')

    dados = normalizador.fit_transform(dados)
    joblib.dump(normalizador,'normalizador')

    #Criando variaveis previsores e preco_real para comparação
    #O input de previsores deve estar no modelo de array 3D (batch, timesteps, feature)
    # 'previsores' irá receber os 90 registros anteriores ao valor de previsão
    # 'preco_real' irá receber  o registro adjacente aos registros previsores
    # O loop for irá percorrer toda base dados com essa lógica criando uma base de treinamento

    previsores = []
    preco_real = []

    for i in range(timesteps, dados.shape[0]):
        previsores.append(dados[i - timesteps : i, 0 : 6 ])
        preco_real.append(dados[i , 0])

    previsores = np.asarray(previsores)
    preco_real = np.asarray(preco_real)
    preco_real = np.expand_dims(preco_real, axis=1)

    return previsores, preco_real

#Estrutura da rede neural
def CriaRede():
    previsores, preco_real = CarregaDados('MXRF11.SA.csv')
    modelo = Sequential()

    modelo.add(LSTM(units=150, return_sequences=True, input_shape=(previsores.shape[1], previsores.shape[2])))
    modelo.add(Dropout(0.3))

    modelo.add(LSTM(units=100, return_sequences=True))
    modelo.add(Dropout(0.3))

    modelo.add(LSTM(units=100, return_sequences=True))
    modelo.add(Dropout(0.3))

    modelo.add(LSTM(units=100))
    modelo.add(Dropout(0.3))

    modelo.add(Dense(units=1, activation='linear'))

    modelo.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

    return modelo

def Treinamento():
    previsores, preco_real = CarregaDados('MXRF11.SA.csv')
    previsores_teste, preco_real_teste = CarregaDados('MXRF11.SA_Teste.csv')


    modelo = CriaRede()

    ers = EarlyStopping(monitor='loss', min_delta= 1e-10, patience=10, verbose=1)
    rlr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, verbose=1)
    mc = ModelCheckpoint(monitor='loss', filepath='Modelo.0.1', save_best_only=True, verbose=1)

    resultado = modelo.fit(previsores,preco_real, batch_size=batch_size, epochs=epochs, validation_data=(previsores_teste,preco_real_teste), callbacks=[ers, rlr, mc])

    desvio = resultado.history['val_mae']
    desvio = stdev(desvio)

    media = resultado.history['val_mae']
    media = mean(media)

    plt.plot(resultado.history['loss'])
    plt.plot(resultado.history['val_loss'])
    plt.title('Relação de Função de Perda Treinamento e Teste')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend(('Treinamento', 'Teste'))
    plt.show()

    plt.plot(resultado.history['mae'])
    plt.plot(resultado.history['val_mae'])
    plt.title('Relação de variação em R$:\nMédia:'+str(media)+'\nDesvio Padrão:'+str(desvio))
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend(('Treinamento', 'Teste'))
    plt.show()

def Previsao(caminho):
    previsores, preco_real = CarregaDados(caminho)

    modelo = load_model('Modelo.0.1')

    resultado = modelo.predict(previsores)


    normalizador_previsao = joblib.load('normalizador_previsao')

    preco_real = normalizador_previsao.inverse_transform(preco_real)
    resultado = normalizador_previsao.inverse_transform(resultado)


    plt.plot(resultado)
    plt.plot(preco_real)
    plt.title('Relação previsão e preço real')
    plt.xlabel('Dias')
    plt.ylabel('Valor R$')
    plt.legend(('Previsão', 'Preço real'))
    plt.show()

Treinamento()
Previsao('MXRF11.SA_Teste.csv')
