from bpnn import BPNN

import json
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.callbacks import Callback
import pandas as pd
import numpy as np
'''
pd.set_option("display.max_rows", None, "display.max_columns", None)
np.set_printoptions(threshold=np.inf)

#Data
data = pd.read_excel("data/bpnn_data.xlsx", engine='openpyxl')

data['TIPO_PRECIPITACION'] = pd.factorize(data.TIPO_PRECIPITACION)[0]
data['INTENSIDAD_PRECIPITACION'] = pd.factorize(data.INTENSIDAD_PRECIPITACION)[0]
data['ESTADO_CARRETERA'] = pd.factorize(data.ESTADO_CARRETERA)[0]
data['ACCIDENTE'] = pd.factorize(data.ACCIDENTE)[0]

# Delete dates
data.drop('FECHA_HORA', inplace=True, axis=1)
data['TEMERATURA_AIRE'] = data['TEMERATURA_AIRE'].astype(float)
data['HUMEDAD_RRELATIVA'] = data['HUMEDAD_RRELATIVA'].astype(float, errors = 'raise')
data['DIRECCION_VIENTO'] = data['DIRECCION_VIENTO'].astype(float, errors = 'raise')
data['VELOCIDAD_VIENTO'] = data['VELOCIDAD_VIENTO'].astype(float, errors = 'raise')

# Delete rows with empty values
data.replace(' ', np.nan, regex=True, inplace=True)
data.dropna(inplace=True)

#data.to_excel("data/data.xlsx", engine='openpyxl')
#data = pd.read_excel("data/data.xlsx", engine='openpyxl')

data = data.sample(frac=1)

#Splits data into 'yes' and 'no'
yesdf = data[data['ACCIDENTE'] >= 0.5].sample(frac=1) #3495
nodf = data[data['ACCIDENTE'] < 0.5].sample(frac=1)

yesSize = round(len(yesdf)*0.8)
noSize = round(len(nodf)*0.8)

data = pd.concat([yesdf.iloc[:yesSize-1,:], nodf.iloc[:noSize-1,:]])
data_Val = pd.concat([yesdf.iloc[yesSize:,:], nodf.iloc[noSize:,:]])

data.to_excel("data/data.xlsx", engine='openpyxl')
data_Val.to_excel("data/data_Val.xlsx", engine='openpyxl')
'''
data = pd.read_excel("data/data.xlsx", engine='openpyxl')
data_Val = pd.read_excel("data/data_Val.xlsx", engine='openpyxl')

size = round(len(data) * 0.15)
p_X = data.iloc[:size, :data.shape[1] - 1].values
p_Y = data.iloc[:size, data.shape[1] - 1:].values

size = round(len(data_Val) * 0.15)
val_X = data_Val.iloc[:size, : data_Val.shape[1] - 1].values
val_Y = data_Val.iloc[:size,  data_Val.shape[1] - 1:].values
print('Data preprocessing')
'''
#Splits data into 'yes' and 'no'
yesdf = data[data['ACCIDENTE'] >= 0.5].sample(frac=0.2) #3495
nodf = data[data['ACCIDENTE'] < 0.5].sample(frac=0.2)

data_Val = pd.concat([yesdf.iloc[:,:], nodf.iloc[:,:]])
data.drop(yesdf.index)
data.drop(nodf.index)

#size = round(len(data) * 0.8)
p_X = data.iloc[:, :data.shape[1] - 1].values
p_Y = data.iloc[:, data.shape[1] - 1:].values
val_X = data_Val.iloc[:, :data.shape[1] - 1].values
val_Y = data_Val.iloc[:, data.shape[1] - 1:].values
'''
epochs = 10

network = BPNN(p_X.shape[1], p_Y.shape[1], hidden_layers=2, hidden_size=p_X.shape[1])
lossH, predictionsH = network.fit(p_X, p_Y, p_eta=0.001, epochs=epochs)

print("Predict: ", network.predict(p_X[1000]))

# Keras
model = Sequential()
model.add(Flatten())
model.add(Dense(p_X.shape[1], input_dim=p_X.shape[1], activation=sigmoid))
model.add(Dense(p_X.shape[1], activation=sigmoid))
model.add(Dense(p_Y.shape[1], activation=sigmoid))#activation=tf.nn.sigmoid_cross_entropy_with_logits))
out = model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracies = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('accuracy'))

history = LossHistory()
#print('Keras evaluate: ', model.evaluate(p_X, p_Y))
model.fit(p_X, p_Y, epochs=epochs, batch_size=1, callbacks=[history])
print("(Keras) Predict: ", model.predict([data.iloc[1000:1020, :p_X.shape[1]].values]))

lossK = history.losses
predictionsK = history.accuracies

with open('data/p_Y.json', 'w') as f:
    json.dump(p_Y.T.tolist()[0],f)
with open('data/lossH.json', 'w') as f:
    json.dump(lossH.tolist(),f)
with open('data/predictionsH.json', 'w') as f:
    json.dump(predictionsH.tolist(),f)
with open('data/lossK.json', 'w') as f:
    json.dump(lossK, f)
with open('data/predictionsK.json', 'w') as f:
    json.dump(predictionsK, f)
with open('data/epochs.json', 'w') as f:
    json.dump(epochs,f)
'''
pHomemade = np.empty([len(val_Y),])
pKeras = np.empty([len(val_Y),])

for x,y,pH,pK in zip(val_X, val_Y, pHomemade, pKeras):
    pH = 1 if round(network.predict(x)[0,0]) == y else 0
    pK = 1 if round(model.predict([x.reshape(1,len(x))])[0,0]) == y else 0

print('End')

accuracyHomemade = 100*pHomemade.sum()/len(val_Y)
accuracyKeras = 100*pKeras.sum()/len(val_Y)
print(accuracyHomemade)
print(accuracyKeras)
'''