from __future__ import print_function
import itertools
import cPickle
import zlib
import numpy as np
np.random.seed(1337)  # for reproducibility
import time
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,TimeDistributed,Flatten
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Convolution1D, MaxPooling1D
from keras.datasets import imdb
import scipy.signal as sg
from collections import Counter
import matplotlib.pyplot as plt
   


class Train_by_LSTM(object):
	"""docstring for Train_by_LSTM"""
	def __init__(self):
		self.config_hyper_para()
		self.define_architecture()
		self.train_the_model()
		# self.perform_outsample_test()

	def config_hyper_para(self):
		self.max_features=80
		self.embedding_size=10
		self.input_length=1024
		self.Drop_out_Embedding=0.00
		self.lstm_output_size=3
		self.lstm_dropout_W=0.00
		self.lstm_dropout_U=0.00

		self.batch_size=32
		self.Epochs=400

	def define_architecture(self):
		self.model=Sequential()
		self.model.add(Embedding(self.max_features,
		self.embedding_size, input_length=self.input_length))
		shape=self.model.output_shape
		print ('Embedding ouput shape = '+str(shape));
		self.model.add(Dropout(self.Drop_out_Embedding))
		
		
		self.model.add(LSTM(self.lstm_output_size,dropout_W=self.lstm_dropout_W,dropout_U=self.lstm_dropout_U,return_sequences=True))
		shape=self.model.output_shape
		
		print ('LSTM ouput shape = '+str(shape));
		self.model.add(TimeDistributed(Dense(1, activation='linear')))
		shape=self.model.output_shape
		print ('final ouput shape = '+str(shape));
		my_sgd=keras.optimizers.SGD(lr=0.5,momentum=0.1,decay=0.00,nesterov=False)
		self.model.compile(loss="mse",optimizer=my_sgd)

 
	def Load_data():
		the_target_file="../Data/zipped_data.data"
		self.The_data_dict=cPickle.loads(zlib.decompress( pen(the_target_file,"r").read()))
		Stock_list=sorted(self.The_data_dict.keys())
		X_train=[self.The_data_dict[stock]["State"] for stock in Stock_list]
		Y_train=[self.The_data_dict[stock]["Ret"] for stock in Stock_list]
		return X_train,Y_train

	def train_the_model(self):
		print('Train...')
		
		X_train,Y_train=self.Load_data()
		Y_train=[[[yt] for yt in sample] for sample in Y_train]
		self.X_train,self.Y_train=np.array(X_train),np.array(Y_train)
		
		print ('X_train_shape = '+str(np.array(self.X_train).shape))
		print ('Y_train_shape = '+str(np.array(self.Y_train).shape))
		
		
		time.sleep(5)
		my_early_stop=keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto')
		self.model.fit(x=self.X_train,y=self.Y_train,batch_size=self.batch_size,nb_epoch=self.Epochs,callbacks=[my_early_stop],validation_split=0.25)
		# self.model.fit(x=self.X_train,y=self.Y_train,batch_size=self.batch_size,nb_epoch=self.Epochs)

	def perform_outsample_test(self):
		pass

if __name__ == '__main__':
	Train_by_LSTM()