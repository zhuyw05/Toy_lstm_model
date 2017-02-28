from __future__ import print_function
import itertools

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
		self.perform_outsample_test()

	def config_hyper_para(self):
		self.max_features=10
		self.embedding_size=5
		self.input_length=1024
		self.Drop_out_Embedding=0.00
		self.lstm_output_size=5
		self.lstm_dropout_W=0.00
		self.lstm_dropout_U=0.00

		self.batch_size=256
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

 


		pass 
	def train_the_model(self):
		print('Train...')
		
		X_train,Y_train,X_test,Y_test,self.ideal_loss_func,self.var_Y=Generate_sample_data().generate()
		Y_train=[[[yt] for yt in sample] for sample in Y_train]
		Y_test=[[[yt] for yt in sample] for sample in Y_test]
		self.X_train,self.Y_train,self.X_test,self.Y_test=np.array(X_train),np.array(Y_train),np.array(X_test),np.array(Y_test)
		
		print ('X_train_shape = '+str(np.array(self.X_train).shape))
		print ('Y_train_shape = '+str(np.array(self.Y_train).shape))
		
		
		time.sleep(5)
		my_early_stop=keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto')
		self.model.fit(x=self.X_train,y=self.Y_train,batch_size=self.batch_size,nb_epoch=self.Epochs,callbacks=[my_early_stop],validation_split=0.25)
		# self.model.fit(x=self.X_train,y=self.Y_train,batch_size=self.batch_size,nb_epoch=self.Epochs)

	def perform_outsample_test(self):
		print ("ideal_loss_func,var_Y",self.ideal_loss_func,self.var_Y)
		test_result=self.model.evaluate(self.X_test,self.Y_test, batch_size=self.batch_size)
		print ("out-sample test_result",test_result)
	def launch_test(self):
		pass
