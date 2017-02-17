from __future__ import print_function
import itertools

import numpy as np
np.random.seed(1337)  # for reproducibility

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

class Generate_sample_data(object):
	"""docstring for Generate_sample_data"""
	def __init__(self):
		self.config_para()
		# self.cgenerate_rule()
		# self.test()


	def config_para(self):
		self.N_series_train = 2048
		self.N_series_test = 512

		self.N_series_length=1024
		self.N_state_count=100
		self.effect_length=40
		self.decay_rate=0.95
		self.noise_amplititude=1

		self.ideal_count=0
		self.ideal_correct_count=0

		self.config_generate_rule()


	def config_generate_rule(self):
		self.state_value_dict=dict(zip(range(self.N_state_count) , np.random.randn(self.N_state_count)))	
		self.lasting_rule=np.random.randn(self.effect_length)*np.array([self.decay_rate**i for i in range(self.effect_length)])
		self.lasting_rule[0]=1
		# plt.plot(self.lasting_rule)
		# plt.show()

	def ideal_predict(self,state_list):
		state_value_list=[self.state_value_dict[s] for s in state_list]
		ideal_predict_result=sg.convolve(state_value_list,self.lasting_rule)[:len(state_list)]
		return ideal_predict_result

	def generate_one_series(self):
		the_random_state_list=np.random.randint(low=0,high=self.N_state_count,size=self.N_series_length) #1024
		ideal_value=self.ideal_predict(the_random_state_list)
		noise=self.noise_amplititude*np.random.randn(self.N_series_length)*np.std(ideal_value)
		the_series=noise+ideal_value

		self.ideal_count+=len(the_series)
		self.ideal_correct_count+=(sum(1+np.sign(the_series*ideal_value))/2)
		return the_random_state_list,the_series
	
	def generate(self):
		Train_data=[self.generate_one_series() for i in range(self.N_series_train)] #2048
		Test_data=[self.generate_one_series() for i in range(self.N_series_test)] #512
		X_train,Y_train=[x[0] for x in Train_data],[x[1] for x in Train_data]
		X_test,Y_test=[x[0] for x in Train_data],[x[1] for x in Train_data]

		print (self.ideal_count,self.ideal_correct_count)
		print ("ideal correct ratio",float(self.ideal_correct_count)/self.ideal_count)
		print ("np.var(Y_test)",np.var(Y_test))
		return X_train,Y_train,X_test,Y_test

		
class Train_by_LSTM(object):
	"""docstring for Train_by_LSTM"""
	def __init__(self):
		self.config_hyper_para()
		self.define_architecture()
		self.train_the_model()

	def config_hyper_para(self):
		self.max_features=100
		self.embedding_size=50
		self.input_length=1024
		self.Drop_out_Embedding=0.25
		self.lstm_output_size=50
		self.lstm_dropout_W=0.25
		self.lstm_dropout_U=0.25

		self.batch_size=128
		self.Epochs=30

	def define_architecture(self):
		self.model=Sequential()
		self.model.add(Embedding(self.max_features, self.embedding_size, input_length=self.input_length))
		shape=self.model.output_shape
		print ('Embedding ouput shape = '+str(shape));
		self.model.add(Dropout(self.Drop_out_Embedding))
		
		
		self.model.add(LSTM(self.lstm_output_size,dropout_W=self.lstm_dropout_W,dropout_U=self.lstm_dropout_U,return_sequences=True))
		shape=self.model.output_shape
		
		print ('LSTM ouput shape = '+str(shape));
		self.model.add(TimeDistributed(Dense(1, activation='linear')))
		shape=self.model.output_shape
		print ('final ouput shape = '+str(shape));
		
		self.model.compile(loss="mse",optimizer="sgd")




		pass
	def train_the_model(self):
		print('Train...')
		
		X_train,Y_train,X_test,Y_test=Generate_sample_data().generate()
		Y_train=[[[yt] for yt in sample] for sample in Y_train]
		Y_test=[[[yt] for yt in sample] for sample in Y_test]
		X_train,Y_train,X_test,Y_test=np.array(X_train),np.array(Y_train),np.array(X_test),np.array(Y_test)
		
		print ('X_train_shape = '+str(np.array(X_train).shape))
		print ('Y_train_shape = '+str(np.array(Y_train).shape))
		
		
		self.model.fit(x=X_train,y=Y_train,batch_size=self.batch_size,nb_epoch=self.Epochs)

	def perform_outsample_test(self):
		pass
	def launch_test(self):
		pass
		

if __name__ == '__main__':
	# Generate_sample_data().generate()
	Train_by_LSTM()