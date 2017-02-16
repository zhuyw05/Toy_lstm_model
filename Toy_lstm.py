from __future__ import print_function
import itertools

import numpy as np
np.random.seed(1337)  # for reproducibility

# from keras.preprocessing import sequence
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation
# from keras.layers import Embedding
# from keras.layers import LSTM
# from keras.layers import Convolution1D, MaxPooling1D
# from keras.datasets import imdb
import scipy.signal as sg
from collections import Counter
import matplotlib.pyplot as plt

class Generate_sample_data(object):
	"""docstring for Generate_sample_data"""
	def __init__(self):
		self.config_para()
		# self.cgenerate_rule()
		self.test()


	def config_para(self):
		self.N_series_train = 1024
		self.N_series_test = 512

		self.N_series_length=1024
		self.N_state_count=100
		self.effect_length=40
		self.decay_rate=0.95
		self.noise_amplititude=2.8
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
		the_random_state_list=np.random.randint(low=0,high=self.N_state_count,size=self.N_series_length)
		print (len(the_random_state_list))
		ideal_value=self.ideal_predict(the_random_state_list)
		print (len(ideal_value))
		noise=self.noise_amplititude*np.random.randn(self.N_series_length)
		the_series=noise+ideal_value
		return the_random_state_list,the_series

	
	def generate(self):
		Train_data=[self.generate_one_series() for i in range(sele.N_series_train)]
		Test_data=[self.generate_one_series() for i in range(sele.N_series_test)]
		X_train,Y_train=[x[0] for x in Train_data],[x[1] for x in Train_data]
		X_test,Y_test=[x[0] for x in Train_data],[x[1] for x in Train_data]


if __name__ == '__main__':
	Generate_sample_data()
