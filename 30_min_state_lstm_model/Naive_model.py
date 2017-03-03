import pandas as pd
import numpy as np 
import itertools
from collections import defaultdict

def Take_N_consecutive(X,N):
	M=len(X)
	for i in range(M-N+1):
		yield tuple([X[j] for j in range(i,i+N)])

def Chain(X):
	return list(itertools.chain(*X))



class Naive_N_gram_model(object):
	"""docstring for Naive_model"""
	def __init__(self,gram_N=1):
		print ("gram_N:",gram_N)
		self.gram_N=gram_N
		self.Load_data()
		self.N_gram_model_fit()
		self.N_gram_model_validate()
		pass
	def Load_data(self):
		the_target_file="../Data/30_min.h5"
		To_load_hdf5= pd.HDFStore(the_target_file)
		Stock_list=sorted(To_load_hdf5.keys())
		self.Stock_list=Stock_list
		self.X_train=[To_load_hdf5[stock]["State"] for stock in Stock_list]
		self.Y_train=[To_load_hdf5[stock]["Ret"] for stock in Stock_list]
		N=len(self.X_train)
		split_N=int(N*0.75)
		self.X_test=self.X_train[split_N:]
		self.Y_test=self.Y_train[split_N:]
		self.X_train=self.X_train[:split_N]
		self.Y_train=self.Y_train[:split_N]
	
		print ("null model mse self.Y_train",np.var(Chain(self.Y_train)))
		print ("null model mse self.Y_test",np.var(Chain(self.Y_test)))
	
	def N_gram_model_fit(self):
		self.N_gram_model=defaultdict(list)
		# for stock in self.Stock_list:
		# 	for n_gram,ret in zip(Take_N_consecutive(self.X_test[stock]))
		for state_list,ret_list in zip(self.X_train,self.Y_train):
			for n_gram,ret in zip(Take_N_consecutive(state_list,self.gram_N),ret_list[self.gram_N-1:]):
				
				self.N_gram_model[n_gram].append(ret)
		self.N_gram_mean_model=defaultdict(float)
		for n_gram in self.N_gram_model.keys():
			self.N_gram_mean_model[n_gram]=np.mean(self.N_gram_model[n_gram])

		Error_list=[]
		for state_list,ret_list in zip(self.X_train,self.Y_train):
			for n_gram,ret in zip(Take_N_consecutive(state_list,self.gram_N),ret_list[self.gram_N-1:]):
				predicted_ret=self.N_gram_mean_model[n_gram] 
				error=ret-predicted_ret
				Error_list.append(error)
		print ("len(self.N_gram_mean_model)",len(self.N_gram_mean_model))
		print ("Insampe mse",np.var(Error_list))

	def N_gram_model_validate(self):
		Error_list=[]
		for state_list,ret_list in zip(self.X_test,self.Y_test):
			for n_gram,ret in zip(Take_N_consecutive(state_list,self.gram_N),ret_list[self.gram_N-1:]):
				predicted_ret=self.N_gram_mean_model[n_gram] 
				error=ret-predicted_ret
				Error_list.append(error)
		print ("len(Error_list),np.var(Error_list)",len(Error_list),np.var(Error_list))

def The_test():
	Naive_N_gram_model(1)
	Naive_N_gram_model(2)
	Naive_N_gram_model(3)

if __name__ == '__main__':
	The_test()