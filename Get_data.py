# -*- coding: utf-8 -*-
import itertools
import numpy as np 
import pickle
import os
import pandas as pd
import h5py
import datetime as dt
import zlib

def describe_quantile(X,A=[0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99],Print=False):
	print "mean",np.mean(X)
	X=sorted(X)
	N=len(X)
	Result=[]
	for a in A:
		if Print:
			print "{}% , {}".format(int(a*100),round(X[int(a*N)],4))
		Result.append(round(X[int(a*N)],4))
	if Print:
		print "min,{}".format(min(X))
		print "max,{}".format(max(X))
		print "mean/std",map(lambda x:round(x,5),[np.mean(X),np.std(X)])
		print "\n"
	return Result

def Chain(X):
	return list(itertools.chain(*X))

def read_hdf5(file_path, node_name):
	f = h5py.File(file_path, 'r')
	try:
		node_data = f[node_name]
		columns = node_data.items()
		data_array = {}
		for c in columns:
			col_name = c[0]
			# 获取每一个DataSet的attributes，通过get()函数获取指定attr名字的值
			data_type = node_data[col_name].attrs.get("DataType")
			#print(data_type[0] == "DateTime")
			d = node_data[col_name].value
			# 如果字段类型是DateTime，对数据进行日期转换
			if data_type[0].decode('utf-8') == "DateTime":
				data_array[col_name] = convert_hdf5_dt(d)
			else:
				data_array[col_name] = d
		df = pd.DataFrame(data_array)
		return df
	except:
		pass

def convert_hdf5_dt(dt_values):
	dt_array = []
	st = dt.datetime(1899, 12, 31)
	for c in dt_values:
		dt_array.append(st + dt.timedelta(days=c-1))
	return dt_array

def Make_up_with(X,compensator=0,target_size=1000):
	if len(X)>=target_size:
		return X 
	else:
		To_compensate=[compensator for i in range(target_size-len(X))]
		Result=To_compensate+X
		return Result



class Get_state_ret_data_from_one_folder(object):
	def __init__(self, The_folder):
		self.The_folder = The_folder
		self.Full_path_list=[]
		self.Full_data_dict={}

		self.Get_file_path_list()
		self.Get_content()
		self.Replace_state_to_num()
		self.Normalize_Ret()
		self.Normalize_size()
		self.Compress_and_dump()

	def Get_file_path_list(self):
		for _,_,file_path_list in os.walk(self.The_folder):
			for file_name in file_path_list:
				full_path=self.The_folder+"/"+file_name
				self.Full_path_list.append(full_path)
		print len(self.Full_path_list)

	def Get_content(self):
		for i,full_path in enumerate(self.Full_path_list):
			print (i,len(self.Full_path_list))
			stock_id=full_path.split("/")[-1][:8]
			self.Full_data_dict[stock_id]={"State":[],"Ret":[]}
			print (stock_id)
			comps = read_hdf5(full_path, 'CompInfos_16')
			N=len(comps["Close"])
			for i in range(3,N-1):
				ret=comps["Close"][i+1]/comps["Close"][i]-1
				state=comps["Symbol"][i]
				self.Full_data_dict[stock_id]["State"].append(state)
				self.Full_data_dict[stock_id]["Ret"].append(ret)
	
	def Replace_state_to_num(self):
		self.All_State_set=set([])

		for stock_id in sorted(self.Full_data_dict.keys()):
			for state in self.Full_data_dict[stock_id]["State"]:
				self.All_State_set.add(state)
		print len(self.All_State_set)
		The_state_num_map=dict([(state,i+1) for i,state in enumerate(sorted(self.All_State_set))])
		print ("max(The_state_num_map.values())",max(The_state_num_map.values()))

		for stock_id in sorted(self.Full_data_dict.keys()):
			self.Full_data_dict[stock_id]["State"]=[The_state_num_map[state] for state in self.Full_data_dict[stock_id]["State"]]

	def Normalize_size(self):
		Max_length=max([len(x["State"]) for x in self.Full_data_dict.values()])
		print ("Max_length",Max_length)
		for stock in sorted(self.Full_data_dict.keys()):
			self.Full_data_dict[stock]["State"]=Make_up_with(self.Full_data_dict[stock]["State"],0,Max_length)
			self.Full_data_dict[stock]["Ret"]=Make_up_with(self.Full_data_dict[stock]["Ret"],0,Max_length)

	def Normalize_Ret(self):
		Whole_Ret_list=Chain([self.Full_data_dict[stock]["Ret"] for stock in self.Full_data_dict.keys()])
		print ("len(Whole_Ret_list)",len(Whole_Ret_list))

		print ("describe_quantile(Whole_Ret_list)",describe_quantile(Whole_Ret_list,A=[0.0001,0.001,0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99,0.999,0.9999],Print=True))
		print ("min(Whole_Ret_list),max(Whole_Ret_list)",min(Whole_Ret_list),max(Whole_Ret_list))
		winsorize_func=lambda x:min(max(x,-0.03),0.03)
		Winsorize_Whole_ret_list=map(winsorize_func,Whole_Ret_list)
		
		the_mean=np.mean(Winsorize_Whole_ret_list)
		the_dist=max(max(Winsorize_Whole_ret_list)-the_mean,the_mean-min(Winsorize_Whole_ret_list))
		print ("the_mean,the_dist",the_mean,the_dist)

		the_trans_func=lambda x:(x-the_mean)/(the_dist/0.5)+0.5
		Trans_Whole_ret=map(the_trans_func,Winsorize_Whole_ret_list)
		print ("np.mean(Trans_Whole_ret),max(Trans_Whole_ret),min(Trans_Whole_ret)",np.mean(Trans_Whole_ret),max(Trans_Whole_ret),min(Trans_Whole_ret))
		for stock in sorted(self.Full_data_dict.keys()):
			self.Full_data_dict[stock]["Ret"]=[the_trans_func(winsorize_func(x)) for x in self.Full_data_dict[stock]["Ret"]]

	def Compress_and_dump(self):

		# pickle.dump(self.Full_data_dict,open("./Data/zipped_data.data","w"))

		h5 = pd.HDFStore('./Data/30_min.h5','w', complevel=4, complib='blosc')
		for stock_id in sorted(self.Full_data_dict.keys()):
			h5[stock_id]=pd.DataFrame(self.Full_data_dict[stock_id])
		# h5["data"]=pd.DataFrame(self.Full_data_dict)
		h5.close()

		To_load_hdf5= pd.HDFStore('./Data/30_min.h5')
		print ("To_load_hdf5['SZ000936']",To_load_hdf5["SZ000936"])
		# cPickle.dump(zlib.compress(dumps,9),open("30_min_data.unzipeddata","w"))


def Check_load_data():
	To_load_hdf5= pd.HDFStore('./Data/30_min.h5')
	print (dir(To_load_hdf5))
	print (To_load_hdf5.keys())


def Operate_script():
	Get_state_ret_data_from_one_folder(r'Z:\HDF5\compinfos\Normal\Stock\HalfHour_true')

	
if __name__ == "__main__":
	# hdf5file = r'Z:\HDF5\compinfos\Normal\Stock\HalfHour_true\SH600005_201001_201703_30mins_HDF5_True.h5'
	# segs = read_hdf5(hdf5file, 'Segments_16')
	# print(segs.head(10))

	# pvts = read_hdf5(hdf5file, 'Pivots_16')
	# print(pvts.head(10))

	# tps = read_hdf5(hdf5file, 'TradePoints_16')
	# print(tps.head(10))

	# comps = read_hdf5(hdf5file, 'CompInfos_16')
	# print(comps.head(6))
	# print (len(comps["Close"]))
	Operate_script()

	# Check_load_data()