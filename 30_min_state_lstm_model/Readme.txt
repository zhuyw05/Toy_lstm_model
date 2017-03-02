数据说明
the_lstm.py 第71行读出的To_load_hdf5
这个对象可以视为一个dict，结构如下
To_load_hdf5[stock]={"State":State_list,"Ret":Ret_list}
每个股票的State_list和Ret_list长度相同且一一对应
state是一个离散的状态，取值从1到88（88种不同的状态）
ret是见到这个state后股票的涨跌，是我们要预测的目标。全样本的ret标准差（standard deviation）在3%左右，均值（mean）在0左右
所有Stock对应的State_list和Ret_list 长度本来不同，已经通过补零（state在开头补一个不存在于正常样本之中的状态 “0” ，ret补 0）让其长度相同
stock共有804只
