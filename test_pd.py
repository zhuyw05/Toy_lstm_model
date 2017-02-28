import pandas as pd
The_dict={}
for stock_id in [str(x) for x in range(90000,90010)]:
	The_dict[stock_id]={"close":[],"Ret":[]}
	for x in range(20):
		The_dict[stock_id]["close"].append(x+0.5)
		The_dict[stock_id]["Ret"].append(x+0.78)
print The_dict

B=pd.DataFrame(The_dict)
print B