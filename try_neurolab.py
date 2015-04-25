import neurolab as nl
import numpy as np
import sys

def read_data(filename):
	X = np.empty([0,10])
	Y = np.empty([0,2])
	training_data = dict()
	with open(filename,"r") as f:
		l = f.readlines()
	for i in l:
		s = eval(i.strip())
		X = np.vstack((X,np.array(s[:-2])))
		Y = np.vstack((Y,np.array(s[-2:])))
	training_data["X"] = X
	training_data["Y"] = Y
	return training_data

if __name__=="__main__":
	trg_file = sys.argv[1]
	tst_file = sys.argv[2]
	trg_data = read_data(trg_file)
	tst_data = read_data(tst_file)
	net = nl.net.newff([[-1,1]]*len(trg_data["X"][0]), [4, 2])
	err = net.train(trg_data["X"], trg_data["Y"], show=15)

	predicted=[]
	output = list(net.sim(tst_data["X"]))
	#print output

	for v in output:
		new=[]
		for u in list(v):
			if(u<0.5):
				new.append(0)
			else:
				new.append(1)
		predicted.append(new)
	
	actual = []
	for i in tst_data["Y"]:
		actual.append(list(i))

	#print actual[0]
	#print predicted[0]

	count = 0
	num_samples = len(predicted)
	for i in range(num_samples):
		if(predicted[i][0]==actual[i][0] and predicted[i][1]==actual[i][1]):
			count+=1
	print "Accuracy = {}%".format(count*100/float(num_samples))
