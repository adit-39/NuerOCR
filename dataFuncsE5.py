import csv
import Image
import numpy as np
import matplotlib.pyplot as plt

def get_data():
	with open("letter.data",'r') as f:
		images = csv.reader(f, delimiter='\t')
		X=[]
		Y=[]
		for row in images:
			inp=[]
			out = row[1]
			#out = getCharIndexArray(char)
			p = row[6:]
			for j in p:
				if j=='':
					continue
				inp.append(j)
			X.append(inp)
			Y.append(out)
		d = {}
		d["X"] = X
		d["Y"] = Y
		return d

 
def generate_subsets(d):
	train = {}
	test = {}
	train["c"] = []
	test["v"] = []
	test["c"] = []
	train["v"] = []

	inputs = d["X"]
	outputs = d["Y"]
	f1 = False
	for i in range(len(outputs)):
		if get_type(outputs[i]):
			if len(train["v"]) < 2000 :
				train["v"].append(inputs[i])
			elif len(test["v"]) < 1000:
				test["v"].append(inputs[i])
			else:
				f1 = True
		else:
			if len(train["c"]) < 2000:
				train["c"].append(inputs[i])
			elif len(test["c"]) < 1000 :
				test["c"].append(inputs[i])
			else :
				if f1 == True:
					break

	return train,test

		
def get_type(c):
	if c in ['a','e','i','o','u']:
		return True
	else:
		return False


def zerolistmaker(n):
    listofzeros = [0] * n
    return listofzeros

def getCharIndexArray(ch):
	y = zerolistmaker(26)
	y[ord(ch) - ord('a')] = 1
	return y

def display_image(l):
	arr = np.zeros([16,8])
	for i in range(0,128,8):
		arr[i/8] = l[i:i+8] 
	plt.imshow(arr, interpolation='nearest')
	plt.show()



def strip_data(d):
	training = {}
	testing = {}
	training["X"] = []
	training["Y"] = []
	testing["X"] = []
	testing["Y"] = []
	for key in d.keys():
		training[key].extend(d[key][:800])
		testing[key].extend(d[key][800:1000])

	return training,testing

