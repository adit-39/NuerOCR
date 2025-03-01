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
			out=[]
			char = row[1]
			out = getCharIndexArray(char)
			p = row[6:]
			for j in p:
				if j=='':
					continue
				inp.append(int(j))
			X.append(inp)
			Y.append(out)
		d = {}
		d["X"] = X
		d["Y"] = Y
		return d

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

def display_image(l):
	#print l
	arr = np.zeros([16,8])
	for i in range(0,128,8):
		arr[i/8] = l[i:i+8] 
	#print arr
	plt.imshow(arr, interpolation='nearest')
	plt.show()

def zerolistmaker(n):
    listofzeros = [0] * n
    return listofzeros

def getCharIndexArray(ch):
	y = zerolistmaker(26)
	y[ord(ch) - ord('a')] = 1
	return y



