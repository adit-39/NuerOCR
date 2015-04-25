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


def generate_subsets(d):
	sets = {}
	sets["v"] = []
	sets["c"] = []
	inputs = d["X"]
	outputs = d["Y"]
	for i in range(len(outputs)):
		if get_type(outputs[i]):
			if len(sets["v"]) < 500 :
				sets["v"].append(inputs[i])
			else:
				if len(sets["c"]) == 500:
					break
		else:
			if len(sets["c"]) < 500:
				sets["c"].append(inputs[i])
			if len(sets["v"]) == 500:
					break
	return sets
		
def get_type(c):
	if c in ['a','e','i','o','u']:
		return True
	else
		return False


def zerolistmaker(n):
    listofzeros = [0] * n
    return listofzeros

def getCharIndexArray(ch):
	y = zerolistmaker(26)
	y[ord(ch) - ord('a')] = 1
	return y


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