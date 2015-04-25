import csv
import Image
import numpy as np
import matplotlib.pyplot as plt

def read_data():
	with open("letter.data",'r') as f:
		images = csv.reader(f, delimiter='\t')
		d = {}
		for row in images:
			char = row[1]
			if char not in d.keys():
				d[char]= []
			p = row[6:]
			pixels =[]
			for j in p:
				if j=='':
					continue
				pixels.append(int(j))
			d[char].append(pixels)
			
		return d

def strip_data(d):
	training = {}
	testing = {}
	for key in d:
		training[key]=[]
		testing[key]=[]
		training[key].extend(d[key][:32])
		testing[key].extend(d[key][32:40])
	return training,testing

data = read_data()
trng,tsting = strip_data(data)
