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


def display_image(l):
	#print l
	arr = np.zeros([16,8])
	for i in range(0,128,8):
		arr[i/8] = l[i:i+8] 
	
	print arr
	plt.imshow(arr, interpolation='nearest')
	plt.show()

data = read_data()
trng,tsting = strip_data(data)
display_image(trng['q'][21])