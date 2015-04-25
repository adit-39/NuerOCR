import csv

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