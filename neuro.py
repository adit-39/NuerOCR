import neurolab as nl
import numpy as np
import dataFuncs as df
import random

d = df.get_data()
trg_data,tst_data = df.strip_data(d)

net = nl.net.newff([[0,1]]*len(trg_data["X"][0]), [16, 26],transf=[nl.trans.TanSig(),nl.trans.SoftMax()])
err = net.train(trg_data["X"], trg_data["Y"], epochs = 1, show=1)

output = list(net.sim(tst_data["X"]))


count = 0
for i in range(len(output)):
	actual = tst_data["Y"][i]
	comp = output[i]
	m1 = max(actual)
	pos1 = [k for k, j in enumerate(actual) if j==m1]
	m2 = max(comp)
	pos2 = [k for k, j in enumerate(comp) if j==m2]
	if pos1[0] == pos2[0] :
		count+=1 
	print "Expected : {} , Generated : {}".format(chr(pos1[0]+97),chr(pos2[0]+97))

acc = float(count)/len(output)
print "Accuracy : {}".format(acc)

