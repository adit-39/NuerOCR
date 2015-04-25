import neurolab as nl
import numpy as np
import dataFuncs as df

d = df.read_data()
train,test = df.strip_data(d)

trg_data = df.getFormattedData(train)
tst_data = df.getFormattedData(test)

net = nl.net.newff([[0,1]]*len(trg_data["X"][0]), [40, 26])
err = net.train(trg_data["X"], trg_data["Y"], show=1)

output = list(net.sim(tst_data["X"]))
print output[0]



# df.display_image(train['k'][9])

# print len(training["X"])
# print len(training["Y"])
# print "__________________________________________________________________"
# print len(testing["X"])
# print len(testing["Y"])

