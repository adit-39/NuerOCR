import neurolab as nl
import numpy as np
import dataFuncsE5 as df
import random
from myhmm_log import MyHmmLog
import json


def train_machine(data,init_model_file):
    """
    Function to take in training data as a list of 10-member lists along with an
    initial model file and output a HMM machine trained with that data.
    """
    M = MyHmmLog(init_model_file)
    M.forward_backward_multi(data)
    return(M)

def test(output_seq,M1,M2):
    """
    Function to take in the test observation sequence and output "silent",
    "single" or "multi" based on which of the three machines gives the highest
    probability in the evaluation problem for that output sequence
    """
    output = []
    vowel_c = 0
    conso_c = 0
    for obs in output_seq:
        p1 = M1.forward(obs)
        p2 = M2.forward(obs)
       	if p1>p2 :
       		vowel_c += 1
       		#print 1 
       	else:
       		conso_c += 1
       		#print 0
       		
    return vowel_c,conso_c


train,testin = df.generate_subsets(df.get_data())
n_vowel = train["v"]
n_conso = train["c"]
M1 = train_machine(n_vowel,"neural_initial.txt")
M2 = train_machine(n_conso,"neural_initial.txt")
# with open("trained_hmm_vowel.txt", "w") as f:
# 	f.write(json.dumps(M1.model))
# with open("trained_hmm_consonant.txt", "w") as f:
# 	f.write(json.dumps(M2.model))
vc, cc = test(testin["v"],M1,M2)
print "---------------"
print "Error : {}".format(100*float(cc)/len(testin["v"]))
print "_____________________________________________-"
vc, cc = test(testin["c"],M1,M2)
print "---------------"
print "Error : {}".format(100*float(vc)/len(testin["c"]))