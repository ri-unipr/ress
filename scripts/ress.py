import subprocess
import os
from os import path
from sieve import execute_sieve

SIEVE = True

#number of variables (starting system)
NA=16;
#number of bits per variable (starting system)
NB=2;

#variable names (source system)
#CSTR
variables=["A","AA","AAA","AAAA","AAAB","AAB","AABBA","AB","ABA","ABBBBA","B","BA","BAA","BAAB","BBB","BBBABA"]
var_string=""
for i in range(0,NA-1):
    var_string+=variables[i]+" "
var_string+=variables[NA-1]
#var_string = "[1] [2] [3] [4] [5] [6] [7] [8] [9] [10] [11] [12] [13] [14] [15] [16] [17] [18] [19] 20] [21] [22] [23] [24] [25] [26] [27] [28]"

input_file = "systems/CSTR16_00_data.txt"
input_encoding_file = "systems/CSTR16_00_var_bit.txt"
output_file = "./results/output-tc-16var.txt"
hs_file = "./hsfile-16.txt"

if not os.path.isfile(hs_file):
    args = ("../bin/homgen", input_file, "--hs-out:"+hs_file)
    popen = subprocess.call(args)

if not SIEVE:
    if (num_var < 22):
        args = ("../bin/dci", input_file, "--rseed:123456", "--tc", "--res:132", "--out:"+output_file, "--hsinputfile:"+hs_file, "--verbose")
    else:
        #args = ("../bin/kmpso", "--dimension:"+str(NA), "--swarm_size:2000", "--n_seeds:7", "--range:3", "--n_iterations:501", "--kmeans_interv:20", "--print_interv:100", "--N_results:100", "--rseed:123456", "--inputfile:"+input_file, "--outputfile:"+output_file, "--tc", "--var_string:"+var_string, "--comp_on:0", "--hsinputfile:"+hs_file)
        args = ("../bin/kmpso", "--dimension:"+str(NA), "--swarm_size:2000", "--n_seeds:7", "--range:3", "--n_iterations:501", "--kmeans_interv:20", "--print_interv:100", "--N_results:100", "--rseed:123456", "--inputfile:"+input_file, "--outputfile:"+output_file, "--tc", "--comp_on:0", "--hsinputfile:"+hs_file)
    popen = subprocess.call(args)

if SIEVE:
    execute_sieve(NA, NB, variables, input_file, input_encoding_file, var_string)
