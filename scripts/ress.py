import subprocess
import os
import glob
from os import path
from sieve import execute_sieve

directory_input_file = "systems/"
if not path.exists(directory_input_file):
    try:
        os.mkdir(directory_input_file)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)
fileList = glob.glob(directory_input_file+"system_*.txt")
for filePath in fileList:
    try:
        os.remove(filePath)
    except:
        print("Error while deleting file : ", filePath)

directory_hs_file = "hsfiles/"
if not path.exists(directory_hs_file):
    try:
        os.mkdir(directory_hs_file)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)
fileList = glob.glob(directory_hs_file+"hsfile_*.txt")
for filePath in fileList:
    try:
        os.remove(filePath)
    except:
        print("Error while deleting file : ", filePath)

directory_output_file = "results/"
if not path.exists(directory_output_file):
    try:
        os.mkdir(directory_output_file)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)
fileList = glob.glob(directory_output_file+"result_*.txt")
for filePath in fileList:
    try:
        os.remove(filePath)
    except:
        print("Error while deleting file : ", filePath)


SIEVE = True

#number of variables (starting system)
NA=21;
#number of bits per variable (starting system)
NB=2;

#variable names (source system)
#CSTR
#variables=["A","AA","AAA","AAAA","AAAB","AAB","AABBA","AB","ABA","ABBBBA","B","BA","BAA","BAAB","BBB","BBBABA"]
variables = ["A","AA","AAA","AAAA","AAAB","AAA_AABBA","AAB","AABBA","AAB_AAAB","AA_AAAA","AB","ABA","ABBBBA","B","BA","BAA","BAAB","BAA_BBBABA","BBB","BBBABA","BBB_ABBBBA"]
var_string=""
for i in range(0,NA-1):
    var_string+=variables[i]+" "
var_string+=variables[NA-1]
#var_string = "[1] [2] [3] [4] [5] [6] [7] [8] [9] [10] [11] [12] [13] [14] [15] [16] [17] [18] [19] 20] [21] [22] [23] [24] [25] [26] [27] [28]"

input_file = "cstr_21.txt"
input_file_only_data = "cstr_21_data.txt"
arg_input_file = directory_input_file + input_file
input_encoding_file = "systems/cstr_21_var_bit.txt"
output_file = "output-tc-21var.txt"
arg_output_file = directory_output_file + output_file
hs_file = "hsfile-21.txt"
arg_hs_file = directory_hs_file + hs_file

if not os.path.isfile(arg_hs_file):
    args = ("../bin/homgen", "--input_file:"+directory_input_file+input_file, "--hs_output_file:"+arg_hs_file)
    popen = subprocess.call(args)

if not SIEVE:
    if (num_var < 18):
        args = ("../bin/dci", "--input_file:"+arg_input_file, "--tc", "--n_results:100", "--output_file:"+arg_output_file, "--hs_input_file:"+arg_hs_file, "--verbose")
    else:
        args = ("../bin/kmpso", "--dimension:"+str(NA), "--swarm_size:2000", "--n_seeds:7", "--range:3", "--n_iterations:501", "--kmeans_interv:20", "--print_interv:100", "-n_results:100", "--rseed:123456", "--input_file:"+arg_input_file, "--output_file:"+arg_output_file, "--tc", "--var_string:"+var_string, "--comp_on:0", "--hs_input_file:"+arg_hs_file)
        #args = ("../bin/kmpso", "--dimension:"+str(NA), "--swarm_size:2000", "--n_seeds:7", "--range:3", "--n_iterations:501", "--kmeans_interv:20", "--print_interv:100", "--N_results:100", "--rseed:123456", "--inputfile:"+input_file, "--outputfile:"+output_file, "--tc", "--comp_on:0", "--hsinputfile:"+hs_file)
    popen = subprocess.call(args)

if SIEVE:
    execute_sieve(NA, NB, variables, input_file, input_file_only_data, directory_input_file, input_encoding_file, hs_file, directory_hs_file, directory_output_file, var_string)
