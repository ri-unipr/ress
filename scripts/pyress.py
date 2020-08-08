import subprocess
#import re
#import sys
#import time
import os
from os import path


num_var = 28
var_string = "[1] [2] [3] [4] [5] [6] [7] [8] [9] [10] [11] [12] [13] [14] [15] [16] [17] [18] [19] 20] [21] [22] [23] [24] [25] [26] [27] [28]"
input_file = "./systems/cstr_28.txt"
dci_output_file = "./results/output-tc-dci-28var.txt"
kmpso_output_file = "./results/output-tc-kmpso-28var.txt"

hs_file = "./hsfile-28.txt"

if not os.path.isfile(hs_file):
    args = ("../bin/homgen", input_file, "--rand-seed:123456", "--hs-out:"+hs_file)
    popen = subprocess.call(args)

if (num_var < 22):
    args = ("../bin/dci", input_file, "--rand-seed:123456", "--tc", "--res:132", "--out:"+dci_output_file, "--hs-in:"+hs_file, "--verbose")
else:
    args = ("../bin/kmpso", "--dimension:"+str(num_var), "--swarm_size:2000", "--n_seeds:1", "--range:3", "--n_iterations:501", "--kmeans_interv:20", "--print_interv:100", "--N_results:100", "--rseed:123456", "--inputfile:"+input_file, "--outputfile:"+kmpso_output_file, "--tc", "--var_string:"+var_string, "--comp_on:0", "--hsinputfile:"+hs_file)

popen = subprocess.call(args)
