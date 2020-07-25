import subprocess
import re
import sys
import time
import os
from os import path

#DCI PARAMETERS

#number of variables (starting system)
NA=16;
#number of bits per variable (starting system)
NB=2;

#SIEVE PARAMETERS

#variable names (source system)
#CSTR
variables=["A","AA","AAA","AAAA","AAAB","AAB","AABBA","AB","ABA","ABBBBA","B","BA","BAA","BAAB","BBB","BBBABA"]

var_string=""
for i in range(0,NA-1):
    var_string+=variables[i]+" "
var_string+=variables[NA-1]

#source system file: variable encoding
forig_var_bit= "systems/CSTR16_00_var_bit.txt"

#source file system: data
forig_data= "systems/CSTR16_00_data.txt"

directory_input_file="system_data/"
if not path.exists(directory_input_file):
    try:
        os.mkdir(directory_input_file)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)

directory_output= "results/"
if not path.exists(directory_output):
    try:
        os.mkdir(directory_output)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)


#HOMOGENEOUS SYSTEM PARAMETERS
arg_seed="--rand-seed:"
seed="123456"
arg_seed+=seed

start_time = time.time()

#SAVING THE BIT NUMBER FOR EACH VARIABLE
#reading the number of bits per original system variable
with open(forig_var_bit) as f:
    lines = f.readlines()

orig_var_bit_list = []

for line in lines:
	num_bit = line.count('1')
	orig_var_bit_list.append(num_bit)

print(orig_var_bit_list)

if len(orig_var_bit_list) != NA:
	print("Size error")
	sys.exit()


#GENERATION OF THE SIEVE FILE HEADING LINE

#header of binary coding variables
header_var_bin=""

for i in range(0,NA):
	for j in range(0,NA):
		if(i==j):
			for k in range(0,NB):
				header_var_bin+="1"
		else:
			for k in range(0,NB):
				header_var_bin+="0"

	header_var_bin+=" "

#removal of the last space
header_var_bin=header_var_bin[:-1]
print(header_var_bin)


#DCI EXECUTION CYCLE PLUS SIEVE

index=3;
iteration=0;
num_var=NA;

while (index >=3 and num_var>2):
        iteration+=1
        print("\n\niteration "+str(iteration)+"\n\n")
        ## SC Aggiunto controllo per stampa risultati dentro kmpso
        if (iteration == 1):
            flag_init = 0
        else:
            flag_init = 1

#EXECUTION IN C ++
        input_file="system_"+str(iteration-1)+".txt"
        arg_input_file=directory_input_file+input_file
        print(arg_input_file)
        output_file="result_"+str(iteration)+".txt"
        arg_output=directory_output+output_file

        if (num_var < 20):
            args = ("../bin/dci", arg_input_file, "--rand-seed:123456", "--tc", "--out:"+arg_output)
        else:
            args = ("../bin/kmpso", str(num_var), "2000", "1", "3", "501", "20", "100", "100", "123456", arg_input_file, arg_output, "tc", var_string, str(flag_init), "")

        print(args)
        popen = subprocess.Popen(args)
        popen.wait()

	#READING FILE OUTPUT

        fout=directory_output+output_file
        with open(fout) as f:
            lines = f.readlines()

        #variable names
        var_names=lines[0]
        var_names=var_names.replace("\t", " ")
        var_names=re.split(' ',var_names)
        print(var_names)

        if(iteration==1): #1
	    #removal of the last element: "index"
            var_names=var_names[:-1]
        else:
            #removal of the last 2 elements: "index", "comp"
            var_names=var_names[:-2]

        print(var_names)
        var_names=' '.join(var_names)
        print(var_names)

        #first group
        group=lines[1]
        group=group.replace("\t", " ")
        group=re.split(' ',group)
        print(group)
        if(iteration==1): #1
            #removal of the last element: "index"
            index=float(group[-1])
            group=group[:-1]
        else:
	        #removal of the last 2 elements: "index", "comp"
            index=float(group[-2])
            group=group[:-2]
        print(group)
        group=' '.join(group)
        print(group)

        print("\nindex: "+str(index)+"\n")

	#NEW VARIABLES GENERATION

	    #new_variable files
        foutname = "variables/variables_"+str(iteration)+".txt"

        names_list=re.split(' ',var_names)
        names_list_len=len(names_list)
        print("Number of variables: "+str(names_list_len))

        group_list=re.split(' ',group)
        group_list_len=len(group_list)
        print(group_list_len)

        if names_list_len!=group_list_len:
            print("Size error")
            sys.exit()

        out_file = open(foutname,"w")

        new_variables_list=[]

	    #new group in first place
        new_variables_list.append("")

        for i in range(0,names_list_len):
            if(group_list[i]=="0"):
                new_variables_list.append(names_list[i])
            else:
                if(new_variables_list[0]!=""):
                    new_variables_list[0]+="+"
                new_variables_list[0]+=names_list[i]

        print("New variables:")
        print(new_variables_list)
        num_var=len(new_variables_list)
        print("Number new variables: "+str(num_var))

        var_string=""

        for i in range(0,len(new_variables_list)-1):
            out_file.write(str(new_variables_list[i])+" ")
            var_string+=str(new_variables_list[i])+" "
        var_string+=str(new_variables_list[len(new_variables_list)-1])

        out_file.write("\n")
        out_file.write("\n")

        for i in range(0,len(new_variables_list)):
            if(i!=0):
                out_file.write(",")
            out_file.write("\""+str(new_variables_list[i])+"\"")

        out_file.close()

	#NEW SYSTEM GENERATION
        new_names=new_variables_list[:]
        print(new_names)

	# Writing a file.
        fnew="system_data/system_"+str(iteration)+".txt"
        out_file = open(fnew,"w")

	#length of the vector of the new variables
        new_names_len=len(new_names)
	#length of the vector variables of the origin system(NA)
        orig_variables_len=len(variables)

	#header generation
        header="%% "+' '.join(new_names)+" %% "+' '.join(variables)+" %% "+header_var_bin
        out_file.write(header)

        out_file.write("\n")

        for i in range(0,new_names_len):
            #description of the bits of the new variables
            for j in range(0,new_names_len):
                if '+' in new_names[j]:
                    vars = new_names[j].split('+')
                    for k in range(0,len(vars)):
                        if(i==j):
                            for n_bit in range(0,NB):
                                out_file.write("1")
                        else:
                            for n_bit in range(0,NB):
                                out_file.write("0")
                else:
                    if(i==j):
                        for n_bit in range(0,NB):
                            out_file.write("1")
                    else:
                        for n_bit in range(0,NB):
                            out_file.write("0")

            out_file.write(" %% ")
        #mapping old variables into new variables
            for j in range(0,orig_variables_len):
                if '+' in new_names[i]:
                    vars=new_names[i].split('+')
                    presence=False
                    for k in range(0,len(vars)):
                        if (variables[j]==vars[k]):
                            presence=True
                            break
                    if presence:
                        out_file.write("1")
                    else:
                        out_file.write("0")
                else:
                    if(variables[j]==new_names[i]):
                        out_file.write("1")
                    else:
                        out_file.write("0")
            out_file.write("\n")

        out_file.write("%%\n")

	#reading the original system data
        with open(forig_data) as f:
            lines = f.readlines()

        data_list=[]

        for line in lines:
            line=line.replace("\n", "")
            data_list.append(line)

	#number of samples
        data_len=len(data_list)
        print(data_len)

	#generation of ordered data
        for i in range(0,data_len):
            #new order data (sieve)
            for j in range(0,new_names_len):
                if '+' in new_names[j]:
                    vars=new_names[j].split('+')
                    for k in range(0,len(vars)):
                        for n_bit in range(0,NB):
                            out_file.write(data_list[i][variables.index(vars[k])*NB+n_bit])
                else:
                    for n_bit in range(0,NB):
                        out_file.write(data_list[i][variables.index(new_names[j])*NB+n_bit])
            out_file.write(" %% ")
            #original data
            out_file.write(data_list[i])
            out_file.write("\n")

        out_file.close()

print("TIME")
print(time.time() - start_time)
