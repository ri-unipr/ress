import subprocess
import re
import sys
import time

#DCI PARAMETERS

#number of variables (starting system)
NA=16;
#number of bits per variable (starting system
NB=2;


#SIEVE PARAMETERS

#variable names (source system)
#CSTR
var_names=["A","AA","AAA","AAAA","AAAB","AAB","AABBA","AB","ABA","ABBBBA","B","BA","BAA","BAAB","BBB","BBBABA"]
#var_names=["A","B","AA","AB","BA","BB","C","CA","CB","AC","BC","CC","D","DA","DB","DADB","AAB","BAB","AABB","CCA","CBAC","BCCC"]

## SC aggiunta
var_string=""
for i in range(0,NA-1):
    var_string+=var_names[i]+"\t"

var_string+=var_names[NA-1]
##var_string+="ZI"


## SC fine aggiunta

#source system file: variable encoding
forig_var_bit= "systems/CSTR20_00_var_bit.txt"

#source file system: data
forig_data= "systems/CSTR20_00_dati.txt"

directory_input_file="system_data/"
## SC TODO: Verfificare che esista, altrimenti crearla

directory_output="results/"
## SC TODO: Verfificare che esista, altrimenti crearla



#HOMOGENEOUS SYSTEM PARAMETERS
arg_seed="--rand-seed:"
seed="123456"
arg_seed+=seed

start_time = time.time()

#SAVING THE BIT NUMBER FOR EACH VARIABLE
#reading the number of bits per original system variable
with open(forig_var_bit) as f:
    lines = f.readlines()

orig_var_bit_list=[]

for line in lines:
	num_bit=line.count('1')
	#print(type(num_bit)) #int
	orig_var_bit_list.append(num_bit)

print(orig_var_bit_list)
#print(type(orig_var_bit_list[0]))

if len(orig_var_bit_list)!=NA:
	print("Errore dimensione")
	sys.exit()


#GENERATION OF THE SIEVE FILE HEADING LINE

#header of binary coding variables
intestazione_var_bin=""

for i in range(0,NA):
	for j in range(0,NA):
		if(i==j):
			for k in range(0,NB):
				intestazione_var_bin+="1"
		else:
			for k in range(0,NB):
				intestazione_var_bin+="0"

	intestazione_var_bin+=" "

#removal of the last space
intestazione_var_bin=intestazione_var_bin[:-1]

print(intestazione_var_bin)



#sys.exit()

#DCI EXECUTION CYCLE PLUS SIEVE

zI_index=3;
iterazione=0;
num_var=NA;


#while (zI_index >=3 or num_var>2): #and?
while (zI_index >=3 and num_var>2):
        iterazione+=1
        print("\n\nITERAZIONE "+str(iterazione)+"\n\n")
        ## SC Aggiunto controllo per stampa risultati dentro kmpso
        if(iterazione==1):
            flag_init=0
        else:
            flag_init=1

#EXECUTION IN C ++
        input_file="system_"+str(iterazione-1)+".txt"
        arg_input_file=directory_input_file+input_file
        print(arg_input_file)
        output_file="result_"+str(iterazione)+".txt"
        arg_output=directory_output+output_file

    ## TODO: stampare intestazione variabili
##  SC	arg_output="--out:"+directory_output+output_file
	#args = ("./dci", arg_input_file, arg_seed, arg_output, "--silent")
	#args = ("./dci", arg_input_file, arg_seed, arg_output,"--hs-count:100000")
	#args = ("../bin/dci", arg_input_file, arg_output)
##  SC        if (num_var <= 12):
        if (num_var <= 12):
            args = ("../bin/dci", arg_input_file, "--tc", "--out:"+arg_output)
        else:
## SC            args_kmpso = str(num_var) + "2000 1 3 501 20 100 50 123456"
            args = ("../bin/kmpso", str(num_var), "2000", "1", "3", "501", "20", "100", "50", "123456", arg_input_file, arg_output, "tc", var_string, str(flag_init))

	#args = ("dci_15_02_2018/dci", arg_input_file, arg_seed, arg_output)
	#Or just:
	#args = "bin/bar -c somefile.xml -d text.txt -r aString -f anotherString".split()

	#args="./dci"+ " "+arg_input_file+" "+ arg_seed+" "+arg_output+ " --silent"
        print(args)

	#popen = subprocess.Popen(args, stdout=subprocess.PIPE)
        popen = subprocess.Popen(args)

        popen.wait()

	#output = popen.stdout.read()
	#print output



	#READING FILE OUTPUT

        fout=directory_output+output_file
        with open(fout) as f:
            lines = f.readlines()


        #variable names
        ## SC Nella prima riga di result_1.txt non ci sono i nomi delle variabili

        nomi_variabili=lines[0]
	#nomi_variabili=nomi_variabili.replace("\n", "")
        nomi_variabili=nomi_variabili.replace("\t", " ")
        nomi_variabili=re.split(' ',nomi_variabili)
        print(nomi_variabili)

        if(iterazione==1): #1
	    #removal of the last element: "zI"
            nomi_variabili=nomi_variabili[:-1]
        else:
            #removal of the last 2 elements: "zI", "comp"
            nomi_variabili=nomi_variabili[:-2]

        print(nomi_variabili)
        nomi_variabili=' '.join(nomi_variabili)
        print(nomi_variabili)

        #first group
        gruppo=lines[1]
        gruppo=gruppo.replace("\t", " ")
        gruppo=re.split(' ',gruppo)
        print(gruppo)
        if(iterazione==1): #1
            #removal of the last element: "zI"
            zI_index=float(gruppo[-1])
            gruppo=gruppo[:-1]
        else:
	    #removal of the last 2 elements: "zI", "comp"
            zI_index=float(gruppo[-2])
            gruppo=gruppo[:-2]
        print(gruppo)
        gruppo=' '.join(gruppo)
        print(gruppo)

        print("\nzI: "+str(zI_index)+"\n")

	#NEW VARIABLES GENERATION

	#new_variable files
        foutname = "variables/variables_"+str(iterazione)+".txt"

        nomi_list=re.split(' ',nomi_variabili)
        nomi_list_len=len(nomi_list)
        print("Numero variabili: "+str(nomi_list_len))

        gruppo_list=re.split(' ',gruppo)
        gruppo_list_len=len(gruppo_list)
        print(gruppo_list_len)

        if nomi_list_len!=gruppo_list_len:
            print("Errore dimensione")
            sys.exit()

        out_file = open(foutname,"w")

        nuove_variabili_list=[]

	#new group in first place
        nuove_variabili_list.append("")
	#print(nuove_variabili_list)

        for i in range(0,nomi_list_len):
            if(gruppo_list[i]=="0"):
                nuove_variabili_list.append(nomi_list[i])
            else:
                if(nuove_variabili_list[0]!=""):
                    nuove_variabili_list[0]+="+"
                nuove_variabili_list[0]+=nomi_list[i]

        print("Nuove variabili:")
        print(nuove_variabili_list)
        num_var=len(nuove_variabili_list)
        print("Numero nuove variabili: "+str(num_var))

        ## SC aggiunta
        var_string=""

        for i in range(0,len(nuove_variabili_list)-1):
            out_file.write(str(nuove_variabili_list[i])+" ")
            ## SC aggiunta
            var_string+=str(nuove_variabili_list[i])+"\t"
        ## SC aggiunta
        ##var_string+="ZI\tComp"
        var_string+=str(nuove_variabili_list[len(nuove_variabili_list)-1])

        out_file.write("\n")
        out_file.write("\n")

        for i in range(0,len(nuove_variabili_list)):
            if(i!=0):
                out_file.write(",")
            out_file.write("\""+str(nuove_variabili_list[i])+"\"")

        out_file.close()


	#NEW SYSTEM GENERATION
        new_names=nuove_variabili_list[:]
        print(new_names)

	# Writing a file.
        fnew="system_data/system_"+str(iterazione)+".txt"
        out_file = open(fnew,"w")

	#length of the vector of the new variables
        new_names_len=len(new_names)
	#length of the vector variables of the origin system(NA)
        orig_var_names_len=len(var_names)

	#header generation
        intestazione="%% "+' '.join(new_names)+" %% "+' '.join(var_names)+" %% "+intestazione_var_bin
        out_file.write(intestazione)
	#print(new_names)
	#print(var_names)

        out_file.write("\n")

        for i in range(0,new_names_len):
            #description of the bits of the new variables
            for j in range(0,new_names_len):
                if '+' in new_names[j]:
                    variables=new_names[j].split('+')
                    for k in range(0,len(variables)):
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
        #mapping of old variables into new variables
            for j in range(0,orig_var_names_len):
                if '+' in new_names[i]:
                    variables=new_names[i].split('+')
                    presence=False
                    for k in range(0,len(variables)):
                        if (var_names[j]==variables[k]):
                            presence=True
                            break
                    if presence:
                        out_file.write("1")
                    else:
                        out_file.write("0")
                else:
                    if(var_names[j]==new_names[i]):
                        out_file.write("1")
                    else:
                        out_file.write("0")
            out_file.write("\n")

        out_file.write("%%\n")


	#reading the original system data
        with open(forig_data) as f:
            lines = f.readlines()

	#print(lines)
        data_list=[]

        for line in lines:
            line=line.replace("\n", "")
            data_list.append(line)

	#print(type(data_list[0]))

	#number of samples
        data_len=len(data_list)
        print(data_len)

	#print(data_list[1][1])

	#generation of ordered data
        for i in range(0,data_len):
	#for i in range(0,1):
            #new order data (sieve)
            for j in range(0,new_names_len):
                if '+' in new_names[j]:
                    variables=new_names[j].split('+')
                    for k in range(0,len(variables)):
        		#var_names.index(variables[k])
        		#print(var_names.index(variables[k]))
                        for n_bit in range(0,NB):
        		    #print(str((var_names.index(variables[k])*NB)+n_bit))
        		    #print(str(n_bit))
                            out_file.write(data_list[i][var_names.index(variables[k])*NB+n_bit])
        		#print("\n")
                else:
                    for n_bit in range(0,NB):
        		#print(str((var_names.index(new_names[j])*NB)+n_bit))
        		#print(str(n_bit))
                        out_file.write(data_list[i][var_names.index(new_names[j])*NB+n_bit])
        		#print("\n")
            out_file.write(" %% ")
            #original data
            out_file.write(data_list[i])
            out_file.write("\n")

        out_file.close()

print("TIME")
print(time.time() - start_time)
