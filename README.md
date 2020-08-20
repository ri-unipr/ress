# ReSS (Relevance Set Search)

Implementation in C++ (requires CUDA 9 or higher), with Python >=3.7 scripts.

Once downloaded the whole project folder, compile the binaries.

On Linux machines:

* make dci
* make kmpso
* make homgen

The following parameters must be passed as arguments to homgen:

* 

The following parameters must be passed as arguments to dci:
* path/to/inputfile
* --tc or --zi, depending on the index that one wants to use (Tc or zI)
* --res:N to specify the number of reported results
* --out:path/to/outputfile
* --hsinputfile:path/to/hsfile to specify the path to the file of the homogeneous system (created by means of homgen)

The following paramaters must be defined and passed as arguments to kmpso:
* --dimension:D to specify the search space dimension
* --swarm_size:S to specify the swarm size  
* --n_seeds:K to specify the number of seeds  
* --range:x to specify the range  
* --n_iterations:T to specify the number of iterations
* --kmeans_interv:c1 to specify the kmeans interval
* --Print_interrv:interv to specify the print interval  
* --n_results:N to specify the number of reported results
* --rseed:r to specify the seed for the pseudorandom number generator
* --inputfile:path/to/inputfile
* --outputfile:path/to/outputfile
* --hsinputfile:path/to/hsfile to specify the path to the file of the homogeneous system (created by means of homgen)
* --tc or --zi, depending on the index that one wants to use (Tc or zI)
* --comp_on:0/1

To run the tools, edit and use Python the ress.py script (in the scripts/ folder):

* python ress.py
