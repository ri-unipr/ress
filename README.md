# Relevance Index (RI)

Implementation of the RI in C++ (requires CUDA 9 or higher).

Once downloaded the whole project folder, compile the binaries.

On Linux machines:

* make dci
* make kmpso

To test kmpso, enter the scripts folder and run:

* ./tc-kmpso-test.sh
* ./zi-kmpso-test.sh

The first test uses the Tc index, while the second test uses the ZI index.

To test dci and kmpso, enter the scripts folder and run:

* python automatic_sieve.py

Python 3 is assumed to be installed.

The following parameters must be defined and passed as arguments to dci:
* path/to/inputfile
* --out:path/to/outputfile

The following paramaters must be defined and passed as arguments to kmpso:
* D = search space dimension
* S = swarm size  
* K = number of seeds  
* x = range  
* T = number of iterations
* c1 = kmeans interval
* interv = print_interv  
* N = number of reported results
* rseed = seed for the pseudorandom number generator
* path/to/inputfile
* path/to/outputfile
* index = either zi or tc
* hseed = optional seed for the generation of the homogeneous system; if not set, hseed = rseed
