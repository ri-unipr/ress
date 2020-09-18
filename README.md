# ReSS (Relevance Set Search)

Implementation in C++ (requires CUDA 9 or higher), with Python >=3.7 scripts.

Once downloaded the whole project folder, compile the binaries.

On Linux machines:

* make eress
* make kress
* make homgen

The following parameters must be passed as arguments to homgen:

* path/to/inputfile
* --hs_output_file:path/to/hsfile to specify the path to the homogeneous system file to be created

The following parameters must be passed as arguments to eress:
* path/to/inputfile
* --tc or --zi, depending on the index that one wants to use (Tc or zI)
* --n_results:number to specify the number of reported results
* --output_file:path/to/outputfile
* --hs_input_file:path/to/hsfile to specify the path to the file of the homogeneous system (created by means of homgen)

The following paramaters must be defined and passed as arguments to kress:
* --tc or --zi, depending on the index that one wants to use (Tc or zI)
* --dimension:number to specify the search space dimension
* --swarm_size:number to specify the swarm size  
* --n_seeds:number to specify the number of seeds  
* --range:number to specify the range  
* --n_iterations:number to specify the number of iterations
* --kmeans_interv:number to specify the kmeans interval
* --print_interv:number to specify the print interval  
* --n_results:number to specify the number of reported results
* --rseed:number to specify the seed for the pseudorandom number generator
* --input_file:path/to/inputfile
* --output_file:path/to/outputfile
* --hs_input_file:path/to/hsfile to specify the path to the file of the homogeneous system (created by means of homgen)
* --comp_on:0/1

To run the tools, edit and use the ress.py script (in the scripts/ folder):

* python ress.py
