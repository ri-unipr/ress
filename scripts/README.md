This folder contains an example ress.py script and a set of bash scripts that refer to different datasets.

The main datasets are:

* CSTR16 
* CSTR26
* CSTR28

They all represent a Continuous-flow Stirred-Tank Reactor. 

The CSTR16 is a simple system  where there are two distinct reaction pathways, a linear reactions chain (CHAIN) and an
autocatalytic set of molecular species (ACS). Both reactions pathways occur in an open well-stirred
chemostat (CSTR) with a constant influx of feed molecules and a continuous outgoing flux of all the molecular species
proportional to their concentration. The dynamics of the system is described adopting a deterministic approach
whereby the reaction scheme is translated in a set of Ordinary Differential Equations (ODE) integrated by means a fourthorder Runge-Kutta method.
The main entities of the model are molecular species (“polymers”) represented by linear strings of letters A and B,
forming together a catalytic reactions system composed of 6 distinct condensation reactions in which two species are glued
to create a longer species. The reactions occur only in presence of a specific catalyst, since spontaneous reactions are
assumed to occur too slowly to affect the system behavior.   According  to  the  three  molecular  nature  of  the 
condensation reaction, reactions occur in 2  two  steps: in the former  the  catalyst  binds  the  first  substrate  forming  a 
molecular  complex,  while  in  latter  the  molecular  complex binds  the  second  substrate  releasing  the  product  and  the 
catalyst.  The  “food  set”  of  the  linear  chain is  formed  by  the  species ABB, BBA, BBB, ABA, BAA, B, whereas the food set  of  and  the  autocatalytic  cycle  is  formed  by  the  species  BA, AAB, AAA, A, AB, AA. Besides, an independent molecular species BB not involved in any reactions has been introduced as control species.  

In the 26-variable system (CSTR26) we consider as variables also some temporary catalyzer-substrate complexes, while in the 16-variable system (CSTR16) we do not consider them.

The CSTR28 system presents again two forms of chemical reactions of type Reﬂexive Autocatalytic Food-generated (RAF): (i) a linear chain and (ii) a simple cycle between two reactions, supporting a linear "tail". The considered CSTR features a constant inﬂux of feed molecules (constantly
present in CSTRs and therefore playing the role of the “food” species constituting the base of RAF arrangements) and a continuous outgoing ﬂux of all the molecular species
proportional to their concentration. The analyzed scheme involves enzymatic condensations, whose process is considered as being composed of three
steps: the ﬁrst two creates (reversibly) a temporary complex (composed by one of the two substrates and the catalyst) that can be used by a third reaction, which combines the
complex and a second substrate to ﬁnally release the catalyst and the ﬁnal product. The aforementioned three steps are summarized as follows:
(1) Complex formation: A + C ⟶ C_comp -> A:C.
(2) Complex dissociation: A:C ⟶ C_diss -> A + C.
(3) Final condensation: A:C + B ⟶ C_cond -> AB + C.
C_comp, C_diss, and C_cond are the reaction kinetic constants of complex formation, complex dissociation, and final condensation, respectively. The dynamic of the systems is described adopting a deterministic approach, whereby the reaction
scheme is translated into a set of Ordinary Differential Equations ruled by the mass action law and integrated by means of a custom Euler method
with step-size control. The main entities of the model are molecular species
(“polymers”), represented by linear strings of letters (A, B, C, and D). There are seven distinct condensation reactions divided into two distinct RAF pathways: a chain
of linear reactions (RAF1), the presence of whose root is guaranteed from the outside, and a RAF where two reciprocally catalysing reactions are the roots of another linear
reaction chain (RAF2).

All CSTR systems have been perturbed across fixed points, in order to trigger a response in the concentration of (some) other species, recording the transients.
According to the used discretization and of the performed perturabations, some variables may result to be constant, thus they are excluded from the considered system (leading to the 21-variable and 22-variable systems).
