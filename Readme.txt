This file contains the python coding for generating the numerical results in the paper "De-confounding causal inference using latent multiple-mediator pathways".

Main required python packages:

Python 3.7
tensorflow 1.14.0 (other version may cause compatibility issue)numpy 1.19.2scikit-learn 0.22.2scipy 1.6.2copyeconml 0.10.0matplotlib 3.3.4keras 2.3.1pandas 1.3.5seaborn 0.11.2pot 0.8.1.0random 1.2.2utils 1.0.1rpy2 2.9.4

Required R packages:

R 3.6.0
ncvregdoParallelHIMA


Sim_6_1_m=2.py: the simulation results under Section 6.1 with number of mediators being 2 (columns 3,4,5 in Table 1 and Figure 3).

Sim_6_1_m=5.py: the simulation results under Section 6.1 with number of mediators being 5 (columns 6,7,8 in Table 1 and Figure 3).

Sim_6_2_low_rank.py: the simulation results under Section 6.2 with low-rank nonlinear confounding effect on mediators(columns 2,3,4 in Table 2 and and Figure 4).

Sim_6_2_full_rank.py: the simulation results under Section 6.2 with full-rank nonlinear confounding effect on mediators(columns 5,6,7 in Table 2 and and Figure 4).

real_data_example.py: mediation analysis on the Normative Aging Study (NAS) dataset by the proposed method, and other competing causal inference methods.   

mediator_index.csv, NIH_med_22.csv, NIH_pheno_norm.csv, sample_index.csv: csv data files used for generating real data application results.

src: folder contains all the functions used in numerical experiments and real data application.
main functions in src:
(1)Prop_FM.py: proposed method using factor model  
(2)Prop_AE: proposed method using autoencoder model
(3)Causal_Forest: causal forest method 
(4)X_learner: meta learner method 
(5)LSEM: linear structure equation model
(6)HIMA: HIMA method 
(7)DataGene.py: simulation data generation 
(8)utli.py: dependent functions library 

For each simulation and real data application, the codes record the average casual effect bias, mediation effect bias, and out-of-sample prediction for outcome. Please load all py files, csv files and src folder under the same python environment and working directory.  

To use HIMA method, need to call R package HIMA in python using rpy2. To use the rpy2, first need to set up the R_HOME environment variable, which points to the R installation on your system:
set the R_HOME: 

1. type "R RHOME" in terminal, and record the R address
2. Open python, and run following commend:

import os
os.environ["R_HOME"] = "R location obtained by R RHOME"

3. Replace "R location obtained by R RHOME" by the R address.



