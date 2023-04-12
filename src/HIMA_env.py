

## to use the rpy2, first need to find the R_HOME environment variable, 
#which points to the R installation on your system:
#set the R_HOME: in terminal, type "R RHOME" and plugin the address into the 
#following commend: 
#import os
#os.environ["R_HOME"] = "R location obtained by R RHOME"

import rpy2
import utils
print(rpy2.__version__)

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1) 


packnames = ('ncvreg', 'doParallel', 'HIMA')
from rpy2.robjects.vectors import StrVector

# Selectively install what needs to be install.
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))
    
    
from rpy2.robjects import FloatVector
from rpy2.robjects import DataFrame
import rpy2.robjects.numpy2ri
from rpy2.robjects import r
import rpy2.robjects as ro
from src.HIMA import *     

   
