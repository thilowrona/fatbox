# Fault analysis toolbox

![alt text](https://github.com/thilowrona/fault_analysis_toolbox/blob/master/fault_network.png)

A python module for the extraction and analysis of faults (and fractures) in raster data. We often observer faults in 2-D or 3-D raster data (e.g. geological maps, numerical models or seismic volumes), yet the extraction of these structures still requires large amounts of our time. The aim of this module is to reduce this time by providing a set of functions, which can perform many of the steps required for the extraction and analysis of fault systems.

The basic idea of the module is to describe fault systems as graphs (or networks) consisting of nodes and edges, which allows us to define faults as components, i.e. sets of nodes connected by edges, of a graph. Nodes, which are not connected through edges, thus belong to different components (faults).

## Getting started
### Online tutorials
One of the easiest ways of getting started is with Jupyter notebooks - an awesome combination of code, documentation and output. The easiest way is to run them with Google Colab, so you don't need any special hardware or software. Just try it out:

### [Tutorial 1](https://github.com/thilowrona/fault_analysis_toolbox/blob/master/examples/1-fault_extraction/1-fault_extraction.ipynb)
- This tutorial shows how to extract a fault network from a 2-D slice of a numerical model

### [Tutorial 2](https://github.com/thilowrona/fault_analysis_toolbox/blob/master/examples/2-fault_correlation/2-fault_correlation.ipynb)
- This tutorial shows how to track fault networks through time in numerical models

### [Tutorial 3](https://github.com/thilowrona/fault_analysis_toolbox/blob/master/examples/3-slip/3-slip.ipynb)
- This tutorial shows how to calculate fault slip from numerical models

### [Tutorial 4](https://github.com/thilowrona/fault_analysis_toolbox/blob/master/examples/4-displacement/4-displacement.ipynb)
- This tutorial shows how to sum up fault slip to displacement through time




### Extra
### [Tutorial E1](https://github.com/thilowrona/fault_analysis_toolbox/blob/master/examples/E1-manual_extraction/E1-manual_extraction.ipynb)
- This tutorial shows how to map fault networks by hand

### [Tutorial E2](https://github.com/thilowrona/fault_analysis_toolbox/blob/master/examples/E2-paraview/E2-paraview.ipynb)
- This tutorial shows how to resample a 2-D numerical model to an array with Paraview, so we can apply our workflow

### [Tutorial E3](https://github.com/thilowrona/fault_analysis_toolbox/blob/master/examples/E3-export/E3-export.ipynb)
- This tutorial shows how to export our fault network for 3-D visualizations in Paraview




### Own machine
You can also use the fault analysis toolbox on your own machine. All you need is Python 3 including a couple of packages. I would recommend to install [ananconda](https://docs.anaconda.com/anaconda/install/), which gives you an enviroment with most of the packages and tools that we will use. Two are however missing. You can install OpenCV in the terminal like this:
```
conda install -c conda-forge opencv
```
and vtk like this:
```
pip install vtk
```
Now you can clone the git repository containing the fault analysis toolbox:

``` git clone https://github.com/thilowrona/fault_analysis_toolbox ```

To load the toolbox, open a Python editor (e.g. spyder) and run the following lines of code:
```
import sys
sys.path.append('./fault_analysis_toolbox/code/')
```
This sets the path to the toolbox and below we import the files containing the code (each one importing the necessary packages):
```
from image_processing import*
from edits import*
from metrics import*
from plots import*
from utils import*
```
If a package is missing, you will get an error highlighting a package that is missing, something like:
``` 
ModuleNotFoundError: No module named 'vtk' 
```
You can easily fix this by installing the missing package using either pip or Anaconda:
```
pip install vtk
```
With Anaconda you probably won't need to install many additional packages, as most of the packages that we use are included.





## Examples

### [1-Fault extraction](https://github.com/thilowrona/fault_analysis_toolbox/blob/master/examples/1-fault_extraction/1-fault_extraction.ipynb)
- This tutorial shows you how to extract a basic fault network from a 2-D slice of a numerical model simulating rifting


