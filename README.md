# Fault analysis toolbox
A python module for the extraction and analysis of faults (and fractures) in raster data. We often observer faults in 2-D or 3-D raster data (e.g. geological maps, numerical models or seismic volumes), yet the extraction of these structures still requires large amounts of our time. The aim of this module is to reduce this time by providing a set of functions, which can perform many of the steps required for the extraction and analysis of fault systems.

The basic idea of the module is to describe fault systems as graphs (or networks) consisting of nodes and edges, which allows us to define faults as components, i.e. sets of nodes connected by edges, of a graph. Nodes, which are not connected through edges, thus belong to different components (faults).

## Getting started
### Online
One of the easiest ways of getting started is with Jupyter notebooks - an awesome combination of code, documentation and output. The easiest way is to run them with Google Colab, so you don't need any special hardware or software. Just try it out:

[https://github.com/thilowrona/fault_analysis_toolbox/blob/master/examples/example_1/example_1.ipynb](https://github.com/thilowrona/fault_analysis_toolbox/blob/master/examples/example_1/example_1.ipynb)





## Examples

### [Example 1](https://github.com/thilowrona/fault_analysis_toolbox/blob/master/examples/example_1/example_1.ipynb)
- This tutorial shows you how to extract a basic fault network from a 2-D numerical model simulating rifting

## Setup
1. Install **Python 3**
2. Install required packages listed in ```requirements.txt```. You can install these packages with:

``` pip install -r /path/to/requirements.txt ```

3. Clone git repository:

```git clone https://github.com/thilowrona/fault_analysis/```



## Fault extraction
![Image description](/examples/1-fault_extraction/flowchart.png)

The extraction of faults from raster data typically involves these three steps: (1) pre-processing, (2) thresholding, and (3) clustering. The basic idea is that we first clean our data, then identify areas which belong to faults, and finally classify these areas into faults.



## Fault analysis
Once we have extracted a fault network, we can perform many of the steps of a fault analysis usign the functions of this module. For example, we can calculate fault lengths as the sum of the edge lengths (i.e. distances between connected nodes) of each component.




