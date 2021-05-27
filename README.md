# Fault analysis toolbox

![alt text](https://github.com/thilowrona/fault_analysis_toolbox/blob/master/fault_network.png)

A python module for the extraction and analysis of faults (and fractures) in raster data. We often observer faults in 2-D or 3-D raster data (e.g. geological maps, numerical models or seismic volumes), yet the extraction of these structures still requires large amounts of our time. The aim of this module is to reduce this time by providing a set of functions, which can perform many of the steps required for the extraction and analysis of fault systems.

The basic idea of the module is to describe fault systems as graphs (or networks) consisting of nodes and edges, which allows us to define faults as components, i.e. sets of nodes connected by edges, of a graph. Nodes, which are not connected through edges, thus belong to different components (faults).

# Getting started
## Tutorials
If you wanna get started, we highly recommend checking out some of our [tutorials](https://github.com/thilowrona/fatbox_tutorials) as well as our [documentation](https://fatbox.readthedocs.io/en/latest/index.html)

## Your own machine
You can also use the fault analysis toolbox on your own machine. All you need is a Python 3 enviromnent, then you can install the toolbox:
```
pip3 install git+https://github.com/thilowrona/fatbox.git
```
Now you can load any function from the toolbox in Python:
```
from fatbox.plots import plot_attribute
```
You can also clone the companion git repository containing the tutorials:

``` 
git clone https://github.com/thilowrona/fatbox_tutorials
```


## Citation
If you use this project in your research or wish to refer to the results of the tutorials, please use the following BibTeX entry.
```
@misc{fatbox2021,
  author =       {Thilo Wrona, Sascha Brune},
  title =        {{Fault analysis toolbox}},
  howpublished = {\url{https://github.com/thilowrona/fatbox}},
  year =         {2021}
}
```
