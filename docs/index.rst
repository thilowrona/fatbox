Fault analysis toolbox
=======================================================
A python module for the extraction and analysis of faults (and fractures) in raster data. We often observer faults in 2-D or 3-D raster data (e.g. geological maps, numerical models or seismic volumes), yet the extraction of these structures still requires large amounts of our time. The aim of this module is to reduce this time by providing a set of functions, which can perform many of the steps required for the extraction and analysis of fault systems.

The basic idea of the module is to describe fault systems as graphs (or networks) consisting of nodes and edges, which allows us to define faults as components, i.e. sets of nodes connected by edges, of a graph. Nodes, which are not connected through edges, thus belong to different components (faults).

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Install

.. automodule:: fatbox.preprocessing
    :members:
.. automodule:: fatbox.metrics
    :members:
.. automodule:: fatbox.edits
    :members:
.. automodule:: fatbox.plots
    :members:
.. automodule:: fatbox.utils
    :members:

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`