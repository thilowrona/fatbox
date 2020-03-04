# File conversion (.csv to .npy)
We can analyse fault systems in 3-D numerical models using slices or horizons extracted in paraview. This is done by:

(1) Loading the data into Paraview

(2) Contouring a near-surface isotherm (e.g. using minium temperature in the model)

(3) Saving the file as .csv

To apply image processing tools in python, we can convert the .csv files into numpy arrays, which we save as .npy files. This can be done using the python script in this folder.
