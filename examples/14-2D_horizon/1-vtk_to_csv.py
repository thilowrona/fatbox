import sys

sys.path.insert(1,'/home/wrona/ParaView-5.8.0-MPI-Linux-Python3.7-64bit/lib/python3.7/site-packages/')



#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'PVD Reader'
solutionpvd = PVDReader(FileName='/home/wrona/Desktop/aspect/28a_lk26e_outdt50ky/output/solution.pvd')

# get animation scene
animationScene1 = GetAnimationScene()

# get the time-keeper
timeKeeper1 = GetTimeKeeper()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# Properties modified on solutionpvd
solutionpvd.PointArrays = ['strain', 'strain_rate']

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [1184, 542]

# get layout
layout1 = GetLayout()

# show data in view
solutionpvdDisplay = Show(solutionpvd, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
solutionpvdDisplay.Representation = 'Surface'

# reset view to fit data
renderView1.ResetCamera()

# get the material library
materialLibrary1 = GetMaterialLibrary()

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Slice'
slice1 = Slice(Input=solutionpvd)

# Properties modified on slice1.SliceType
slice1.SliceType.Origin = [225000.0, 75000.0, 195000.0]
slice1.SliceType.Normal = [0.0, 0.0, 1.0]

# show data in view
slice1Display = Show(slice1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
slice1Display.Representation = 'Surface'

# hide data in view
Hide(solutionpvd, renderView1)

# update the view to ensure updated data information
renderView1.Update()


for time in range(0, 11250000, 50000):


    # Properties modified on animationScene1
    animationScene1.AnimationTime = time
    
    # Properties modified on timeKeeper1
    timeKeeper1.Time = time
    
    # reset view to fit data
    renderView1.ResetCamera()
    
    # reset view to fit data
    renderView1.ResetCamera()
    
    # reset view to fit data
    renderView1.ResetCamera()
    
    # set scalar coloring
    ColorBy(slice1Display, ('POINTS', 'strain'))
    
    # rescale color and/or opacity maps used to include current data range
    slice1Display.RescaleTransferFunctionToDataRange(True, False)
    
    # show color bar/color legend
    slice1Display.SetScalarBarVisibility(renderView1, True)
    
    # get color transfer function/color map for 'strain'
    strainLUT = GetColorTransferFunction('strain')
    
    # get opacity transfer function/opacity map for 'strain'
    strainPWF = GetOpacityTransferFunction('strain')
    
    # create a new 'Resample To Image'
    resampleToImage1 = ResampleToImage(Input=slice1)
    
    # Properties modified on resampleToImage1
    resampleToImage1.SamplingDimensions = [5000, 1600, 1]
    
    # show data in view
    resampleToImage1Display = Show(resampleToImage1, renderView1, 'UniformGridRepresentation')
    
    # trace defaults for the display properties.
    resampleToImage1Display.Representation = 'Slice'
    
    # hide data in view
    Hide(slice1, renderView1)
    
    # show color bar/color legend
    resampleToImage1Display.SetScalarBarVisibility(renderView1, True)
    
    # update the view to ensure updated data information
    renderView1.Update()
    
    
    
    
    
    # format file name
    file_name = str(time).zfill(8)
    
    
    # save data
    SaveData('/home/wrona/fault_analysis/examples/14-2D_horizon/csv/' + file_name + '.csv', proxy=resampleToImage1, ChooseArraysToWrite=1,
        PointDataArrays=['strain', 'strain_rate'])



    # delete resampleToImage1
    Delete(resampleToImage1)
    del resampleToImage1
