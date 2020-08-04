import sys

sys.path.insert(1,'/home/wrona/ParaView-5.8.0-MPI-Linux-Python3.7-64bit/lib/python3.7/site-packages/')


# trace generated using paraview version 5.8.0
#
# To ensure correct image size when batch processing, please search 
# for and uncomment the line `# renderView*.ViewSize = [*,*]`

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()



# create a new 'PVD Reader'
solutionpvd = PVDReader(FileName='/home/wrona/Desktop/aspect/09a_lk07n_JohnsSolverSetting_CFL0.5_LinThresh1e-5_MinVisc19_ElemNoiseRes_FixedComposBottom/output/solution.pvd')

# get animation scene
animationScene1 = GetAnimationScene()

# get the time-keeper
timeKeeper1 = GetTimeKeeper()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# Properties modified on solutionpvd
solutionpvd.PointArrays = ['plastic_strain']

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [1448, 810]

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


for time in range(0, 10400000, 200000):

    # Properties modified on animationScene1
    animationScene1.AnimationTime = time
    
    # Properties modified on timeKeeper1
    timeKeeper1.Time = time
    
    # create a new 'Resample To Image'
    resampleToImage1 = ResampleToImage(Input=solutionpvd)
    
    # Properties modified on resampleToImage1
    resampleToImage1.SamplingDimensions = [4000, 2000, 1]
    
    # show data in view
    resampleToImage1Display = Show(resampleToImage1, renderView1, 'UniformGridRepresentation')
    
    # trace defaults for the display properties.
    resampleToImage1Display.Representation = 'Slice'
    
    # hide data in view
    Hide(solutionpvd, renderView1)
    
    # show color bar/color legend
    resampleToImage1Display.SetScalarBarVisibility(renderView1, True)
    
    # update the view to ensure updated data information
    renderView1.Update()
    
    # get color transfer function/color map for 'plastic_strain'
    plastic_strainLUT = GetColorTransferFunction('plastic_strain')
    
    # get opacity transfer function/opacity map for 'plastic_strain'
    plastic_strainPWF = GetOpacityTransferFunction('plastic_strain')
    
    # save data
    SaveData('/home/wrona/fault_analysis/examples/12-extension_estimate/csv/' + str(time).zfill(8) + '.csv', proxy=resampleToImage1, ChooseArraysToWrite=1,
        PointDataArrays=['plastic_strain'],
        AddMetaData=0)

    # destroy resampleToImage1
    Delete(resampleToImage1)
    del resampleToImage1





















































