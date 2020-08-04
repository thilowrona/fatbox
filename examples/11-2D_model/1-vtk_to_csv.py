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
solutionpvd = PVDReader(FileName='/home/wrona/Desktop/aspect/23b_lka_NonLinThresh5e-5/output/solution.pvd')

# get animation scene
animationScene1 = GetAnimationScene()

# get the time-keeper
timeKeeper1 = GetTimeKeeper()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# Properties modified on solutionpvd
solutionpvd.PointArrays = ['plastic_yielding', 'velocity']

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [1447, 795]

# get layout
layout1 = GetLayout()

# show data in view
solutionpvdDisplay = Show(solutionpvd, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
solutionpvdDisplay.Representation = 'Surface'

# reset view to fit data
renderView1.ResetCamera()

#changing interaction mode based on data extents
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [250000.0, 80000.0, 10000.0]
renderView1.CameraFocalPoint = [250000.0, 80000.0, 0.0]

# get the material library
materialLibrary1 = GetMaterialLibrary()

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Threshold'
threshold1 = Threshold(Input=solutionpvd)

# Properties modified on threshold1
threshold1.ThresholdRange = [1.0, 1.0]

# show data in view
threshold1Display = Show(threshold1, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
threshold1Display.Representation = 'Surface'

# hide data in view
Hide(solutionpvd, renderView1)

# update the view to ensure updated data information
renderView1.Update()





# create a new 'Clip'
clip1 = Clip(Input=threshold1)

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=clip1.ClipType)

# Properties modified on clip1
clip1.ClipType = 'Box'

# Properties modified on clip1.ClipType
clip1.ClipType.Position = [150000.0, 100000.0, -1.0]
clip1.ClipType.Length = [200000.0, 64250.2805540672, 2.0]

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [1447, 795]

# get layout
layout1 = GetLayout()

# show data in view
clip1Display = Show(clip1, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
clip1Display.Representation = 'Surface'

# hide data in view
Hide(solutionpvd, renderView1)

# show color bar/color legend
clip1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()



# create a new 'Compute Derivatives'
computeDerivatives1 = ComputeDerivatives(Input=clip1)

# Properties modified on computeDerivatives1
computeDerivatives1.Scalars = ['POINTS', '']

# show data in view
computeDerivatives1Display = Show(computeDerivatives1, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
computeDerivatives1Display.Representation = 'Surface'

# hide data in view
Hide(solutionpvd, renderView1)

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Calculator'
calculator1 = Calculator(Input=computeDerivatives1)

# Properties modified on calculator1
calculator1.AttributeType = 'Cell Data'
calculator1.Function = 'VectorGradient_1-VectorGradient_3'

# show data in view
calculator1Display = Show(calculator1, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
calculator1Display.Representation = 'Surface'

# hide data in view
Hide(computeDerivatives1, renderView1)

# show color bar/color legend
calculator1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# get color transfer function/color map for 'Result'
resultLUT = GetColorTransferFunction('Result')

# get opacity transfer function/opacity map for 'Result'
resultPWF = GetOpacityTransferFunction('Result')







for time in range(0,13000000,500000):

    # Properties modified on animationScene1
    animationScene1.AnimationTime = time
    
    # Properties modified on timeKeeper1
    timeKeeper1.Time = time

    # find source
    calculator1 = FindSource('Calculator1')
    
    # create a new 'Resample To Image'
    resampleToImage1 = ResampleToImage(Input=calculator1)
    
    # Properties modified on resampleToImage1
    resampleToImage1.SamplingDimensions = [3000, 1000, 1]
    
    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')
    # uncomment following to set a specific view size
    # renderView1.ViewSize = [1447, 835]
    
    # get layout
    layout1 = GetLayout()
    
    # show data in view
    resampleToImage1Display = Show(resampleToImage1, renderView1, 'UniformGridRepresentation')
    
    # trace defaults for the display properties.
    resampleToImage1Display.Representation = 'Slice'
    
    # hide data in view
    Hide(calculator1, renderView1)
    
    # show color bar/color legend
    resampleToImage1Display.SetScalarBarVisibility(renderView1, True)
    
    # find source
    solutionpvd = FindSource('solution.pvd')
    
    # find source
    computeDerivatives1 = FindSource('ComputeDerivatives1')
    
    # update the view to ensure updated data information
    renderView1.Update()
    
    # get color transfer function/color map for 'Result'
    resultLUT = GetColorTransferFunction('Result')
    
    # get opacity transfer function/opacity map for 'Result'
    resultPWF = GetOpacityTransferFunction('Result')
    
    
    file_name = str(time).zfill(7)
    
    # save data
    SaveData('/home/wrona/fault_analysis/examples/11-2D_model/csv/time_' + file_name + '.csv', 
        proxy=resampleToImage1,
        ChooseArraysToWrite=1,
        PointDataArrays=['Result'],
        AddMetaData=0)


    # destroy resampleToImage1
    Delete(resampleToImage1)
    del resampleToImage1








































































