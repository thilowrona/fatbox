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
solutionpvd = PVDReader(FileName='/home/wrona/Desktop/aspect/27c_lk22c_Markers50k_CentralBox/output/solution.pvd')

# get animation scene
animationScene1 = GetAnimationScene()

# get the time-keeper
timeKeeper1 = GetTimeKeeper()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# Properties modified on solutionpvd
solutionpvd.PointArrays = ['velocity']

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [1448, 846]

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
renderView1.CameraPosition = [225000.0, 100358.734375, 10000.0]
renderView1.CameraFocalPoint = [225000.0, 100358.734375, 0.0]

# get the material library
materialLibrary1 = GetMaterialLibrary()

# update the view to ensure updated data information
renderView1.Update()

# create a new 'PVD Reader'
particlespvd = PVDReader(FileName='/home/wrona/Desktop/aspect/27c_lk22c_Markers50k_CentralBox/output/particles.pvd')

# show data in view
particlespvdDisplay = Show(particlespvd, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
particlespvdDisplay.Representation = 'Surface'

# update the view to ensure updated data information
renderView1.Update()

# set active source
SetActiveSource(solutionpvd)

# create a new 'Resample With Dataset'
resampleWithDataset1 = ResampleWithDataset(SourceDataArrays=solutionpvd,
    DestinationMesh=particlespvd)

# show data in view
resampleWithDataset1Display = Show(resampleWithDataset1, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
resampleWithDataset1Display.Representation = 'Surface'

# hide data in view
Hide(particlespvd, renderView1)

# hide data in view
Hide(solutionpvd, renderView1)

# update the view to ensure updated data information
renderView1.Update()


times = [0,
         200000,
         400000,
         600000,
         800000,
         1000000,
         1200000,
         1403870,
         1604490,
         1803930,
         2002420,
         2204740,
         2404690,
         2604690,
         2804690,
         3004690,
         3204690,
         3403430,
         3600940,
         3801860,
         4002760,
         4200040,
         4401070,
         4601360,
         4802030,
         5002980,
         5202070,
         5402320,
         5602120,
         5800180,
         6001400,
         6200680]



for time in times:

    # Properties modified on animationScene1
    animationScene1.AnimationTime = time
    
    # Properties modified on timeKeeper1
    timeKeeper1.Time = time
    
    name = str(time).zfill(7)
    
    # save data
    SaveData('/home/wrona/fault_analysis/examples/15-2D_particles/csv/particles_from_field/' + name + '.csv', proxy=resampleWithDataset1, PointDataArrays=['velocity', 'vtkValidPointMask'],
        FieldDataArrays=['CYCLE', 'TIME'])
    
    #### saving camera placements for all active views
    
    # current camera placement for renderView1
    renderView1.InteractionMode = '2D'
    renderView1.CameraPosition = [225000.0, 100358.734375, 10000.0]
    renderView1.CameraFocalPoint = [225000.0, 100358.734375, 0.0]
    renderView1.CameraParallelScale = 246367.35896898317
    
    #### uncomment the following to render all views
    # RenderAllViews()
    # alternatively, if you want to write images, you can use SaveScreenshot(...).