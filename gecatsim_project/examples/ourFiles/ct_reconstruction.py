# Copyright 2024, GE Precision HealthCare. All rights reserved. See https://github.com/xcist/main/tree/master/license

###------------ import XCIST-CatSim
import gecatsim as xc
import gecatsim.reconstruction.pyfiles.recon as recon
from gecatsim.pyfiles.CommonTools import my_path
import os
import numpy as np
import matplotlib.pyplot as plt
import json

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import phantom_definitions as pd

##--------- Initialize
my_path.add_search_path(".")

ct = xc.CatSim("../cfg/Phantom_Sample", "../cfg/Scanner_Sample_generic", "../cfg/Protocol_Sample_axial")

##--------- Make changes to parameters 
ct.resultsName = "test"
ct.protocol.viewsPerRotation = 1000
ct.protocol.viewCount = ct.protocol.viewsPerRotation
ct.protocol.stopViewId = ct.protocol.viewCount-1

ct.protocol.mA = 800
ct.scanner.detectorRowsPerMod = 1
ct.scanner.detectorRowCount = ct.scanner.detectorRowsPerMod

ct.recon.fov = 300.0
ct.recon.sliceCount = 1 
ct.recon.sliceThickness = 0.568 

size = ct.recon.imageSize if hasattr(ct.recon, "imageSize") else 512
Y, X = np.ogrid[:size, :size]

# Assume FOV = 300 mm 
fov_mm = 300
pixel_size = fov_mm / size 

# Select Phantom 
phantom = pd.generate_phantom_2(size, pixel_size, X, Y)

# ---- Point CatSim to the JSON file ----
ct.phantom.filename = "my_phantom.json"
ct.phantom.callback = "Phantom_Voxelized"
ct.phantom.projectorCallback = "C_Projector_Voxelized"
ct.phantom.centerOffset = [0.0, 0.0, 0.0]
ct.phantom.scale = 1.0


# TURN OFF ALL PHYSICS ARTIFACTS FOR PERFECT BASELINE
ct.recon.unit = 'HU'
ct.physics.monochromatic = 70         
ct.physics.enableQuantumNoise = 0     
ct.physics.enableElectronicNoise = 0  
ct.physics.scatterCallback = ""
ct.physics.scatterScaleFactor = 0
ct.physics.colSampleCount = 1
ct.physics.rowSampleCount = 1
ct.physics.srcXSampleCount = 1
ct.physics.srcYSampleCount = 1
ct.physics.viewSampleCount = 1

##--------- Run simulation
ct.run_all()  

##--------- Reconstruction
ct.do_Recon = 1
recon.recon(ct)

##--------- Show results 

imgFname = "%s_%dx%dx%d.raw" %(ct.resultsName, ct.recon.imageSize, ct.recon.imageSize, ct.recon.sliceCount)
img = xc.rawread(imgFname, [ct.recon.sliceCount, ct.recon.imageSize, ct.recon.imageSize], 'float')
plt.figure(figsize=(8,8))
plt.imshow(img[0,:,:], cmap='gray', vmin=-1000, vmax=500)
plt.colorbar(label='Hounsfield Units (HU)')
plt.title("Simulated Fan-Beam Reconstruction")
plt.axis('off')
plt.show()