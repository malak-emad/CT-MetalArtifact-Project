# Copyright 2024, GE Precision HealthCare. All rights reserved. See https://github.com/xcist/main/tree/master/license

###------------ import XCIST-CatSim
import gecatsim as xc
import gecatsim.reconstruction.pyfiles.recon as recon
import os
import numpy as np
import matplotlib.pyplot as plt
import json
##--------- Initialize
#my_path = xc.pyfiles.CommonTools.my_path
# add any additional search directories
#my_path.add_search_path("my-experiments")

ct = xc.CatSim("./cfg/Phantom_Sample", "./cfg/Scanner_Sample_generic", "./cfg/Protocol_Sample_axial")  # initialization

##--------- Make changes to parameters (optional)
ct.resultsName = "test"
ct.protocol.viewsPerRotation = 1000
ct.protocol.viewCount = ct.protocol.viewsPerRotation
ct.protocol.stopViewId = ct.protocol.viewCount-1
# ct.protocol.scanTypes = [1, 0, 0, 0]  # flags for airscan, offset scan, phantom scan, prep
# ct.load_cfg("Protocol_Sample_axial", "Physics_Sample", "Recon_Sample_2d")  # new cfg overrides existing parameters

ct.protocol.mA = 800
ct.scanner.detectorRowsPerMod = 1
ct.scanner.detectorRowCount = ct.scanner.detectorRowsPerMod

ct.recon.fov = 300.0
ct.recon.sliceCount = 1        # number of slices to reconstruct
ct.recon.sliceThickness = 0.568  # reconstruction inter-slice interval (in mm)

size = ct.recon.imageSize if hasattr(ct.recon, "imageSize") else 512
Y, X = np.ogrid[:size, :size]

# Assume FOV = 300 mm (from your config)
fov_mm = 300
pixel_size = fov_mm / size   # mm per pixel


#####################################################
#   PHANTOM CODE
#######################################################

#####CHOOSE ONE PHANTOM AND UNCOMMENT IT AND UNCOMMENT ITS MAT LINE IN THE JASON DECRIPTOR #######


# ============================================================
# Phantom 1: Water bowl + iron rod 
# ============================================================

phantom = np.zeros((size, size))  # air

# Bowl: full rectangular water container with curved top
cx_bowl = size // 2
cy_bowl = int(size * 0.5)    # center at middle
radius_bowl = int(size * 0.48)

bowl_mask = (X - cx_bowl)**2 + (Y - cy_bowl)**2 <= radius_bowl**2

# Keep bottom portion only (flat top surface like a bowl)
flat_cut = Y > int(size * 0.25)   # cut the top

phantom[bowl_mask & flat_cut] = 1  # water (using 1 for volume fraction, not HU)

# ---- Iron rod (11.6 mm diameter)
rod_radius_mm = 11.6 / 2
rod_radius_px = int(rod_radius_mm / pixel_size)

cx = int(size * 0.40)
cy = int(size * 0.55)

rod_mask = (X - cx)**2 + (Y - cy)**2 <= rod_radius_px**2
phantom[rod_mask] = 2   # iron (using 2 to distinguish from water)


# ============================================================
# Phantom 2 
# ============================================================

# Uncomment this and comment Phantom 1 above to run Phantom 2

# 1. Initialize empty phantom (0 = air background)
phantom = np.zeros((size, size))

# 2.  the Plexiglas plate (a large rectangle fitting inside the FOV)
y_min, y_max = int(size * 0.15), int(size * 0.85)
x_min, x_max = int(size * 0.10), int(size * 0.90)
phantom[y_min:y_max, x_min:x_max] = 1  # 1 represents Plexiglas

# 3. the metal fillings (small cylinders in a triangle)
radius_mm = 3
radius_px = int(radius_mm / pixel_size)

# Coordinates matching the triangle pattern from your earlier image
centers = [
    (int(size * 0.4), int(size * 0.4)),  # Top left
    (int(size * 0.7), int(size * 0.4)),  # Top right
    (int(size * 0.55), int(size * 0.6))  # Bottom middle
]

for (cx, cy) in centers:
    mask = (X - cx)**2 + (Y - cy)**2 <= radius_px**2
    phantom[mask] = 2   # 2 represents Amalgam (Metal)


# ---- separate raw files per material ----
# Material 1: water/plexi (background)
mat1_vol = (phantom == 1).astype(np.float32)
mat1_vol.tofile("material1.raw")

# Material 2: iron/amalgam (fillings)
mat2_vol = (phantom == 2).astype(np.float32)
mat2_vol.tofile("material2.raw")


# ---- JSON descriptor ---- 
vp = {
    "n_materials": 2,
    # "mat_name": ["ncat_water", "ncat_iron"],  # For Phantom 1
   
    "mat_name": ["plexi", "Ag"], # For Phantom 2
    "volumefractionmap_filename": ["material1.raw", "material2.raw"],
    "volumefractionmap_datatype": ["float", "float"],
    "cols":     [size, size],
    "rows":     [size, size],
    "slices":   [1, 1],
    "x_offset": [size/2, size/2],
    "y_offset": [size/2, size/2],
    "z_offset": [0.5, 0.5],
    "x_size":   [pixel_size, pixel_size],
    "y_size":   [pixel_size, pixel_size],
    "z_size":   [pixel_size, pixel_size]
}

with open("my_phantom.json", "w") as f:
    json.dump(vp, f, indent=2)

print("JSON phantom saved!")

# ---- Point CatSim to the JSON file ----
ct.phantom.filename = "my_phantom.json"
ct.phantom.callback = "Phantom_Voxelized"
ct.phantom.projectorCallback = "C_Projector_Voxelized"
ct.phantom.centerOffset = [0.0, 0.0, 0.0]
ct.phantom.scale = 1.0

plt.imshow(phantom, cmap='gray')
plt.title("Phantom BEFORE simulation")
plt.colorbar()
plt.show()


# TURN OFF ALL PHYSICS ARTIFACTS FOR PERFECT BASELINE
ct.recon.unit = 'HU'
# 1. Turn off Beam Hardening (Forces a single 70 keV energy instead of a spectrum)
ct.physics.monochromatic = 70         
# 2. Turn off Noise (Disables both Poisson quantum noise and sensor noise)
ct.physics.enableQuantumNoise = 0     
ct.physics.enableElectronicNoise = 0  
# 3. Turn off Scatter 
ct.physics.scatterCallback = ""
ct.physics.scatterScaleFactor = 0
# 4. Turn off Exponential Edge-Gradient Effect (EEGE)
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
import matplotlib.pyplot as plt

imgFname = "%s_%dx%dx%d.raw" %(ct.resultsName, ct.recon.imageSize, ct.recon.imageSize, ct.recon.sliceCount)
img = xc.rawread(imgFname, [ct.recon.sliceCount, ct.recon.imageSize, ct.recon.imageSize], 'float')
plt.figure(figsize=(8,8))
plt.imshow(img[0,:,:], cmap='gray', vmin=-1000, vmax=500)
plt.colorbar(label='Hounsfield Units (HU)')
plt.title("Simulated Fan-Beam Reconstruction")
plt.axis('off')
plt.show()