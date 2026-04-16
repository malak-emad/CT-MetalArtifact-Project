# Copyright 2024, GE Precision HealthCare. All rights reserved.

###------------ import XCIST-CatSim
import gecatsim as xc
import gecatsim.reconstruction.pyfiles.recon as recon
import numpy as np 
import matplotlib.pyplot as plt
import os
##--------- Initialize
ct = xc.CatSim("./cfg/Phantom_Sample", "./cfg/Scanner_Sample_generic", "./cfg/Protocol_Sample_axial")

# Verify your settings
print("=" * 50)
print("YOUR DETECTOR CONFIGURATION:")
print(f"  Detector columns: {ct.scanner.detectorColCount}")
print(f"  Detector pitch: {ct.scanner.detectorColSize} mm")
print(f"  Focal spot: {ct.scanner.focalspotWidth} mm")
print(f"  Energy count: {ct.physics.energyCount}")
print("=" * 50)

##--------- Make changes to parameters (optional)
ct.resultsName = "test"

# Comment out or remove these overriding lines:
# ct.protocol.viewsPerRotation = 500
# ct.protocol.viewCount = ct.protocol.viewsPerRotation
# ct.protocol.stopViewId = ct.protocol.viewCount-1
# ct.protocol.mA = 800
# ct.scanner.detectorRowsPerMod = 4
# ct.scanner.detectorRowCount = ct.scanner.detectorRowsPerMod
# ct.recon.fov = 300.0
# ct.recon.sliceCount = 4
# ct.recon.sliceThickness = 0.568



size = ct.recon.imageSize if hasattr(ct.recon, "imageSize") else 512
Y, X = np.ogrid[:size, :size]

# Assume FOV = 300 mm (from your config)
fov_mm = 300
pixel_size = fov_mm / size   # mm per pixel

# ============================================================
# Phantom 1: Water bowl + iron rod 
# ============================================================

phantom = np.zeros((size, size))  # air

# ---- Bowl geometry (circular bottom + flat top)
cx_bowl = size // 2
cy_bowl = int(size * 1.2)   # center below image
radius_bowl = int(size * 0.9)

bowl_mask = (X - cx_bowl)**2 + (Y - cy_bowl)**2 <= radius_bowl**2

# flat cut (top surface)
flat_cut = Y < int(size * 0.75)

phantom[bowl_mask & flat_cut] = 183     # water

# ---- Iron rod (11.6 mm diameter)
rod_radius_mm = 11.6 / 2
rod_radius_px = int(rod_radius_mm / pixel_size)

cx = int(size * 0.65)
cy = int(size * 0.55)

rod_mask = (X - cx)**2 + (Y - cy)**2 <= rod_radius_px**2
phantom[rod_mask] = 155   # iron


# ============================================================
# Phantom 2 
# ============================================================
"""
phantom = np.ones((size, size)) * 174   # plexiglas

# metal fillings (small cylinders)
radius_mm = 3
radius_px = int(radius_mm / pixel_size)

centers = [
    (int(size*0.4), int(size*0.4)),
    (int(size*0.6), int(size*0.6)),
    (int(size*0.7), int(size*0.4))
]

for (cx, cy) in centers:
    mask = (X - cx)**2 + (Y - cy)**2 <= radius_px**2
    phantom[mask] = 128   # brass as amalgam substitute (similar density)   
"""

ct.phantom.callback = "Phantom_Voxelized"
ct.phantom.projectorCallback = "C_Projector_Voxelized"
ct.phantom.filename = os.path.abspath("my_phantom.raw")
ct.phantom.dimensions = [size, size, 1]
ct.phantom.voxelSize = [pixel_size, pixel_size, pixel_size]
ct.phantom.centerOffset = [0.0, 0.0, 0.0]
ct.phantom.scale = 1.0  # no scaling

plt.imshow(phantom, cmap='gray')
plt.title("Phantom BEFORE simulation")
plt.colorbar()
plt.show()
phantom.astype('int16').tofile("my_phantom.raw")
##--------- Run simulation
# Save phantom BEFORE ct.run_all(), with explicit path
phantom_path = os.path.abspath("my_phantom.raw")
phantom.astype('int16').tofile(phantom_path)

# Verify the file was written
print(f"Phantom saved to: {phantom_path}")
print(f"File exists: {os.path.exists(phantom_path)}")
print(f"File size: {os.path.getsize(phantom_path)} bytes")

ct.phantom.filename = phantom_path

ct.run_all()

##--------- Reconstruction
ct.do_Recon = 1
recon.recon(ct)

##--------- Show results
import matplotlib.pyplot as plt
import numpy as np

imgFname = "%s_%dx%dx%d.raw" %(ct.resultsName, ct.recon.imageSize, ct.recon.imageSize, ct.recon.sliceCount)
img = xc.rawread(imgFname, [ct.recon.sliceCount, ct.recon.imageSize, ct.recon.imageSize], 'float')

# Show the first slice (index 0) instead of index 2
plt.figure(figsize=(10, 10))
plt.imshow(img[0,:,:], cmap='gray', vmin=-200, vmax=200)
plt.colorbar(label='HU')
plt.title('CT Reconstruction - Fan Beam Detector (672 channels, 1.2mm pitch)')
plt.show()
