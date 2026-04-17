# Copyright 2024, GE Precision HealthCare. All rights reserved.

###------------ import XCIST-CatSim
import gecatsim as xc
import gecatsim.reconstruction.pyfiles.recon as recon


##--------- Initialize
ct = xc.CatSim("../cfg/Phantom_Sample", "../cfg/Scanner_Sample_generic", "../cfg/Protocol_Sample_axial")

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

