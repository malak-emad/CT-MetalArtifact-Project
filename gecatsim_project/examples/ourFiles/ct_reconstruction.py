# Copyright 2024, GE Precision HealthCare. All rights reserved. 
import gecatsim as xc
import gecatsim.reconstruction.pyfiles.recon as recon
from gecatsim.pyfiles.CommonTools import my_path
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
from skimage.transform import iradon
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import phantom_definitions as pd


def setup_clean_baseline(ct):
    """Applies your exact protocol changes and zero-physics baseline."""
    ct.protocol.viewsPerRotation = 1000
    ct.protocol.viewCount = ct.protocol.viewsPerRotation
    ct.protocol.stopViewId = ct.protocol.viewCount-1

    ct.protocol.mA = 800
    ct.scanner.detectorRowsPerMod = 1
    ct.scanner.detectorRowCount = ct.scanner.detectorRowsPerMod

    ct.recon.fov = 300.0
    ct.recon.sliceCount = 1  # number of slices (2D = 1)
    ct.recon.sliceThickness = 0.568 

    # Turn off all physics for a clean baseline 
    ct.recon.unit = 'HU'
    ct.physics.monochromatic = 70         # single energy in keV, or -1 for polychromatic
    ct.physics.enableQuantumNoise = 0     # 0=off, 1=on
    ct.physics.enableElectronicNoise = 0  # 0=off, 1=on
    ct.physics.scatterCallback = ""       # "" = no scatter, or "Scatter_ConvolutionModel"
    ct.physics.scatterScaleFactor = 0     # magnitude of scatter

    ct.physics.colSampleCount = 1
    ct.physics.rowSampleCount = 1
    ct.physics.srcXSampleCount = 1
    ct.physics.srcYSampleCount = 1
    ct.physics.viewSampleCount = 1
    
    return ct

def perform_recon_and_plot(ct, plot_title="Simulated Fan-Beam Reconstruction"):
    """Runs the reconstruction and plots the resulting .raw image."""
    ct.do_Recon = 1
    recon.recon(ct)

    imgFname = "%s_%dx%dx%d.raw" %(ct.resultsName, ct.recon.imageSize, ct.recon.imageSize, ct.recon.sliceCount)
    img = xc.rawread(imgFname, [ct.recon.sliceCount, ct.recon.imageSize, ct.recon.imageSize], 'float')
    
    plt.figure(figsize=(8,8))
    plt.imshow(img[0,:,:], cmap='gray', vmin=-1000, vmax=500)
    plt.colorbar(label='Hounsfield Units (HU)')
    plt.title(plot_title) 
    plt.axis('off')
    plt.show()

def generate_sinogram(ct):
    n_views = ct.protocol.viewCount
    n_rows  = ct.scanner.detectorRowCount
    n_cols  = ct.scanner.detectorColCount

    raw = np.fromfile(f"{ct.resultsName}.prep", dtype=np.float32)
    projections = raw.reshape(n_views, n_rows, n_cols)
    sinogram = projections[:, 0, :]  
    return sinogram

def plot_sinogram(sinogram, title="Fan-Beam Sinogram"):
    plt.figure(figsize=(8, 8))
    plt.imshow(sinogram, cmap='gray', aspect='auto')
    plt.colorbar(label='Log-attenuation (a.u.)')
    plt.xlabel("Detector column")
    plt.ylabel("View index")
    plt.title(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    my_path.add_search_path(".")
    ct = xc.CatSim("../cfg/Phantom_Sample", "../cfg/Scanner_Sample_generic", "../cfg/Protocol_Sample_axial")
    
    ct.resultsName = "test"
    ct = setup_clean_baseline(ct)

    size = ct.recon.imageSize if hasattr(ct.recon, "imageSize") else 512
    Y, X = np.ogrid[:size, :size]
    fov_mm = 300
    pixel_size = fov_mm / size 

    phantom = pd.generate_phantom_2(size, pixel_size, X, Y)

    ct.phantom.filename = "my_phantom.json"
    ct.phantom.callback = "Phantom_Voxelized"
    ct.phantom.projectorCallback = "C_Projector_Voxelized"
    ct.phantom.centerOffset = [0.0, 0.0, 0.0]
    ct.phantom.scale = 1.0

    ct.run_all()  
    perform_recon_and_plot(ct)

    sinogram = generate_sinogram(ct)
    plot_sinogram(sinogram)
