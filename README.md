# CT-MetalArtifact-Project
Fan-beam detector configuration for CT metal artifact simulation (De Man et al. 1999)
# CT Metal Artifact Simulation - Detector Configuration

## My Work (Your Name)
Configured the fan-beam detector geometry to match De Man et al. (1999) paper.

## Configuration Changes

| Parameter | Original Value | New Value |
|-----------|---------------|-----------|
| Detector columns | 900 | **672** |
| Detector pitch | 1.0 mm | **1.2 mm** |
| Focal spot width | 1.0 mm | **0.6 mm** |
| Focal spot length | 1.0 mm | **0.6 mm** |
| Energy bins | 20 | **5** |

## Files Modified
  **Detector fan beam geometry**
  - `Scanner_Sample_generic.cfg` - Detector geometry settings
  - `Physics_Sample.cfg` - Physics settings (polychromatic spectrum)
  - `Protocol_Sample_axial.cfg` - View and Flux "did not change them just made sure they are right"
  - `Sim_Recon_Sample.py` - Changed it to make sure my configuration is working
  **add your parts here**

## How to Use
1. Install XCIST: `pip install gecatsim`
2. Copy these `.cfg` files to `gecatsim/examples/cfg/`
3. Run simulation: `python Sim_Recon_Sample.py`

## Verification 
  **Detector fan beam geometry**
  - Detector: 672 channels at 1.2mm pitch
  - Focal spot: 0.6mm
  - Fan-beam geometry: `Detector_ThirdgenCurved`
  - Polychromatic with 5 energy bins
  **add your parts here**

## References
De Man, B., et al. (1999). "Metal streak artifacts in X-ray computed tomography: a simulation study." IEEE Trans. Nucl. Sci., 46(3):691-696.
