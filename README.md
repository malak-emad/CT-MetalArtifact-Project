# CT-MetalArtifact-Project
Fan-beam detector configuration for CT metal artifact simulation (De Man et al. 1999)

# How to Use
1. Install XCIST: `pip install gecatsim`
2. Open your terminal and navigate to our working directory:
   ```bash
   cd gecatsim_project/examples/ourFiles
    python main.py

### Physics-Based Artifacts

**Aliasing Artifact**
![Aliasing](gecatsim_project/examples/ourFiles/readme%20images/aliasing.png)
*Under-sampling of detectors and views creates geometric patterns.*

---

**Beam Hardening**
![Beam Hardening](gecatsim_project/examples/ourFiles/readme%20images/beam%20hardening.png)
*The spectral shift towards higher average energy causes dark streaks and cupping.*

---

**Scatter Artifact**
![Scatter](gecatsim_project/examples/ourFiles/readme%20images/scatter.png)
*Additive background signals from scattered photons distort image intensity.*

---

### Motion & Combined Effects

**Motion Artifact**
![Motion](gecatsim_project/examples/ourFiles/readme%20images/motion.png)
*Object inconsistency during the scanning process leads to directional streaks.*

---

**Combined Artifacts**
![Combined](gecatsim_project/examples/ourFiles/readme%20images/combined%20artifact.png)
*Multiple artifacts (Noise, Scatter, Beam Hardening ..etc).*

---

### Correction Results

**NMAR Correction**
![NMAR](gecatsim_project/examples/ourFiles/readme%20images/Nmar.png)
*The final result after applying Normalization-based Metal Artifact Reduction.*

---
# CT Metal Artifact Simulation - Detector Configuration

## My Work (Malak)
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

## Verification 
  **Detector fan beam geometry**
  - Detector: 672 channels at 1.2mm pitch
  - Focal spot: 0.6mm
  - Fan-beam geometry: `Detector_ThirdgenCurved`
  - Polychromatic with 5 energy bins

## References
De Man, B., et al. (1999). "Metal streak artifacts in X-ray computed tomography: a simulation study." IEEE Trans. Nucl. Sci., 46(3):691-696.
