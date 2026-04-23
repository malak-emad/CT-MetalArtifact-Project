import numpy as np
import matplotlib.pyplot as plt
import gecatsim as xc

# Test reading and displaying just one image to verify visualization works
bh_img = xc.rawread("bh_poly_p1_512x512x1.raw", [1, 512, 512], 'float')[0, :, :]

plt.figure(figsize=(10, 8))
plt.imshow(bh_img, cmap='gray', vmin=-1000, vmax=500)
plt.title("Beam Hardening - Phantom 1")
plt.colorbar(label='HU')
plt.axis('off')
plt.show()

print(f"Image shape: {bh_img.shape}")
print(f"Image stats - Min: {bh_img.min():.1f}, Max: {bh_img.max():.1f}, Mean: {bh_img.mean():.1f}")