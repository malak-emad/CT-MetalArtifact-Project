import numpy as np
import json
import gecatsim as xc
import matplotlib.pyplot as plt

def generate_phantom_1(size, pixel_size, X, Y):
    phantom = np.zeros((size, size)) 
    cx_bowl, cy_bowl = size // 2, int(size * 0.5)    
    radius_bowl = int(size * 0.48)
    bowl_mask = (X - cx_bowl)**2 + (Y - cy_bowl)**2 <= radius_bowl**2
    flat_cut = Y > int(size * 0.25)   
    phantom[bowl_mask & flat_cut] = 1  
    rod_radius_px = int((11.6 / 2) / pixel_size)
    cx, cy = int(size * 0.40), int(size * 0.55)
    phantom[(X - cx)**2 + (Y - cy)**2 <= rod_radius_px**2] = 2   
    save_json(phantom, size, pixel_size, ["ncat_water", "ncat_iron"])
    return phantom

def generate_phantom_2(size, pixel_size, X, Y):
    phantom = np.zeros((size, size))
    y_min, y_max = int(size * 0.15), int(size * 0.85)
    x_min, x_max = int(size * 0.10), int(size * 0.90)
    phantom[y_min:y_max, x_min:x_max] = 1  
    radius_px = int(3 / pixel_size)
    centers = [(int(size * 0.4), int(size * 0.4)), (int(size * 0.7), int(size * 0.4)), (int(size * 0.55), int(size * 0.6))]
    for (cx, cy) in centers:
        mask = (X - cx)**2 + (Y - cy)**2 <= radius_px**2
        phantom[mask] = 2   
    save_json(phantom, size, pixel_size, ["plexi", "Ag"])
    return phantom

def save_json(phantom, size, pixel_size, mat_names):
    (phantom == 1).astype(np.float32).tofile("material1.raw")
    (phantom == 2).astype(np.float32).tofile("material2.raw")
    vp = {
        "n_materials": 2, "mat_name": mat_names,
        "volumefractionmap_filename": ["material1.raw", "material2.raw"],
        "volumefractionmap_datatype": ["float", "float"],
        "cols": [size, size], "rows": [size, size], "slices": [1, 1],
        "x_offset": [size/2, size/2], "y_offset": [size/2, size/2],
        "z_offset": [0.5, 0.5], "x_size": [pixel_size, pixel_size], "y_size": [pixel_size, pixel_size], "z_size": [pixel_size, pixel_size]
    }
    with open("my_phantom.json", "w") as f:
        json.dump(vp, f, indent=2)

if __name__ == "__main__":
    size = 512
    Y, X = np.ogrid[:size, :size]
    pixel_size = 300.0 / size
    # Show Phantom 1
    p1 = generate_phantom_2(size, pixel_size, X, Y)
    plt.imshow(p1, cmap='gray')
    plt.title("Phantom 1")
    plt.show()