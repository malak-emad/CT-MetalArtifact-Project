import gecatsim as xc
import os

# Find where gecatsim is installed
package_path = os.path.dirname(xc.__file__)
material_path = os.path.join(package_path, "material")

print("Material folder:", material_path)

# List all available materials
materials = sorted(os.listdir(material_path))
for i, m in enumerate(materials):
    print(i, m)