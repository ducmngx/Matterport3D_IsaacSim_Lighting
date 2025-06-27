# Matterport3D_IsaacSim_Lighting

## Automated Lighting Solution for Matterport3D Houses in Isaac Sim

For those working with the full Matterport3D dataset (90 houses), I've developed an automated approach to add lighting that might be helpful after running `IsaacLab/scripts/tools/convert_mesh.py`:

### The Challenge
When importing Matterport3D houses into Isaac Sim, the converted meshes lack proper lighting, resulting in dark, flat-looking environments. Manually adding lights to 90+ houses isn't practical.

### Solution Overview
I created an automated lighting system using the semantic data from Matterport3D's `.house` files:

1. **Extract room information** from `XXX.house` files in `/house_segmentation/`
  - Example path: `matterport3d/v1/scans/1LXtFkjw3qL/house_segmentations/1LXtFkjw3qL/house_segmentations/1LXtFkjw3qL.house`
  - Parse region/room bounding boxes and labels
  - Calculate center positions for each room/region

2. **Handle coordinate transformation** 
  - Account for the -90° X-axis rotation applied during OBJ→USD conversion
  - Transform light positions: `(x, y, z) → (x, z, -y)`

3. **Scale correction**
  - Extract mesh bounds using `UsdGeom.BBoxCache` in Isaac Sim
  - Compare with house bbox from the `.house` file (first line)
  - Calculate and apply scaling factor to match actual mesh dimensions

4. **Automated light placement**
  - Place at least one light source per room/region
  - Use room centers as base positions (typically ceiling-mounted)

### Results
This approach automatically generates reasonable lighting for all 90 houses without manual intervention. While not perfect, it provides a solid foundation that eliminates the need for tedious manual light placement in each scene.

The method leverages Matterport3D's existing semantic annotations, making it a scalable solution for large-scale experiments with the dataset.

#### Link to data organization: https://github.com/niessner/Matterport/blob/master/data_organization.md
