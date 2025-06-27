#!/usr/bin/env python3
"""
Combined USD Loader with Matterport-based Lighting
==================================================

This script:
1. Loads a USD file (Matterport house) into Isaac Sim
2. Extracts room information from the corresponding .house file
3. Places lights in each room based on Matterport room segmentation data
4. Provides intelligent lighting based on room types

Usage:
    python combined_usd_lighting.py --house_name 1LXtFkjw3qL
    python combined_usd_lighting.py --house_name 1LXtFkjw3qL --usd_path /custom/path/to/house.usd
"""

from omni.isaac.kit import SimulationApp

# Launch Isaac Sim with GUI
simulation_app = SimulationApp({
    "headless": False,
    "width": 1920,
    "height": 1080
})

import omni.usd
from pxr import UsdGeom, UsdLux, Gf, Usd
import numpy as np
import time
import os
import argparse
import json
from typing import Dict, List

class MatterportLightingSystem:
    """Combined system for loading USD and adding Matterport-based lighting"""
    
    def __init__(self):
        self.matterport_base_path = "/home/aaron/matterport3d/v1/scans"
        
        # Room type mapping
        self.room_type_mapping = {
            'a': 'bathroom',
            'b': 'bedroom', 
            'c': 'closet',
            'd': 'dining_room',
            'e': 'entryway',
            'f': 'family_room',
            'g': 'garage',
            'h': 'hallway',
            'i': 'library',
            'j': 'laundry_room',
            'k': 'kitchen',
            'l': 'living_room',
            'm': 'meeting_room',
            'n': 'lounge',
            'o': 'office',
            'p': 'porch',
            'r': 'recreation',
            's': 'stairs',
            't': 'toilet',
            'u': 'utility_room',
            'v': 'tv_room',
            'w': 'workout',
            'x': 'outdoor',
            'y': 'balcony',
            'z': 'other',
            'B': 'bar',
            'C': 'classroom',
            'D': 'dining_booth',
            'S': 'spa',
            'Z': 'junk',
            '-': 'unlabeled'
        }
        
        # Light intensity mapping based on room type
        self.room_intensities = {
            'kitchen': 15000,      # Bright task lighting
            'bathroom': 10000,     # Good visibility
            'office': 15000,       # Task lighting
            'dining_room': 15000,  # Ambient dining
            'living_room': 12000,  # Comfortable ambient
            'family_room': 12000,  # Comfortable ambient
            'bedroom': 8000,       # Soft ambient
            'hallway': 6000,       # Wayfinding
            'stairs': 8000,        # Safety lighting
            'closet': 6000,       # Utility
            'garage': 10000,       # Utility
            'laundry_room': 10000, # Task lighting
            'utility_room': 10000, # Task lighting
            'library': 10000,      # Reading
            'meeting_room': 10000, # Conference
            'lounge': 10000,       # Relaxed
            'recreation': 12000,   # Activity
            'bar': 8000,           # Ambient
            'spa': 6000,           # Relaxing
            'entryway': 10000,     # Welcome
            'other': 10000,        # Default
            'unlabeled': 8000      # Conservative
        }
    
    def find_house_files(self, house_name: str) -> Dict[str, str]:
        """Find USD and .house files for a given house"""
        
        files = {
            'usd_file': None,
            'house_file': None
        }
        
        house_dir = os.path.join(self.matterport_base_path, house_name)
        
        # Find .house file
        house_seg_path = os.path.join(house_dir, "house_segmentations", house_name, "house_segmentations")
        house_file = os.path.join(house_seg_path, f"{house_name}.house")
        
        if os.path.exists(house_file):
            files['house_file'] = house_file
        
        # Find USD file (check common locations)
        usd_candidates = [
            os.path.join(house_dir, f"{house_name}_corrected.usd"),
            os.path.join(house_dir, f"{house_name}.usd"),
            os.path.join(house_dir, "matterport_mesh", f"{house_name}.usd"),
        ]
        
        # Also check for any USD files in matterport_mesh subdirectories
        mesh_dir = os.path.join(house_dir, "matterport_mesh")
        if os.path.exists(mesh_dir):
            for subdir in os.listdir(mesh_dir):
                subdir_path = os.path.join(mesh_dir, subdir)
                if os.path.isdir(subdir_path):
                    for file in os.listdir(subdir_path):
                        if file.endswith('.usd'):
                            usd_candidates.append(os.path.join(subdir_path, file))
        
        for usd_path in usd_candidates:
            if os.path.exists(usd_path):
                files['usd_file'] = usd_path
                break
        
        return files
    
    def parse_house_file(self, house_file_path: str) -> Dict:
        """Parse .house file and extract region data"""
        
        house_data = {
            'house_name': os.path.basename(house_file_path).replace('.house', ''),
            'header': None,
            'levels': [],
            'regions': []
        }
        
        print(f"📖 Reading house file: {house_file_path}")
        
        with open(house_file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split()
                if len(parts) == 0:
                    continue
                    
                line_type = parts[0]
                
                try:
                    if line_type == 'H':
                        # Header line
                        if len(parts) >= 24:
                            house_data['header'] = {
                                'name': parts[1],
                                'num_regions': int(parts[10]),
                                'num_levels': int(parts[12]),
                                'bbox_min': [float(parts[18]), float(parts[19]), float(parts[20])],
                                'bbox_max': [float(parts[21]), float(parts[22]), float(parts[23])]
                            }
                            
                    elif line_type == 'L':
                        # Level line
                        if len(parts) >= 13:
                            level_data = {
                                'level_index': int(parts[1]),
                                'num_regions': int(parts[2]),
                                'label': parts[3],
                                'center': [float(parts[4]), float(parts[5]), float(parts[6])],
                                'bbox_min': [float(parts[7]), float(parts[8]), float(parts[9])],
                                'bbox_max': [float(parts[10]), float(parts[11]), float(parts[12])]
                            }
                            house_data['levels'].append(level_data)
                            
                    elif line_type == 'R':
                        # Region line
                        if len(parts) >= 16:
                            region_data = {
                                'region_index': int(parts[1]),
                                'level_index': int(parts[2]),
                                'label': parts[5],
                                'center': [float(parts[6]), float(parts[7]), float(parts[8])],
                                'bbox_min': [float(parts[9]), float(parts[10]), float(parts[11])],
                                'bbox_max': [float(parts[12]), float(parts[13]), float(parts[14])],
                                'height': float(parts[15])
                            }
                            house_data['regions'].append(region_data)
                            
                except (ValueError, IndexError) as e:
                    print(f"⚠️  Warning: Error parsing line {line_num}: {e}")
                    continue
        
        print(f"✅ Parsed {len(house_data['regions'])} regions across {len(house_data['levels'])} levels")
        return house_data
    
    def calculate_light_position(self, region: Dict) -> List[float]:
        """Calculate optimal light position for a region"""
        
        # Use region center as base
        center_x, center_y, center_z = region['center']
        
        # Calculate floor and ceiling levels
        floor_z = region['bbox_min'][2]
        ceiling_z = region['bbox_max'][2]
        room_height = ceiling_z - floor_z
        
        # Determine light height based on room type and height
        if region['label'] == 's':  # stairs
            light_height = floor_z + room_height * 0.4
        elif region['label'] in ['c', 't']:  # closets, toilets
            light_height = floor_z + min(2.2, room_height * 0.8)
        elif room_height > 4.0:  # High ceiling rooms
            light_height = floor_z + room_height * 0.6
        else:  # Normal rooms
            light_height = floor_z + min(2.7, room_height * 0.8)
        
        return [center_x, center_y, light_height]
    
    def extract_light_positions(self, house_data: Dict) -> Dict:
        """Extract all light positions from house data"""
        
        light_positions = {
            'house_name': house_data['house_name'],
            'total_lights': 0,
            'levels': {},
            'all_positions': []
        }
        
        # Process each region
        for region in house_data['regions']:
            # Skip junk regions
            if region['label'] == 'Z':
                continue
            
            level_idx = region['level_index']
            
            # Initialize level if needed
            if level_idx not in light_positions['levels']:
                light_positions['levels'][level_idx] = {
                    'level_name': f"Level_{level_idx}",
                    'lights': []
                }
            
            # Find level name
            for level in house_data['levels']:
                if level['level_index'] == level_idx:
                    light_positions['levels'][level_idx]['level_name'] = level['label']
                    break
            
            # Calculate light position
            light_pos = self.calculate_light_position(region)
            room_type = self.room_type_mapping.get(region['label'], 'other')
            intensity = self.room_intensities.get(room_type, 1000)
            
            light_info = {
                'region_index': region['region_index'],
                'room_label': region['label'],
                'room_type': room_type,
                'position': light_pos,
                'intensity': intensity,
                'room_center': region['center'],
                'room_height': region['height'],
                'floor_z': region['bbox_min'][2],
                'ceiling_z': region['bbox_max'][2]
            }
            
            light_positions['levels'][level_idx]['lights'].append(light_info)
            light_positions['all_positions'].append(light_info)
            light_positions['total_lights'] += 1
        
        return light_positions
    
    def load_usd_file(self, usd_file_path: str) -> bool:
        """Load USD file into Isaac Sim"""
        
        print(f"📁 Loading USD file: {usd_file_path}")
        
        try:
            # Load the USD file
            success = omni.usd.get_context().open_stage(usd_file_path)
            if not success:
                print("❌ Failed to load USD file")
                return False
            
            stage = omni.usd.get_context().get_stage()
            if not stage:
                print("❌ No stage available after loading")
                return False
            
            print("✅ USD file loaded successfully")
            
            # Give it a moment to load
            time.sleep(2)
            return True
            
        except Exception as e:
            print(f"❌ Error loading USD file: {e}")
            return False
    
    def get_house_mesh_transform(self) -> Dict:
        """Get the transform of the loaded house mesh"""
        
        stage = omni.usd.get_context().get_stage()
        
        # Find the house mesh prim (look for common mesh locations)
        house_prim = None
        mesh_transform = None
        
        # Search for mesh prims in the stage
        mesh_candidates = []
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Xform):
                # Skip lighting prims
                if "lighting" not in prim_path.lower() and "light" not in prim_path.lower():
                    mesh_candidates.append(prim)
        
        if mesh_candidates:
            # print(f"{mesh_candidates} Mesh candidates found")
            # Use the first significant mesh found
            house_prim = mesh_candidates[1]  # The first one is a XForm that wraps the whole house, not the actual mesh
            
            # Get transform matrix
            if house_prim.IsA(UsdGeom.Xformable):
                xformable = UsdGeom.Xformable(house_prim)
                transform_matrix = xformable.ComputeLocalToWorldTransform(0)  # At time 0
            else:
                # If not xformable, assume identity transform
                transform_matrix = Gf.Matrix4d().SetIdentity()
            
            # Get bounding box with proper USD API
            try:
                # Create BBoxCache with required parameters
                time_code = Usd.TimeCode.Default()
                purposes = [UsdGeom.Tokens.default_]
                bbox_cache = UsdGeom.BBoxCache(time_code, purposes)
                
                bbox = bbox_cache.ComputeWorldBound(house_prim)
                bbox_range = bbox.GetRange()
                
                mesh_transform = {
                    'prim_path': str(house_prim.GetPath()),
                    'transform_matrix': transform_matrix,
                    'center': bbox.ComputeCentroid(),
                    'bbox_min': bbox_range.GetMin(),
                    'bbox_max': bbox_range.GetMax(),
                    'size': bbox_range.GetSize()
                }
                
            except Exception as bbox_error:
                print(f"⚠️  Could not compute bounding box: {bbox_error}")
                # Fallback: just use transform without bbox
                mesh_transform = {
                    'prim_path': str(house_prim.GetPath()),
                    'transform_matrix': transform_matrix,
                    'center': Gf.Vec3d(0, 0, 0),
                    'bbox_min': Gf.Vec3d(-10, -10, -1),
                    'bbox_max': Gf.Vec3d(10, 10, 5),
                    'size': Gf.Vec3d(20, 20, 6)
                }
            
            print(f"🏠 Found house mesh: {mesh_transform['prim_path']}")
            print(f"📏 House bounding box:")
            print(f"   Min: ({mesh_transform['bbox_min'][0]:.1f}, {mesh_transform['bbox_min'][1]:.1f}, {mesh_transform['bbox_min'][2]:.1f})")
            print(f"   Max: ({mesh_transform['bbox_max'][0]:.1f}, {mesh_transform['bbox_max'][1]:.1f}, {mesh_transform['bbox_max'][2]:.1f})")
            print(f"   Center: ({mesh_transform['center'][0]:.1f}, {mesh_transform['center'][1]:.1f}, {mesh_transform['center'][2]:.1f})")
        
        return mesh_transform
    
    def scale_light_positions(self, light_positions: Dict, mesh_transform: Dict, house_header: Dict) -> Dict:
        # House file bounds and center (original coordinates)
        house_min = house_header['bbox_min']  # [x_min, y_min, z_min]
        house_max = house_header['bbox_max']  # [x_max, y_max, z_max]
        house_center = (
            (house_max[0] + house_min[0]) / 2,  
            (house_max[1] + house_min[1]) / 2,  
            (house_max[2] + house_min[2]) / 2   
        )
        
        # Step 1: Apply -90° X rotation to house coordinates first
        # -90° X rotation: (x, y, z) -> (x, z, -y)
        # rotated_x = house_x
        # rotated_y = house_z
        # rotated_z = -house_y
        
        # Also rotate the house center
        rotated_house_center = (
            house_center[0],   # x stays same
            house_center[2],   # y becomes z
            -house_center[1]   # z becomes -y
        )
        
        # Isaac Sim bounds and center (already in rotated space from your measurement)
        isaac_min = mesh_transform['bbox_min']  
        isaac_max = mesh_transform['bbox_max']  
        isaac_center = (
            (isaac_max[0] + isaac_min[0]) / 2, 
            (isaac_max[1] + isaac_min[1]) / 2,  
            (isaac_max[2] + isaac_min[2]) / 2   
        )
        
        # Calculate scale factors (using rotated house bounds)
        rotated_house_size = (
            house_max[0] - house_min[0],  # x size unchanged
            house_max[2] - house_min[2],  # y size from original z
            house_max[1] - house_min[1]   # z size from original y
        )
        isaac_size = (isaac_max[0] - isaac_min[0], isaac_max[1] - isaac_min[1], isaac_max[2] - isaac_min[2])
        scale_factors = (isaac_size[0] / rotated_house_size[0], isaac_size[1] / rotated_house_size[1], isaac_size[2] / rotated_house_size[2])

        transformed_positions = light_positions.copy()
        # Transform each light position in levels: 90° X rotation (x, y, z) -> (x, -z, y)
        for level_idx, level_data in transformed_positions['levels'].items():
            for light_info in level_data['lights']:
                original_pos = light_info['position']

                rel_x = original_pos[0] - rotated_house_center[0]
                rel_y = original_pos[1] - rotated_house_center[1] 
                rel_z = original_pos[2] - rotated_house_center[2]
                
                # Step 3: Scale
                scaled_x = rel_x * scale_factors[0]
                scaled_y = rel_y * scale_factors[1]
                scaled_z = rel_z * scale_factors[2]
                
                # Step 4: Translate to Isaac Sim center
                isaac_x = scaled_x + isaac_center[0]
                isaac_y = scaled_y + isaac_center[1]
                isaac_z = scaled_z + isaac_center[2]

                transformed_pos = [isaac_x, isaac_y, isaac_z]
                # Update light position
                light_info['position'] = transformed_pos
        
        # Rebuild all_positions from the already transformed levels data
        transformed_positions['all_positions'] = []
        for level_data in transformed_positions['levels'].values():
            transformed_positions['all_positions'].extend(level_data['lights'])
        
        return transformed_positions #(isaac_x, isaac_y, isaac_z)
        
    
    def transform_light_positions(self, light_positions: Dict, mesh_transform: Dict) -> Dict:
        """Transform light positions to match house mesh coordinate system"""
        
        # if not mesh_transform:
        #     print("⚠️  No mesh transform found - applying standard 90° X rotation for Matterport houses")
        #     # Apply standard Matterport house rotation: 90° around Z-axis
        #     return self.apply_matterport_rotation(light_positions)
        
        # print("🔄 Transforming light positions to match house mesh...")
        
        # # For Matterport houses, we typically need a 90° X rotation
        # # Based on the house transform showing 90° rotation
        transformed_positions = self.apply_matterport_rotation(light_positions)
        
        print("✅ Light positions transformed with Matterport -90° X rotation")
        return transformed_positions

    def apply_matterport_rotation(self, light_positions: Dict) -> Dict:
        """Apply standard Matterport house 90° X rotation to light positions"""
        transformed_positions = light_positions.copy()
        
        # Transform each light position in levels: 90° X rotation (x, y, z) -> (x, -z, y)
        for level_idx, level_data in transformed_positions['levels'].items():
            for light_info in level_data['lights']:
                original_pos = light_info['position']
                # 90° counterclockwise rotation around X-axis
                new_x = original_pos[0]
                new_y = original_pos[2]
                new_z = -original_pos[1]
                transformed_pos = [new_x, new_y, new_z]
                # Update light position
                light_info['position'] = transformed_pos
                light_info['original_position'] = original_pos
        
        # Rebuild all_positions from the already transformed levels data
        transformed_positions['all_positions'] = []
        for level_data in transformed_positions['levels'].values():
            transformed_positions['all_positions'].extend(level_data['lights'])
        
        return transformed_positions
    
    # def apply_manual_offset(self, light_positions: Dict, offset: List[float]) -> Dict:
    #     """Apply a simple manual offset to all light positions"""
        
    #     print(f"🔧 Applying manual offset: ({offset[0]}, {offset[1]}, {offset[2]})")
        
    #     transformed_positions = light_positions.copy()
        
    #     # Apply offset to each light position
    #     for level_idx, level_data in transformed_positions['levels'].items():
    #         for light_info in level_data['lights']:
    #             original_pos = light_info['position']
                
    #             new_pos = [
    #                 original_pos[0] + offset[0],
    #                 original_pos[1] + offset[1], 
    #                 original_pos[2] + offset[2]
    #             ]
                
    #             light_info['position'] = new_pos
    #             light_info['original_position'] = original_pos
        
    #     # Also update all_positions list
    #     for light_info in transformed_positions['all_positions']:
    #         original_pos = light_info['position']
            
    #         new_pos = [
    #             original_pos[0] + offset[0],
    #             original_pos[1] + offset[1],
    #             original_pos[2] + offset[2]
    #         ]
            
    #         light_info['position'] = new_pos
    #         light_info['original_position'] = original_pos
        
    #     print("✅ Applied manual offset to all light positions")
    #     return transformed_positions
    
    def create_matterport_lights(self, light_positions: Dict, mesh_transform: Dict = None) -> int:
        """Create lights in Isaac Sim based on Matterport room data"""
        
        stage = omni.usd.get_context().get_stage()
        total_lights = 0
        
        # Find the house mesh prim to put lights inside it
        house_prim_path = None
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Xform):
                # Skip lighting prims and find the main house mesh
                if ("lighting" not in prim_path.lower() and 
                    "light" not in prim_path.lower() and
                    "/World/" in prim_path and
                    prim_path != "/World"):
                    house_prim_path = prim_path
                    break
        
        if not house_prim_path:
            house_prim_path = "/World/House"  # Fallback

        print(house_prim_path + "Check")
        
        # Create lighting group INSIDE the house mesh
        lights_path = f"{house_prim_path}/MatterportLighting"
        if not stage.GetPrimAtPath(lights_path):
            UsdGeom.Xform.Define(stage, lights_path)
        
        print(f"💡 Creating {light_positions['total_lights']} room-based lights inside {house_prim_path}...")
        
        # Process each level
        for level_idx, level_data in light_positions['levels'].items():
            level_name = level_data['level_name']
            level_lights_path = f"{lights_path}/Level_{level_idx}"
            
            if not stage.GetPrimAtPath(level_lights_path):
                UsdGeom.Xform.Define(stage, level_lights_path)
            
            print(f"  🏠 Level {level_idx} ({level_name}): {len(level_data['lights'])} rooms")
            
            # Create lights for each room
            for light_info in level_data['lights']:
                try:
                    region_idx = light_info['region_index']
                    room_type = light_info['room_type']
                    position = light_info['position']
                    intensity = light_info['intensity']
                    
                    # Create light
                    light_path = f"{level_lights_path}/{room_type}_{region_idx:03d}"
                    light_prim = UsdLux.SphereLight.Define(stage, light_path)
                    
                    # Set light properties
                    light_prim.CreateIntensityAttr(float(intensity))
                    light_prim.CreateRadiusAttr(0.1)
                    
                    # Color based on room type
                    if room_type in ['kitchen', 'bathroom', 'office']:
                        color = Gf.Vec3f(1.0, 1.0, 0.95)  # Cool white
                    elif room_type in ['bedroom', 'living_room', 'family_room']:
                        color = Gf.Vec3f(1.0, 0.95, 0.85)  # Warm white
                    else:
                        color = Gf.Vec3f(1.0, 0.98, 0.9)   # Neutral white
                    
                    light_prim.CreateColorAttr(color)
                    
                    # Position the light (now relative to house)
                    xform = UsdGeom.Xformable(light_prim)
                    translate_op = xform.AddTranslateOp()
                    translate_op.Set(Gf.Vec3d(position[0], position[1], position[2]))
                    
                    total_lights += 1
                    
                    if total_lights <= 5:  # Show details for first few lights
                        if 'original_position' in light_info:
                            orig = light_info['original_position']
                            print(f"    ✨ {room_type:12} -> ({position[0]:6.1f}, {position[1]:6.1f}, {position[2]:6.1f}) [was ({orig[0]:6.1f}, {orig[1]:6.1f}, {orig[2]:6.1f})] [{intensity} lumens]")
                        else:
                            print(f"    ✨ {room_type:12} -> ({position[0]:6.1f}, {position[1]:6.1f}, {position[2]:6.1f}) [{intensity} lumens]")
                    
                except Exception as e:
                    print(f"    ❌ Failed to create light for region {region_idx}: {e}")
        
        print(f"✅ Created {total_lights} lights total inside house at {house_prim_path}")
        return total_lights
    
    def add_ambient_lighting(self) -> int:
        """Add ambient and fill lighting"""
        
        stage = omni.usd.get_context().get_stage()
        lights_added = 0
        
        try:
            # Add dome light for ambient illumination
            dome_path = "/World/MatterportLighting/AmbientDome"
            dome_prim = UsdLux.DomeLight.Define(stage, dome_path)
            dome_prim.CreateIntensityAttr(300.0)
            dome_prim.CreateColorAttr(Gf.Vec3f(0.9, 0.95, 1.0))  # Cool ambient
            lights_added += 1
            
            # Add key light for overall scene illumination
            key_path = "/World/MatterportLighting/KeyLight"
            key_prim = UsdLux.DistantLight.Define(stage, key_path)
            key_prim.CreateIntensityAttr(1000.0)
            key_prim.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 0.95))
            
            # Angle the key light
            xform = UsdGeom.Xformable(key_prim)
            rotate_op = xform.AddRotateXYZOp()
            rotate_op.Set(Gf.Vec3f(-30, 45, 0))
            lights_added += 1
            
            print(f"✅ Added {lights_added} ambient lights")
            
        except Exception as e:
            print(f"❌ Failed to add ambient lighting: {e}")
        
        return lights_added
    
    def process_house(self, house_name: str, custom_usd_path: str = None) -> bool:
        """Complete processing pipeline for a house"""
        
        print(f"🏠 Processing house: {house_name}")
        print("=" * 60)
        
        # Find files
        if custom_usd_path:
            self.mesh_name = custom_usd_path.split('.')[0]
            full_usd_path = os.path.join(self.matterport_base_path, house_name, "matterport_mesh", 
                                         house_name, "matterport_mesh", custom_usd_path)
            files = {'usd_file': full_usd_path, 'house_file': None}
            # Still try to find .house file
            house_dir = os.path.join(self.matterport_base_path, house_name)
            house_seg_path = os.path.join(house_dir, "house_segmentations", house_name, "house_segmentations")
            house_file = os.path.join(house_seg_path, f"{house_name}.house")
            if os.path.exists(house_file):
                files['house_file'] = house_file
        else:
            files = self.find_house_files(house_name)
        
        print(f"📁 USD file: {files['usd_file']}")
        print(f"📖 House file: {files['house_file']}")
        
        # Load USD file
        if not files['usd_file']:
            print("❌ No USD file found")
            return False
        
        if not self.load_usd_file(files['usd_file']):
            return False
        
        # Process lighting if .house file is available
        if files['house_file']:
            print("\n💡 Processing Matterport lighting data...")
            
            # Parse house file
            house_data = self.parse_house_file(files['house_file'])
            
            # Extract light positions
            light_positions = self.extract_light_positions(house_data)
            
            # Get house mesh transform to align coordinates
            print("\n🔍 Analyzing house mesh coordinate system...")
            mesh_transform = self.get_house_mesh_transform()
            
            # Transform light positions to match house orientation
            print("\n🔄 Applying coordinate transformation for house orientation...")
            # Since house has 90° X rotation, apply same to light positions
            light_positions = self.apply_matterport_rotation(light_positions)

            scaled_light_positions = self.scale_light_positions(light_positions, mesh_transform, house_data['header'])
            
            # Create lights
            room_lights = self.create_matterport_lights(scaled_light_positions, mesh_transform)
            ambient_lights = self.add_ambient_lighting()
            
            print(f"\n✨ Lighting Summary:")
            print(f"   Room-based lights: {room_lights}")
            print(f"   Ambient lights: {ambient_lights}")
            print(f"   Total lights: {room_lights + ambient_lights}")
            
            # Show coordinate system info
            if mesh_transform:
                print(f"\n🌐 Coordinate System:")
                print(f"   House mesh center: ({mesh_transform['center'][0]:.1f}, {mesh_transform['center'][1]:.1f}, {mesh_transform['center'][2]:.1f})")
                print(f"   Lights positioned relative to house mesh")
            else:
                print(f"\n⚠️  Using original coordinate system (no transform applied)")
            
        else:
            print("\n⚠️  No .house file found - adding basic lighting only")
            ambient_lights = self.add_ambient_lighting()
            print(f"✨ Added {ambient_lights} basic lights")
        
        return True


def main():
    """Main function"""
    
    print("🚀 Starting Matterport USD Lighting System")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(description="Load Matterport USD with intelligent lighting")
    parser.add_argument("--house_name", required=True, help="House name (e.g., 1LXtFkjw3qL)")
    parser.add_argument("--usd_path", help="Custom path to USD file (optional)")
    parser.add_argument("--save_config", help="Save lighting configuration to JSON file")
    parser.add_argument("--debug_coords", action="store_true", help="Show coordinate transformation details")
    
    try:
        args = parser.parse_args()
        print(f"✅ Arguments parsed successfully:")
        print(f"   House name: {args.house_name}")
        print(f"   USD path: {args.usd_path}")
        
    except Exception as e:
        print(f"❌ Error parsing arguments: {e}")
        return 1
    
    # Initialize the system
    try:
        print("🔧 Initializing lighting system...")
        lighting_system = MatterportLightingSystem()
        print("✅ Lighting system initialized")
        
    except Exception as e:
        print(f"❌ Error initializing lighting system: {e}")
        return 1
    
    try:
        # Process the house
        print(f"🏠 Starting to process house: {args.house_name}")
        success = lighting_system.process_house(args.house_name, args.usd_path)
        
        if success:
            print("\n🎉 SUCCESS!")
        else:
            print("\n❌ FAILED!")
            print("Check the error messages above for details")
            return 1
        
        try:
            while simulation_app.is_running():
                simulation_app.update()
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
        
    except Exception as e:
        print(f"❌ Unexpected error in main: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        try:
            simulation_app.close()
            print("🔚 Isaac Sim closed")
        except Exception as e:
            print(f"❌ Error closing Isaac Sim: {e}")
    
    return 0


if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code)
