import os
import open3d as o3d
import numpy as np
from pathlib import Path
import time
import tempfile
import shutil
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

# Optional imports - graceful fallback if not available
try:
    from pygltflib import GLTF2
    GLTF_AVAILABLE = True
except ImportError:
    GLTF_AVAILABLE = False
    logger.warning("pygltflib not available - GLB/GLTF conversion disabled")

try:
    from OCC.Core.IGESControl import IGESControl_Reader
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.Interface import Interface_Static
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Extend.DataExchange import write_obj_file
    OPENCASCADE_AVAILABLE = True
except ImportError:
    OPENCASCADE_AVAILABLE = False
    logger.warning("OpenCascade (pythonocc) not available - STEP/IGES conversion disabled")

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    logger.warning("trimesh not available - 3MF/GLB fallback conversion disabled")


def convert_to_obj(input_path):
    """
    Simple function: Convert any 3D file to OBJ in the same directory
    
    Args:
        input_path (str): Path to input file
        
    Returns:
        dict: {
            'success': bool,
            'output_file': str (path to converted OBJ),
            'unit': str,
            'assumption': bool,
            'error': str (if failed)
        }
    """
    input_path = Path(input_path)
    
    # Validate input
    if not input_path.exists():
        return {
            'success': False,
            'output_file': None,
            'unit': 'mm',
            'assumption': True,
            'error': f"Input file not found: {input_path}"
        }
    
    # Generate output path (same directory, .obj extension)
    output_path = input_path.parent / f"{input_path.stem}.obj"
    
    # Extract file extension
    file_ext = input_path.suffix.lower()[1:]  # Remove the dot
    
    print(f"Converting {file_ext.upper()} to OBJ: {input_path} -> {output_path}")
    
    try:
        # Extract units first
        unit, assumption = extract_unit_info(str(input_path), file_ext)
        
        # If already OBJ, just copy
        if file_ext == 'obj':
            if input_path != output_path:
                shutil.copy2(input_path, output_path)
            return {
                'success': True,
                'output_file': str(output_path),
                'unit': unit,
                'assumption': assumption,
                'error': None
            }
        
        # Convert based on format
        success = False
        error_msg = None
        
        if file_ext == 'stl':
            success, error_msg = convert_stl(input_path, output_path)
        elif file_ext == 'ply':
            success, error_msg = convert_ply(input_path, output_path)
        elif file_ext in ['glb', 'gltf']:
            success, error_msg = convert_gltf(input_path, output_path, file_ext)
        elif file_ext in ['stp', 'step']:
            success, error_msg = convert_step(input_path, output_path)
        elif file_ext in ['igs', 'iges']:
            success, error_msg = convert_iges(input_path, output_path)
        elif file_ext == '3mf':
            success, error_msg = convert_3mf(input_path, output_path)
        else:
            success = False
            error_msg = f"Unsupported format: {file_ext}"
        
        if success and output_path.exists():
            return {
                'success': True,
                'output_file': str(output_path),
                'unit': unit,
                'assumption': assumption,
                'error': None
            }
        else:
            return {
                'success': False,
                'output_file': str(output_path),
                'unit': unit,
                'assumption': assumption,
                'error': error_msg or "Conversion failed"
            }
            
    except Exception as e:
        return {
            'success': False,
            'output_file': str(output_path),
            'unit': 'mm',
            'assumption': True,
            'error': str(e)
        }


def convert_stl(input_path, output_path):
    """Convert STL to OBJ"""
    try:
        mesh = o3d.io.read_triangle_mesh(str(input_path))
        if len(mesh.vertices) == 0:
            return False, "STL file contains no vertices"
        
        success = o3d.io.write_triangle_mesh(str(output_path), mesh)
        return success, None if success else "Failed to write OBJ"
    except Exception as e:
        return False, str(e)


def convert_ply(input_path, output_path):
    """Convert PLY to OBJ"""
    try:
        mesh = o3d.io.read_triangle_mesh(str(input_path))
        
        if len(mesh.vertices) == 0:
            # Try as point cloud
            pcd = o3d.io.read_point_cloud(str(input_path))
            if len(pcd.points) == 0:
                return False, "PLY file contains no geometry"
            
            # Convert point cloud to mesh
            pcd.estimate_normals()
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
            
            if len(mesh.vertices) == 0:
                return False, "Failed to create mesh from point cloud"
        
        success = o3d.io.write_triangle_mesh(str(output_path), mesh)
        return success, None if success else "Failed to write OBJ"
    except Exception as e:
        return False, str(e)


def convert_gltf(input_path, output_path, format_name):
    """Convert GLTF/GLB to OBJ"""
    try:
        # Try Open3D first
        try:
            mesh = o3d.io.read_triangle_mesh(str(input_path))
            if len(mesh.vertices) > 0:
                success = o3d.io.write_triangle_mesh(str(output_path), mesh)
                if success:
                    return True, None
        except:
            pass
        
        # Fallback to trimesh
        if not TRIMESH_AVAILABLE:
            return False, f"trimesh required for {format_name.upper()} conversion"
        
        mesh = trimesh.load(str(input_path), file_type=format_name)
        if hasattr(mesh, 'vertices') and len(mesh.vertices) > 0:
            mesh.export(str(output_path), file_type='obj')
            return True, None
        else:
            return False, f"No geometry found in {format_name.upper()} file"
            
    except Exception as e:
        return False, str(e)


def convert_step(input_path, output_path):
    """Convert STEP to OBJ"""
    try:
        if not OPENCASCADE_AVAILABLE:
            return False, "OpenCascade required for STEP conversion"
        
        step_reader = STEPControl_Reader()
        if step_reader.ReadFile(str(input_path)) != 1:
            return False, "Unable to read STEP file"
        
        step_reader.TransferRoots()
        shape = step_reader.OneShape()
        
        if shape.IsNull():
            return False, "STEP file contains no geometry"
        
        mesh = BRepMesh_IncrementalMesh(shape, 0.05)
        write_obj_file(shape, str(output_path))
        return True, None
        
    except Exception as e:
        return False, str(e)


def convert_iges(input_path, output_path):
    """Convert IGES to OBJ"""
    try:
        if not OPENCASCADE_AVAILABLE:
            return False, "OpenCascade required for IGES conversion"
        
        iges_reader = IGESControl_Reader()
        if iges_reader.ReadFile(str(input_path)) != 1:
            return False, "Unable to read IGES file"
        
        iges_reader.TransferRoots()
        shape = iges_reader.OneShape()
        
        if shape.IsNull():
            return False, "IGES file contains no geometry"
        
        mesh = BRepMesh_IncrementalMesh(shape, 0.05)
        write_obj_file(shape, str(output_path))
        return True, None
        
    except Exception as e:
        return False, str(e)


def convert_3mf(input_path, output_path):
    """Convert 3MF to OBJ"""
    try:
        if not TRIMESH_AVAILABLE:
            return False, "trimesh required for 3MF conversion"
        
        mesh_scene = trimesh.load_scene(str(input_path), file_type='3mf', process=False)
        final_geometries = []

        for node_name in mesh_scene.graph.nodes_geometry:
            transform, geometry_name = mesh_scene.graph[node_name]
            geos = mesh_scene.geometry[geometry_name].split()
            transformed = geos[int(geometry_name)-1].copy()
            transformed.apply_transform(transform)
            final_geometries.append(transformed)

        if len(final_geometries) == 0:
            return False, "No geometries found in 3MF file"
        elif len(final_geometries) == 1:
            final_geometries[0].export(str(output_path), file_type='obj')
        else:
            combine = trimesh.util.concatenate(final_geometries)
            combine.export(str(output_path), file_type='obj')
        
        return True, None
        
    except Exception as e:
        return False, str(e)


def extract_unit_info(file_path, file_extension):
    """Extract unit information from file"""
    unit = "mm"  # Default
    assumption = True
    
    try:
        if file_extension == 'ply':
            with open(file_path, 'r', errors='ignore') as f:
                for line in f:
                    if line.startswith("end_header"):
                        break
                    if "unit:" in line.lower():
                        parts = line.lower().split("unit:")
                        if len(parts) > 1:
                            unit = parts[1].strip()
                            assumption = False
                            break
        
        elif file_extension in ['stp', 'step']:
            with open(file_path, 'r', errors='ignore') as f:
                for i, line in enumerate(f):
                    if i > 100:  # Only check header
                        break
                    if 'LENGTHUNIT' in line and 'NAMED_UNIT' in line:
                        if 'MILLI' in line and 'METRE' in line:
                            unit = 'mm'
                            assumption = False
                        elif 'CENTI' in line and 'METRE' in line:
                            unit = 'cm'
                            assumption = False
                        elif 'METRE' in line or 'METER' in line:
                            unit = 'm'
                            assumption = False
                        elif 'INCH' in line:
                            unit = 'in'
                            assumption = False
                        break
        
        elif file_extension == 'obj':
            with open(file_path, 'r', errors='ignore') as f:
                for i, line in enumerate(f):
                    if i > 50:  # Only check first 50 lines
                        break
                    if line.startswith('#') and 'unit' in line.lower():
                        for unit_candidate in ['mm', 'cm', 'm', 'in', 'ft']:
                            if unit_candidate in line.lower():
                                unit = unit_candidate
                                assumption = False
                                break
        
        elif file_extension == 'stl':
            with open(file_path, 'r', errors='ignore') as f:
                first_line = f.readline().lower()
                if 'solid' in first_line:  # ASCII STL
                    for unit_candidate in ['mm', 'cm', 'm', 'in', 'ft']:
                        if unit_candidate in first_line:
                            unit = unit_candidate
                            assumption = False
                            break
                            
    except Exception as e:
        logger.warning(f"Error extracting unit info: {e}")
    
    return unit, assumption


# Legacy compatibility function for the analyzer
def convert_file(input_path, output_path=None, use_local=True):
    """
    Legacy interface for analyzer compatibility
    Just calls convert_to_obj and adapts the interface
    """
    result = convert_to_obj(input_path)
    return result


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert 3D files to OBJ format')
    parser.add_argument('input_file', help='Path to input 3D file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    result = convert_to_obj(args.input_file)
    
    if result['success']:
        print(f"✅ Conversion successful: {result['output_file']}")
        print(f"📏 Unit: {result['unit']} (Assumed: {result['assumption']})")
    else:
        print(f"❌ Conversion failed: {result['error']}")
        sys.exit(1)