import os
import trimesh
from pygltflib import GLTF2
import cadquery as cq
from OCC.Core.IGESControl import IGESControl_Reader
from OCC.Core.Interface import Interface_Static
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Extend.DataExchange import write_obj_file
import time

import tempfile
import shutil
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

class FormatConverter:
    """Convert 3D file formats using Django's storage backend or local filesystem"""

    def __init__(self, input_file, output_file, use_local=False):
        """
        Initialize converter with input and output files
        
        Args:
            input_file (str): Path to input file
            output_file (str): Path to output file
            use_local (bool): If True, use local file operations instead of Django storage
        """
        self.input_file = input_file
        self.output_file = output_file
        self.use_local = use_local
        
        # Set up the storage backend
        if not use_local:
            try:
                from django.core.files.storage import default_storage
                self.storage = default_storage
                logger.info("Using Django storage backend")
            except ImportError:
                logger.warning("Django not available, falling back to local storage")
                self.use_local = True
        
        if self.use_local:
            logger.info("Using local file system for storage")

    def _read_from_storage(self, file_path):
        """Read file from storage (either Django storage or local)"""
        try:
            if self.use_local:
                with open(file_path, 'rb') as source_file:
                    return source_file.read()
            else:
                with self.storage.open(file_path, 'rb') as source_file:
                    return source_file.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            raise

    def _save_to_storage(self, file_path, content):
        """Save file to storage (either Django storage or local)"""
        try:
            if self.use_local:
                # Ensure directory exists
                dir_path = os.path.dirname(file_path)
                if dir_path and not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                    
                # Save the file locally
                if isinstance(content, bytes):
                    with open(file_path, 'wb') as f:
                        f.write(content)
                elif isinstance(content, BytesIO):
                    with open(file_path, 'wb') as f:
                        f.write(content.getvalue())
                else:
                    raise ValueError(f"Unsupported content type: {type(content)}")
                return file_path
            else:
                if isinstance(content, bytes):
                    content = BytesIO(content)
                elif isinstance(content, BytesIO):
                    pass  # Keep BytesIO objects as is
                else:
                    raise ValueError(f"Unsupported content type: {type(content)}")
                return self.storage.save(file_path, content)
        except Exception as e:
            logger.error(f"Error saving file {file_path}: {str(e)}")
            raise

    def _handle_temp_files(self, process_func):
        """Handle temporary file processing with secure temp files and automatic cleanup"""
        temp_dir = None
        temp_input_path = None
        temp_output_path = None
        
        try:
            # Create temp directory
            temp_dir = tempfile.mkdtemp()
            temp_input_path = os.path.join(temp_dir, os.path.basename(self.input_file))
            temp_output_path = os.path.join(temp_dir, os.path.basename(self.output_file))
            
            # Get input file content
            if self.use_local and os.path.exists(self.input_file):
                # If using local and file exists locally, copy it directly
                shutil.copy2(self.input_file, temp_input_path)
            else:
                # Otherwise read from storage (Django or local)
                input_content = self._read_from_storage(self.input_file)
                with open(temp_input_path, 'wb') as f:
                    f.write(input_content)
            
            # Process the files
            result = process_func(temp_input_path, temp_output_path)
            
            # Save the output file
            if os.path.exists(temp_output_path):
                if self.use_local:
                    # If using local, copy the file directly
                    output_dir = os.path.dirname(self.output_file)
                    if output_dir and not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    shutil.copy2(temp_output_path, self.output_file)
                else:
                    # Otherwise save to storage (Django)
                    with open(temp_output_path, 'rb') as f:
                        self._save_to_storage(self.output_file, f.read())
            
            return result

        except Exception as e:
            logger.error(f"File processing error: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "unit": "mm",
                "assumption": True
            }
        finally:
            # Clean up temp files
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up temp directory: {temp_dir}")

    def convert_to_obj(self):
        """Convert any supported 3D file to OBJ format"""
        try:
            file_extension = os.path.splitext(self.input_file)[1].lower()[1:]
            
            # First, extract unit information from the input file
            def extract_wrapper(temp_input, _):
                return self.extract_unit_info(temp_input, file_extension)
                
            unit_info = self._handle_temp_files(extract_wrapper)
            
            # Store the unit information to include in the result
            if isinstance(unit_info, tuple) and len(unit_info) >= 2:
                self.unit = unit_info[0]
                self.assumption = unit_info[1]
            else:
                self.unit = "mm"
                self.assumption = True
            
            # Now proceed with the conversion
            converters = {
                'obj': self.convert_obj_to_obj,
                'stl': self.convert_stl_to_obj,
                'ply': self.convert_ply_to_obj,
                'glb': self.convert_glb_to_obj,
                'igs': self.convert_igs_to_obj,
                'iges': self.convert_igs_to_obj,
                'stp': self.convert_stp_to_obj,
                'step': self.convert_stp_to_obj,
                '3mf': self.convert_3mf_to_obj,
                # 'x_t': self.convert_xt_to_obj,
                # 'sldprt': self.convert_sldprt_to_obj,
                # 'magics': self.convert_magics_to_obj,
            }
            
            converter = converters.get(file_extension)
            if not converter:
                return {
                    "success": False,
                    "error": f"Unsupported file format: {file_extension}",
                    "unit": self.unit,
                    "assumption": self.assumption
                }
                
            result = converter()
            
            # Add the unit information to the conversion result
            if isinstance(result, dict):
                result["unit"] = self.unit
                result["assumption"] = self.assumption
                return result
            else:
                return {
                    "success": True,
                    "error": None,
                    "unit": self.unit,
                    "assumption": self.assumption
                }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "unit": getattr(self, 'unit', "mm"),
                "assumption": getattr(self, 'assumption', True)
            }

    def convert_obj_to_obj(self):
        """Simply copy OBJ file to new location"""
        try:
            if self.use_local and os.path.exists(self.input_file):
                # Direct file copy if local and file exists
                shutil.copy2(self.input_file, self.output_file)
            else:
                # Storage-based copy
                content = self._read_from_storage(self.input_file)
                self._save_to_storage(self.output_file, content)  # content is already bytes

            return {
                "success": True,
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def convert_stl_to_obj(self):
        """Convert STL to OBJ format"""
        def process(temp_input, temp_output):
            try:
                mesh = trimesh.load(temp_input, file_type="stl")
                mesh.export(temp_output, file_type="obj")
                return {
                    "success": True,
                    "error": None
                }
            except Exception as e:
                logger.error(f"STL conversion failed: {str(e)}")
                raise
                    
        return self._handle_temp_files(process)
    
    def convert_3mf_to_obj(self):
        """Convert 3MF file to OBJ"""
        def process(temp_input, temp_output):
            try:
                # Load the 3MF file
                meshScene = trimesh.load_scene(temp_input, file_type='3mf', process=False)
                final_geometries = []

                for node_name in meshScene.graph.nodes_geometry:
                    transform, geometry_name = meshScene.graph[node_name]
                    geos = meshScene.geometry[geometry_name].split()

                    # Apply the transform to the geometry
                    transformed = geos[int(geometry_name)-1].copy()
                    transformed.apply_transform(transform)
                    final_geometries.append(transformed)

                if len(final_geometries) == 0:
                    raise ValueError("No geometries found in the 3MF file.")
                elif len(final_geometries) == 1:
                    # If only one geometry, export it directly
                    final_geometries[0].export(temp_output, file_type='obj')
                else:
                    combine = trimesh.util.concatenate(final_geometries)
                    combine.export(temp_output, file_type='obj')
                return {
                    "success": True,
                    "error": None
                    }
            except Exception as e:
                logger.error(f"3MF conversion failed: {str(e)}")
                raise
        
        return self._handle_temp_files(process)
    
    def convert_stp_to_obj(self):
        """Convert STEP/STP to OBJ format."""
        def process(temp_input, temp_output):
            try:
                # Load the STEP file using cadquery
                shape = cq.importers.importStep(temp_input)

                # Create a temporary STL file to export the shape
                temp_output_stl = os.path.splitext(temp_output)[0] + ".stl"
                
                # Export the shape to OBJ format
                cq.exporters.export(shape, temp_output_stl)

                # Load the STL file using trimesh
                mesh = trimesh.load(temp_output_stl, file_type="stl")
                # Export to OBJ format
                mesh.export(temp_output, file_type="obj")
                
                return {
                    "success": True,
                    "error": None
                }
            except Exception as e:
                logger.error(f"STEP conversion failed: {str(e)}")
                raise
                    
        return self._handle_temp_files(process)

    def convert_ply_to_obj(self):
        """Convert PLY to OBJ format"""
        def process(temp_input, temp_output):
            try:
                mesh = trimesh.load(temp_input, file_type='ply')
                mesh.export(temp_output, file_type='obj')
                return {
                    "success": True,
                    "error": None
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e)
                }
        return self._handle_temp_files(process)

    def convert_glb_to_obj(self):
        """Convert GLB to OBJ format"""
        def process(temp_input, temp_output):
            try:
                mesh = trimesh.load(temp_input, file_type='glb')
                mesh.export(temp_output, file_type='obj')
                return {
                    "success": True,
                    "error": None
                }
            except Exception as e:
                logger.error(f"GLB conversion failed: {str(e)}")
                raise
                    
        return self._handle_temp_files(process)

    def convert_igs_to_obj(self):
        """Convert IGES/IGS to OBJ format"""
        def process(temp_input, temp_output):
            try:
                iges_reader = IGESControl_Reader()
                if iges_reader.ReadFile(temp_input) != 1:
                    raise Exception("Unable to read IGES file")

                iges_reader.TransferRoots()
                shape = iges_reader.OneShape()
                BRepMesh_IncrementalMesh(shape, 0.05)
                write_obj_file(shape, temp_output)

                return {
                    "success": True,
                    "error": None
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e)
                }
        return self._handle_temp_files(process)
    
    ###### Extract unit information from the input file ######
    def extract_unit_info(self, file_path, file_extension):
        """
        Extract unit information from the input file before conversion.
        
        Args:
            file_path (str): Path to the input file
            file_extension (str): File extension (without the dot)
            
        Returns:
            tuple: (unit, assumption) where unit is a string like 'mm', 'cm', etc.,
                and assumption is a boolean indicating if the unit was assumed
        """
        # Default values - assume mm for all formats that don't explicitly specify units
        unit = "mm"
        assumption = True
        
        try:
            # PLY files: Check header comments
            if file_extension == 'ply':
                with open(file_path, 'r', errors='ignore') as file:
                    for line in file:
                        if line.startswith("end_header"):
                            break
                        if line.startswith("comment Unit:"):
                            unit = line.split("Unit:")[-1].strip()
                            assumption = False
                            break
            
            # GLB/GLTF files: Check scale information
            elif file_extension in ['glb', 'gltf']:
                gltf = GLTF2().load(file_path)
                unit_map = {
                    1.0: 'm',
                    0.0254: 'in',
                    0.3048: 'ft',
                    100.0: 'cm',
                    1000.0: 'mm'
                }
                
                # Check nodes for scale information
                for node in gltf.nodes or []:
                    if hasattr(node, 'scale') and node.scale:
                        # Get first scale value (assuming uniform scaling)
                        scale = node.scale[0] if isinstance(node.scale, list) else node.scale
                        # Find closest match in unit_map
                        unit = unit_map.get(scale, 'm')
                        assumption = False
                        break
                        
                # If no explicit scale in nodes, check asset metadata
                if assumption and hasattr(gltf, 'asset') and hasattr(gltf.asset, 'extras'):
                    extras = gltf.asset.extras
                    if isinstance(extras, dict) and 'unit' in extras:
                        unit = extras['unit']
                        assumption = False
            
            # IGES/IGS files: Use OpenCascade
            elif file_extension in ['igs', 'iges']:
                iges_reader = IGESControl_Reader()
                if iges_reader.ReadFile(file_path) == 1:
                    iges_reader.TransferRoots()
                    cascade_unit = Interface_Static.CVal("xstep.cascade.unit")
                    if cascade_unit:
                        unit = cascade_unit.lower()
                        assumption = False
            
            # STEP/STP files: Parse header
            elif file_extension in ['stp', 'step']:
                # Try to read from the file header
                with open(file_path, 'r', errors='ignore') as file:
                    for i, line in enumerate(file):
                        if 'LENGTHUNIT' in line and 'NAMED_UNIT' in line:
                            if 'MILLI' in line and 'METRE' in line:
                                unit = 'mm'
                                assumption = False
                                break
                            elif 'CENTI' in line and 'METRE' in line:
                                unit = 'cm'
                                assumption = False
                                break
                            elif ('METRE' in line or 'METER' in line) and not any(prefix in line for prefix in ['MILLI', 'CENTI', 'DECI']):
                                unit = 'm'
                                assumption = False
                                break
                            elif 'INCH' in line:
                                unit = 'in'
                                assumption = False
                                break
                            elif 'FOOT' in line or 'FEET' in line:
                                unit = 'ft'
                                assumption = False
                                break
                        
                        # Only scan the header section (first ~100 lines)
                        if i > 100:
                            break
                            
            # 3MF files: Check metadata in the 3MF XML structure
            elif file_extension == '3mf':
                try:
                    import zipfile
                    import xml.etree.ElementTree as ET
                    
                    # 3MF files are ZIP archives with XML content
                    with zipfile.ZipFile(file_path) as z:
                        # Look for the model file
                        for filename in z.namelist():
                            if filename.endswith('3dmodel.model'):
                                with z.open(filename) as f:
                                    tree = ET.parse(f)
                                    root = tree.getroot()
                                    
                                    # Find namespaces to parse XML properly
                                    namespaces = dict([node for _, node in ET.iterparse(
                                        BytesIO(ET.tostring(root)), events=['start-ns'])])
                                    
                                    # Check for unit attribute in the model
                                    for k, v in namespaces.items():
                                        if 'metadata' in v:
                                            metadata_ns = f"{{{v}}}"
                                            break
                                    else:
                                        metadata_ns = ''
                                    
                                    # Look for unit metadata
                                    for metadata in root.findall(f".//{metadata_ns}metadata"):
                                        if metadata.get('name') == 'unit':
                                            unit = metadata.text.lower()
                                            assumption = False
                                            break
                except Exception:
                    pass
                    
            # OBJ files: Check comments for unit info
            elif file_extension == 'obj':
                with open(file_path, 'r', errors='ignore') as file:
                    for i, line in enumerate(file):
                        if line.startswith('#'):
                            if 'unit' in line.lower():
                                parts = line.lower().split('unit')
                                if len(parts) > 1:
                                    for unit_candidate in ['mm', 'cm', 'm', 'in', 'ft']:
                                        if unit_candidate in parts[1]:
                                            unit = unit_candidate
                                            assumption = False
                                            break
                        
                        # Only scan the first 50 lines for comments
                        if i > 50:
                            break
            
            # For STL and other formats without unit information, just use default mm
            
        except Exception as e:
            logger.warning(f"Error extracting unit information: {str(e)}")
            # Fall back to default values on any error
        
        return unit, assumption


# Helper function to check if Django is available
def is_django_available():
    try:
        import django
        # Check if Django settings are configured
        from django.conf import settings
        # Try accessing a setting to verify Django is properly configured
        settings.DEFAULT_FILE_STORAGE
        return True
    except ImportError:
        # Django isn't installed
        return False
    except django.core.exceptions.ImproperlyConfigured:
        # Django is installed but not configured properly
        return False
    except Exception:
        # Catch any other potential Django-related issues
        return False

# Usage example
def convert_file(input_path, output_path, use_local=None):
    """
    Convert a 3D file to OBJ format
    
    Args:
        input_path (str): Path to input file
        output_path (str): Path to output file
        use_local (bool): Force local file operations (if None, auto-detect)
    
    Returns:
        dict: Result of conversion with success status
    """
    # Auto-detect if Django is available (if not specified)
    if use_local is None:
        use_local = not is_django_available()
        
    converter = FormatConverter(input_path, output_path, use_local=use_local)
    return converter.convert_to_obj()


# Command-line interface if run directly
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert 3D files to OBJ format')
    parser.add_argument('input_file', help='Path to input 3D file')
    parser.add_argument('output_file', help='Path to output OBJ file')
    parser.add_argument('--local', action='store_true', help='Force local file operations')
    parser.add_argument('--django', action='store_true', help='Force Django storage backend')
    args = parser.parse_args()
    
    # Determine storage mode
    use_local = None  # Auto-detect by default
    if args.local:
        use_local = True
    elif args.django:
        use_local = False
    
    # Configure basic logging for CLI usage
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Run conversion
    start_time = time.time()
    result = convert_file(args.input_file, args.output_file, use_local=use_local)
    end_time = time.time()
    logger.info(f"Conversion completed in {end_time - start_time:.2f} seconds")
    
    if result["success"]:
        print(f"Conversion successful: {args.output_file}")
        print(f"Unit: {result['unit']} (Assumed: {result['assumption']})")
        sys.exit(0)
    else:
        print(f"Conversion failed: {result['error']}")
        sys.exit(1)