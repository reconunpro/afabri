"""
3D Mesh Analyzer - GPU-Accelerated Version with Open3D

A high-performance implementation for analyzing 3D mesh files (.obj format)
with GPU-accelerated raycasting using Open3D for accurate thin wall and gap detection.
Enhanced with additional geometric calculations and performance timing.
"""

import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Union

@dataclass
class ConversionInfo:
    """Container for file conversion information"""
    original_format: str
    format_name: str
    original_file: str
    converted_file: str
    unit: str
    unit_assumed: bool
    reused_existing: bool = False

from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path
import time
import json
import argparse
import sys
import os

# Import the external thin wall analysis module
import thinWallAnalysis

# Import format converter for automatic file conversion
try:
    from formatConverter import convert_to_obj
    FORMAT_CONVERTER_AVAILABLE = True
except ImportError as e:
    FORMAT_CONVERTER_AVAILABLE = False


@dataclass
class MeshDimensions:
    """Container for mesh physical dimensions with enhanced metrics"""
    length: float
    width: float
    height: float
    volume: Optional[float]
    surface_area: float
    is_watertight: bool
    # New metrics
    surface_to_volume_ratio: Optional[float] = None
    bounding_to_actual_volume_ratio: Optional[float] = None
    concavity_ratio: Optional[float] = None  # convex_hull_volume / actual_volume
    bounding_box_volume: Optional[float] = None
    convex_hull_volume: Optional[float] = None


@dataclass
class WallAnalysisResult:
    """Container for wall thickness analysis results"""
    thinnest_feature: Optional[float] = None
    vertex_agreement: Optional[str] = None
    error: Optional[str] = None


@dataclass
class GapAnalysisResult:
    """Container for gap analysis results"""
    smallest_gap: Optional[float] = None
    vertex_agreement: Optional[str] = None
    error: Optional[str] = None


@dataclass
class RaycastResult:
    """Container for raycast operation results"""
    inward_distances: List[float] = field(default_factory=list)
    outward_distances: List[float] = field(default_factory=list)
    inward_face_data: Optional[np.ndarray] = None
    outward_face_data: Optional[np.ndarray] = None


@dataclass
class TimingInfo:
    """Container for timing information"""
    format_conversion: float = 0.0
    mesh_repair: float = 0.0
    analysis_setup: float = 0.0
    raycast_analysis: float = 0.0
    dimensional_analysis: float = 0.0
    wall_analysis: float = 0.0
    gap_analysis: float = 0.0
    total_time: float = 0.0


class GPUManager:
    """Manages GPU device selection and configuration for Open3D"""
    
    @staticmethod
    def setup_gpu():
        """Setup GPU acceleration if available"""
        try:
            if o3d.core.cuda.device_count() > 0:
                device = o3d.core.Device("CUDA:0")
                return device
            else:
                return o3d.core.Device("CPU:0")
        except Exception as e:
            return o3d.core.Device("CPU:0")
    
    @staticmethod
    def get_device_info(device):
        """Get information about the current device"""
        if device.get_type() == o3d.core.Device.DeviceType.CUDA:
            return {
                "type": "GPU",
                "name": f"CUDA Device {device.get_id()}",
                "memory_info": "GPU acceleration active"
            }
        else:
            return {
                "type": "CPU",
                "name": f"CPU Device {device.get_id()}",
                "memory_info": "CPU computation"
            }


class MeshLoader:
    """Handles loading 3D mesh files with automatic format conversion to OBJ"""
    
    # Supported input formats
    SUPPORTED_FORMATS = {
        '.obj': 'Wavefront OBJ',
        '.stl': 'STL',
        '.ply': 'PLY',
        '.glb': 'GLB',
        '.gltf': 'GLTF',
        '.igs': 'IGES',
        '.iges': 'IGES',
        '.stp': 'STEP',
        '.step': 'STEP',
        '.3mf': '3MF'
    }
    
    @staticmethod
    def load(file_input: Union[str, Path, bytes]) -> Tuple[o3d.geometry.TriangleMesh, Optional[ConversionInfo], float]:
        """
        Load a mesh from file path with automatic format conversion
        
        Args:
            file_input: Path to file or bytes data
            
        Returns:
            Tuple of (mesh, conversion_info, conversion_time)
        """
        if isinstance(file_input, bytes):
            mesh = MeshLoader._load_from_bytes(file_input)
            return mesh, None, 0.0
        else:
            return MeshLoader._load_from_path(file_input)
    
    @staticmethod
    def _load_from_path(file_path: Union[str, Path]) -> Tuple[o3d.geometry.TriangleMesh, Optional[ConversionInfo], float]:
        """Load mesh from file path with automatic conversion"""
        conversion_start = time.time()
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ValueError(f"File not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        if file_extension not in MeshLoader.SUPPORTED_FORMATS:
            supported_list = ', '.join(MeshLoader.SUPPORTED_FORMATS.keys())
            raise ValueError(f"Unsupported file format: {file_extension}. Supported formats: {supported_list}")
        
        try:
            # If it's already an OBJ file, load directly
            if file_extension == '.obj':
                mesh = o3d.io.read_triangle_mesh(str(file_path))
                MeshLoader._validate_mesh(mesh)
                MeshLoader._prepare_mesh(mesh)
                
                conversion_info = ConversionInfo(
                    original_format=".obj",
                    format_name="Wavefront OBJ",
                    original_file=str(file_path),
                    converted_file=str(file_path),
                    unit="mm",
                    unit_assumed=True,
                    reused_existing=False
                )
                conversion_time = time.time() - conversion_start
                return mesh, conversion_info, conversion_time
            
            # Convert other formats to OBJ first
            if not FORMAT_CONVERTER_AVAILABLE:
                raise ValueError(f"Format converter not available. Cannot process {file_extension} files.")
            
            print(f"Converting {MeshLoader.SUPPORTED_FORMATS[file_extension]} to OBJ...")
            result = convert_to_obj(str(file_path))
            
            if not result['success']:
                raise ValueError(f"Format conversion failed: {result['error']}")
            
            obj_path = Path(result['output_file'])
            
            # Load the converted OBJ file
            mesh = o3d.io.read_triangle_mesh(str(obj_path))
            MeshLoader._validate_mesh(mesh)
            MeshLoader._prepare_mesh(mesh)
            
            conversion_info = ConversionInfo(
                original_format=file_extension,
                format_name=MeshLoader.SUPPORTED_FORMATS[file_extension],
                original_file=str(file_path),
                converted_file=str(obj_path),
                unit=result['unit'],
                unit_assumed=result['assumption'],
                reused_existing=False
            )
            
            conversion_time = time.time() - conversion_start
            print(f"Conversion completed in {conversion_time:.2f}s")
            return mesh, conversion_info, conversion_time
            
        except Exception as e:
            raise ValueError(f"Failed to load {MeshLoader.SUPPORTED_FORMATS.get(file_extension, 'unknown')} file: {e}")
    
    @staticmethod
    def _load_from_bytes(file_bytes: bytes) -> o3d.geometry.TriangleMesh:
        """Load mesh from bytes (assumes OBJ format for bytes input)"""
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as tmp:
                tmp.write(file_bytes)
                tmp.flush()
                temp_path = tmp.name
            
            try:
                mesh = o3d.io.read_triangle_mesh(temp_path)
                MeshLoader._validate_mesh(mesh)
                MeshLoader._prepare_mesh(mesh)
                return mesh
            finally:
                Path(temp_path).unlink()
            
        except Exception as e:
            raise ValueError(f"Failed to load mesh from bytes: {e}")
    
    @staticmethod
    def _validate_mesh(mesh: o3d.geometry.TriangleMesh) -> None:
        """Validate that mesh has valid geometry"""
        if len(mesh.vertices) == 0:
            raise ValueError("Invalid mesh geometry: no vertices found")
        if len(mesh.triangles) == 0:
            raise ValueError("Invalid mesh geometry: no triangles found")
    
    @staticmethod
    def _prepare_mesh(mesh: o3d.geometry.TriangleMesh) -> None:
        """Prepare mesh for analysis by computing normals and other properties"""
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        if not mesh.has_triangle_normals():
            mesh.compute_triangle_normals()
        
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
    
    @staticmethod
    def get_supported_formats() -> Dict[str, str]:
        """Get dictionary of supported file formats"""
        return MeshLoader.SUPPORTED_FORMATS.copy()


class FaceSampler:
    """Handles intelligent face sampling for large meshes"""
    
    def __init__(self, mesh: o3d.geometry.TriangleMesh):
        self.mesh = mesh
        self.total_faces = len(mesh.triangles)
        self._triangle_areas = None
        self._triangle_centers = None
        
    def sample(self, accuracy: str = 'medium') -> np.ndarray:
        """Sample faces based on accuracy level"""
        if accuracy == 'full':
            return np.arange(self.total_faces)
        
        sample_rate = self._get_sample_rate(accuracy)
        num_samples = max(100, int(self.total_faces * sample_rate))
        
        return self._spatial_sampling(num_samples)
    
    def _get_sample_rate(self, accuracy: str) -> float:
        """Get sampling rate based on mesh size and accuracy"""
        if self.total_faces < 10000:
            rates = {'low': 0.3, 'medium': 0.5, 'high': 0.7}
        elif self.total_faces < 50000:
            rates = {'low': 0.1, 'medium': 0.2, 'high': 0.4}
        else:
            rates = {'low': 0.01, 'medium': 0.05, 'high': 0.1}
        
        return rates.get(accuracy, rates['medium'])
    
    def _get_triangle_areas(self) -> np.ndarray:
        """Compute triangle areas if not cached"""
        if self._triangle_areas is None:
            vertices = np.asarray(self.mesh.vertices)
            triangles = np.asarray(self.mesh.triangles)
            
            v0 = vertices[triangles[:, 0]]
            v1 = vertices[triangles[:, 1]]
            v2 = vertices[triangles[:, 2]]
            
            cross = np.cross(v1 - v0, v2 - v0)
            self._triangle_areas = 0.5 * np.linalg.norm(cross, axis=1)
        
        return self._triangle_areas
    
    def _get_triangle_centers(self) -> np.ndarray:
        """Compute triangle centers if not cached"""
        if self._triangle_centers is None:
            vertices = np.asarray(self.mesh.vertices)
            triangles = np.asarray(self.mesh.triangles)
            
            v0 = vertices[triangles[:, 0]]
            v1 = vertices[triangles[:, 1]]
            v2 = vertices[triangles[:, 2]]
            
            self._triangle_centers = (v0 + v1 + v2) / 3.0
        
        return self._triangle_centers
    
    def _spatial_sampling(self, num_samples: int) -> np.ndarray:
        """Perform spatially-aware sampling for better coverage"""
        grid_size = max(2, int(np.ceil(num_samples ** (1.0/3))))
        grid_cells = self._assign_faces_to_grid(grid_size)
        
        sampled_indices = []
        target_per_cell = max(1, num_samples // len(grid_cells))
        
        for cell_faces in grid_cells.values():
            cell_sampled = self._sample_from_cell(cell_faces, target_per_cell)
            sampled_indices.extend(cell_sampled)
        
        sampled_indices = self._ensure_minimum_samples(sampled_indices, num_samples)
        return np.array(sampled_indices)
    
    def _assign_faces_to_grid(self, grid_size: int) -> Dict:
        """Assign faces to spatial grid cells"""
        grid_cells = {}
        bbox = self.mesh.get_axis_aligned_bounding_box()
        model_min = bbox.min_bound
        model_max = bbox.max_bound
        cell_size = (model_max - model_min) / grid_size
        
        face_areas = self._get_triangle_areas()
        face_centers = self._get_triangle_centers()
        
        for face_idx in range(self.total_faces):
            center = face_centers[face_idx]
            cell_coords = tuple(
                np.clip(
                    np.floor((center - model_min) / cell_size).astype(int),
                    0, grid_size - 1
                )
            )
            
            if cell_coords not in grid_cells:
                grid_cells[cell_coords] = []
            grid_cells[cell_coords].append((face_idx, face_areas[face_idx]))
        
        return grid_cells
    
    def _sample_from_cell(self, cell_faces: List[Tuple[int, float]], 
                         target_samples: int) -> List[int]:
        """Sample faces from a single grid cell"""
        if not cell_faces:
            return []
        
        indices = [f[0] for f in cell_faces]
        weights = np.array([f[1] for f in cell_faces])
        
        if np.sum(weights) == 0:
            num_samples = min(target_samples, len(indices))
            if num_samples > 0:
                return np.random.choice(indices, size=num_samples, replace=False).tolist()
            return []
        
        non_zero_mask = weights > 0
        non_zero_indices = np.array(indices)[non_zero_mask]
        non_zero_weights = weights[non_zero_mask]
        
        if len(non_zero_indices) == 0:
            num_samples = min(target_samples, len(indices))
            if num_samples > 0:
                return np.random.choice(indices, size=num_samples, replace=False).tolist()
            return []
        
        non_zero_weights = non_zero_weights / np.sum(non_zero_weights)
        num_samples = min(target_samples, len(non_zero_indices))
        
        if num_samples > 0:
            return np.random.choice(
                non_zero_indices, size=num_samples, replace=False, p=non_zero_weights
            ).tolist()
        
        return []
    
    def _ensure_minimum_samples(self, sampled: List[int], target: int) -> List[int]:
        """Ensure we have at least the target number of samples"""
        if len(sampled) >= target:
            return sampled[:target]
        
        remaining = target - len(sampled)
        available = np.setdiff1d(np.arange(self.total_faces), sampled)
        
        if len(available) > 0:
            additional = np.random.choice(
                available, 
                size=min(remaining, len(available)), 
                replace=False
            )
            sampled.extend(additional.tolist())
        
        return sampled


class GPURaycastEngine:
    """GPU-accelerated ray casting using Open3D"""
    
    def __init__(self, mesh: o3d.geometry.TriangleMesh, device: o3d.core.Device):
        self.mesh = mesh
        self.device = device
        self._setup_raycasting_scene()
        
    def _setup_raycasting_scene(self):
        """Setup the raycasting scene for GPU acceleration"""
        try:
            self.scene = o3d.t.geometry.RaycastingScene()
            mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh, device=self.device)
            self.scene.add_triangles(mesh_t)
            
        except Exception as e:
            self.device = o3d.core.Device("CPU:0")
            mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh, device=self.device)
            self.scene = o3d.t.geometry.RaycastingScene()
            self.scene.add_triangles(mesh_t)
    
    def cast_rays(self, face_indices: np.ndarray, 
                  export_mode: bool = False) -> RaycastResult:
        """Cast rays inward and outward from specified faces using GPU/CPU"""
        max_distance = float(np.max(self.mesh.get_axis_aligned_bounding_box().get_extent()) * 0.5)
        
        face_centers, face_normals = self._get_face_properties(face_indices)
        
        batch_size = self._get_optimal_batch_size()
        result = RaycastResult()
        
        if export_mode:
            total_faces = len(self.mesh.triangles)
            result.inward_face_data = np.full(total_faces, np.nan)
            result.outward_face_data = np.full(total_faces, np.nan)
        
        num_batches = (len(face_indices) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(face_indices))
            
            batch_indices = face_indices[start_idx:end_idx]
            batch_centers = face_centers[start_idx:end_idx]
            batch_normals = face_normals[start_idx:end_idx]
            
            self._cast_batch_rays(
                batch_centers, -batch_normals, batch_indices, max_distance,
                result.inward_distances, result.inward_face_data, export_mode
            )
            
            self._cast_batch_rays(
                batch_centers, batch_normals, batch_indices, max_distance,
                result.outward_distances, result.outward_face_data, export_mode
            )
        
        if export_mode:
            result.inward_face_data = self._interpolate_missing_values(result.inward_face_data)
            result.outward_face_data = self._interpolate_missing_values(result.outward_face_data)
        
        return result
    
    def _get_optimal_batch_size(self) -> int:
        """Determine optimal batch size based on device memory"""
        if self.device.get_type() == o3d.core.Device.DeviceType.CUDA:
            return min(10000, max(1000, len(self.mesh.triangles) // 10))
        else:
            return min(5000, max(500, len(self.mesh.triangles) // 20))
    
    def _get_face_properties(self, face_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get face centers and normals for specified faces"""
        vertices = np.asarray(self.mesh.vertices)
        triangles = np.asarray(self.mesh.triangles)
        triangle_normals = np.asarray(self.mesh.triangle_normals)
        
        selected_triangles = triangles[face_indices]
        
        v0 = vertices[selected_triangles[:, 0]]
        v1 = vertices[selected_triangles[:, 1]]
        v2 = vertices[selected_triangles[:, 2]]
        face_centers = (v0 + v1 + v2) / 3.0
        
        face_normals = triangle_normals[face_indices]
        
        return face_centers, face_normals
    
    def _cast_batch_rays(self, origins: np.ndarray, directions: np.ndarray,
                        face_indices: np.ndarray, max_distance: float,
                        measurements: List[float], face_data: Optional[np.ndarray],
                        export_mode: bool):
        """Cast a batch of rays using GPU acceleration"""
        try:
            ray_origins = o3d.core.Tensor(origins, dtype=o3d.core.float32, device=self.device)
            ray_directions = o3d.core.Tensor(directions, dtype=o3d.core.float32, device=self.device)
            
            rays = o3d.core.concatenate([ray_origins, ray_directions], axis=1)
            ans = self.scene.cast_rays(rays)
            
            hit_distances = ans['t_hit'].cpu().numpy()
            geometry_ids = ans['geometry_ids'].cpu().numpy()
            
            for i, (distance, geom_id) in enumerate(zip(hit_distances, geometry_ids)):
                if geom_id != self.scene.INVALID_ID and distance > 1e-3 and distance < max_distance:
                    measurements.append(distance)
                    
                    if export_mode and face_data is not None:
                        global_idx = face_indices[i]
                        face_data[global_idx] = distance
                        
        except Exception as e:
            self._cast_batch_rays_fallback(
                origins, directions, face_indices, max_distance,
                measurements, face_data, export_mode
            )
    
    def _cast_batch_rays_fallback(self, origins: np.ndarray, directions: np.ndarray,
                                 face_indices: np.ndarray, max_distance: float,
                                 measurements: List[float], face_data: Optional[np.ndarray],
                                 export_mode: bool):
        """Fallback CPU-based raycasting for when GPU fails"""
        try:
            legacy_scene = o3d.t.geometry.RaycastingScene()
            mesh_cpu = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh, device=o3d.core.Device("CPU:0"))
            legacy_scene.add_triangles(mesh_cpu)
            
            ray_origins = o3d.core.Tensor(origins, dtype=o3d.core.float32, device=o3d.core.Device("CPU:0"))
            ray_directions = o3d.core.Tensor(directions, dtype=o3d.core.float32, device=o3d.core.Device("CPU:0"))
            
            rays = o3d.core.concatenate([ray_origins, ray_directions], axis=1)
            ans = legacy_scene.cast_rays(rays)
            
            hit_distances = ans['t_hit'].cpu().numpy()
            geometry_ids = ans['geometry_ids'].cpu().numpy()
            
            for i, (distance, geom_id) in enumerate(zip(hit_distances, geometry_ids)):
                if geom_id != legacy_scene.INVALID_ID and distance > 1e-3 and distance < max_distance:
                    measurements.append(distance)
                    
                    if export_mode and face_data is not None:
                        global_idx = face_indices[i]
                        face_data[global_idx] = distance
                        
        except Exception as e:
            pass  # Silent fallback failure
    
    def _interpolate_missing_values(self, data: np.ndarray) -> np.ndarray:
        """Interpolate missing values using nearest neighbors"""
        if not np.any(np.isnan(data)):
            return data
        
        valid_mask = ~np.isnan(data)
        if not np.any(valid_mask):
            return data
        
        try:
            face_centers = self._get_triangle_centers()
            valid_indices = np.where(valid_mask)[0]
            valid_centers = face_centers[valid_indices]
            valid_values = data[valid_indices]
            
            tree = cKDTree(valid_centers)
            missing_indices = np.where(np.isnan(data))[0]
            
            for idx in missing_indices:
                center = face_centers[idx]
                distances, neighbors = tree.query(center, k=min(3, len(valid_centers)))
                
                if np.all(distances > 0):
                    weights = 1.0 / distances
                    data[idx] = np.average(valid_values[neighbors], weights=weights)
                else:
                    data[idx] = valid_values[neighbors[0]]
                    
        except Exception as e:
            median_value = np.median(data[valid_mask])
            data[np.isnan(data)] = median_value
        
        return data
    
    def _get_triangle_centers(self) -> np.ndarray:
        """Get all triangle centers"""
        vertices = np.asarray(self.mesh.vertices)
        triangles = np.asarray(self.mesh.triangles)
        
        v0 = vertices[triangles[:, 0]]
        v1 = vertices[triangles[:, 1]]
        v2 = vertices[triangles[:, 2]]
        
        return (v0 + v1 + v2) / 3.0


class VisualizationExporter:
    """Handles exporting colored mesh visualizations"""
    
    def __init__(self, mesh: o3d.geometry.TriangleMesh):
        self.mesh = mesh
        
    def export_colored_mesh(self, face_data: np.ndarray, 
                           output_path: str, legend_path: str,
                           title: str, colormap: str = 'jet'):
        """Export mesh with colors based on face data"""
        try:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            face_colors = self._map_values_to_colors(face_data, colormap)
            vertex_colors = self._propagate_face_colors_to_vertices(face_colors)
            
            colored_mesh = self.mesh.__copy__()
            colored_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors / 255.0)
            
            o3d.io.write_triangle_mesh(output_path, colored_mesh)
            self._create_legend(face_data, legend_path, title, colormap)
            
        except Exception as e:
            pass  # Silent failure for visualization
    
    def _map_values_to_colors(self, values: np.ndarray, colormap: str) -> np.ndarray:
        """Map scalar values to RGB colors"""
        valid_values = values[~np.isnan(values)]
        if len(valid_values) == 0:
            return np.zeros((len(values), 3), dtype=np.uint8)
        
        vmin, vmax = np.min(valid_values), np.max(valid_values)
        normalized = np.zeros_like(values)
        valid_mask = ~np.isnan(values)
        
        if vmax > vmin:
            normalized[valid_mask] = (values[valid_mask] - vmin) / (vmax - vmin)
        
        cmap = plt.get_cmap(colormap)
        colors = cmap(normalized)
        
        return (colors[:, :3] * 255).astype(np.uint8)
    
    def _propagate_face_colors_to_vertices(self, face_colors: np.ndarray) -> np.ndarray:
        """Average face colors to vertices"""
        vertices = np.asarray(self.mesh.vertices)
        triangles = np.asarray(self.mesh.triangles)
        
        vertex_colors = np.zeros((len(vertices), 3), dtype=np.float64)
        vertex_counts = np.zeros(len(vertices), dtype=int)
        
        for face_idx, face in enumerate(triangles):
            for vertex_idx in face:
                vertex_colors[vertex_idx] += face_colors[face_idx]
                vertex_counts[vertex_idx] += 1
        
        mask = vertex_counts > 0
        vertex_colors[mask] /= vertex_counts[mask, np.newaxis]
        
        return vertex_colors.astype(np.uint8)
    
    def _create_legend(self, values: np.ndarray, output_path: str,
                      title: str, colormap: str):
        """Create a color legend"""
        valid_values = values[~np.isnan(values)]
        if len(valid_values) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(8, 2))
        
        norm = Normalize(vmin=np.min(valid_values), vmax=np.max(valid_values))
        sm = cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(colormap))
        sm.set_array([])
        
        cbar = fig.colorbar(sm, ax=ax, orientation='horizontal')
        cbar.set_label(title, fontsize=12)
        
        stats_text = (
            f"Min: {np.min(valid_values):.3f}\n"
            f"Max: {np.max(valid_values):.3f}\n"
            f"Mean: {np.mean(valid_values):.3f}"
        )
        fig.text(0.02, 0.5, stats_text, fontsize=10, 
                verticalalignment='center')
        
        ax.set_visible(False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


class MeshAnalyzer:
    """Main analyzer class that orchestrates all analysis operations for maximum accuracy"""
    
    def __init__(self, file_input: Union[str, Path, bytes], 
                 use_gpu: bool = True, 
                 repair_mesh: bool = True,
                 force_volume: bool = False):
        """Initialize analyzer with mesh file for maximum accuracy analysis"""
        # Initialize timing
        self.timing = TimingInfo()
        total_start = time.time()
        
        # Load mesh with timing
        mesh_result = MeshLoader.load(file_input)
        if isinstance(mesh_result, tuple):
            self.mesh, self.conversion_info, self.timing.format_conversion = mesh_result
        else:
            self.mesh = mesh_result
            self.conversion_info = None
            self.timing.format_conversion = 0.0
        
        self.force_volume = force_volume
        
        # Mesh repair with timing
        repair_start = time.time()
        if repair_mesh:
            self.mesh = self._repair_mesh(self.mesh)
        else:
            try:
                self.is_watertight = self.mesh.is_watertight()
                self.attempted_repair = False
            except Exception as e:
                self.is_watertight = False
                self.attempted_repair = False
        self.timing.mesh_repair = time.time() - repair_start
        
        # Setup components with timing
        setup_start = time.time()
        self.device = GPUManager.setup_gpu() if use_gpu else o3d.core.Device("CPU:0")
        self.device_info = GPUManager.get_device_info(self.device)
        
        self.sampler = FaceSampler(self.mesh)
        self.raycast_engine = GPURaycastEngine(self.mesh, self.device)
        self.visualizer = VisualizationExporter(self.mesh)
        self._separate_objects()
        self._raycast_cache = None
        self.timing.analysis_setup = time.time() - setup_start
        
        # Show concise initialization info
        device_type = "GPU" if self.device_info['type'] == 'GPU' else "CPU"
        print(f"Initialized: {len(self.mesh.vertices)} vertices, {len(self.mesh.triangles)} faces ({device_type})")
        
        if self.conversion_info and self.conversion_info.original_format != '.obj':
            print(f"Converted: {self.conversion_info.format_name} → OBJ ({self.timing.format_conversion:.2f}s)")
        
        if repair_mesh:
            status = "watertight" if self.is_watertight else "not watertight"
            print(f"Mesh repair: {status} ({self.timing.mesh_repair:.2f}s)")
        
        if not hasattr(self, 'is_watertight'):
            try:
                self.is_watertight = self.mesh.is_watertight()
                self.attempted_repair = repair_mesh
            except:
                self.is_watertight = False
                self.attempted_repair = repair_mesh
        
        self.timing.total_time = time.time() - total_start
    
    def _repair_mesh(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """Attempt aggressive mesh repair for better watertightness"""
        try:
            original_faces = len(mesh.triangles)
            repaired = mesh.__copy__()
            
            # Basic cleanup
            repaired.remove_duplicated_vertices()
            repaired.remove_duplicated_triangles()
            repaired.remove_degenerate_triangles()
            
            # Merge close vertices
            for tolerance in [1e-8, 1e-7, 1e-6, 1e-5]:
                repaired.merge_close_vertices(tolerance)
                repaired.remove_degenerate_triangles()
            
            # Remove non-manifold edges
            for attempt in range(3):
                try:
                    repaired.remove_non_manifold_edges()
                    repaired.remove_degenerate_triangles()
                    break
                except Exception as e:
                    if attempt == 2:
                        break
            
            # Orient normals and recompute
            try:
                repaired.orient_triangles()
            except:
                pass
            
            repaired.compute_vertex_normals()
            repaired.compute_triangle_normals()
            
            # Final cleanup
            repaired.remove_duplicated_vertices()
            repaired.remove_duplicated_triangles()
            repaired.remove_degenerate_triangles()
            
            # Check watertightness
            try:
                self.is_watertight = repaired.is_watertight()
                self.attempted_repair = True
            except Exception as e:
                self.is_watertight = False
                self.attempted_repair = True
            
            return repaired
            
        except Exception as e:
            self.is_watertight = False
            self.attempted_repair = True
            return mesh
    
    def _calculate_volume_permissive(self) -> Optional[float]:
        """Calculate volume with multiple methods, even for 'non-watertight' meshes"""
        # Method 1: Open3D's built-in method
        try:
            volume = self.mesh.get_volume()
            if abs(volume) > 1e-12:
                return abs(volume)
        except Exception as e:
            pass
        
        # Method 2: Manual calculation using divergence theorem
        try:
            volume = self._calculate_volume_divergence_theorem()
            if volume is not None and abs(volume) > 1e-12:
                return abs(volume)
        except Exception as e:
            pass
        
        # Method 3: Tetrahedralization approach
        try:
            volume = self._calculate_volume_tetrahedralization()
            if volume is not None and abs(volume) > 1e-12:
                return abs(volume)
        except Exception as e:
            pass
        
        return None
    
    def _calculate_volume_divergence_theorem(self) -> Optional[float]:
        """Calculate volume using divergence theorem"""
        vertices = np.asarray(self.mesh.vertices)
        triangles = np.asarray(self.mesh.triangles)
        
        volume = 0.0
        for triangle in triangles:
            v0, v1, v2 = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]
            
            normal = np.cross(v1 - v0, v2 - v0)
            area = 0.5 * np.linalg.norm(normal)
            
            if area > 1e-12:
                normal = normal / (2 * area)
                centroid = (v0 + v1 + v2) / 3.0
                volume += (1.0/3.0) * area * np.dot(centroid, normal)
        
        return volume if abs(volume) > 1e-12 else None
    
    def _calculate_volume_tetrahedralization(self) -> Optional[float]:
        """Calculate volume by dividing into tetrahedra from centroid"""
        vertices = np.asarray(self.mesh.vertices)
        triangles = np.asarray(self.mesh.triangles)
        
        centroid = np.mean(vertices, axis=0)
        
        volume = 0.0
        for triangle in triangles:
            v0, v1, v2 = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]
            tetrahedron_volume = np.abs(np.dot(v0 - centroid, np.cross(v1 - centroid, v2 - centroid))) / 6.0
            volume += tetrahedron_volume
        
        return volume if volume > 1e-12 else None
    
    def _calculate_bounding_box_volume(self) -> float:
        """Calculate axis-aligned bounding box volume"""
        bbox = self.mesh.get_axis_aligned_bounding_box()
        extents = bbox.get_extent()
        return extents[0] * extents[1] * extents[2]
    
    def _calculate_convex_hull_volume(self) -> Optional[float]:
        """Calculate convex hull volume"""
        try:
            hull, _ = self.mesh.compute_convex_hull()
            return self._calculate_volume_for_mesh(hull)
        except Exception as e:
            return None
    
    def _calculate_volume_for_mesh(self, mesh: o3d.geometry.TriangleMesh) -> Optional[float]:
        """Calculate volume for any mesh using multiple methods"""
        try:
            volume = mesh.get_volume()
            if abs(volume) > 1e-12:
                return abs(volume)
        except:
            pass
        
        # Fallback to divergence theorem
        try:
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            
            volume = 0.0
            for triangle in triangles:
                v0, v1, v2 = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]
                normal = np.cross(v1 - v0, v2 - v0)
                area = 0.5 * np.linalg.norm(normal)
                
                if area > 1e-12:
                    normal = normal / (2 * area)
                    centroid = (v0 + v1 + v2) / 3.0
                    volume += (1.0/3.0) * area * np.dot(centroid, normal)
            
            return abs(volume) if abs(volume) > 1e-12 else None
        except:
            return None
    
    def _separate_objects(self):
        """Identify separate objects in the mesh"""
        try:
            self.objects = self._split_connected_components()
            self.total_objects = len(self.objects)
        except:
            self.objects = [self.mesh]
            self.total_objects = 1
    
    def _split_connected_components(self) -> List[o3d.geometry.TriangleMesh]:
        """Split mesh into connected components"""
        try:
            triangle_clusters, cluster_n_triangles, cluster_area = (
                self.mesh.cluster_connected_triangles()
            )
            triangle_clusters = np.asarray(triangle_clusters)
            cluster_n_triangles = np.asarray(cluster_n_triangles)
            
            objects = []
            vertices = np.asarray(self.mesh.vertices)
            triangles = np.asarray(self.mesh.triangles)
            
            for cluster_id in range(len(cluster_n_triangles)):
                if cluster_n_triangles[cluster_id] > 10:
                    cluster_triangles = triangles[triangle_clusters == cluster_id]
                    
                    used_vertices = np.unique(cluster_triangles.flatten())
                    cluster_vertices = vertices[used_vertices]
                    
                    vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}
                    remapped_triangles = np.array([[vertex_map[v] for v in tri] for tri in cluster_triangles])
                    
                    obj_mesh = o3d.geometry.TriangleMesh()
                    obj_mesh.vertices = o3d.utility.Vector3dVector(cluster_vertices)
                    obj_mesh.triangles = o3d.utility.Vector3iVector(remapped_triangles)
                    obj_mesh.compute_vertex_normals()
                    obj_mesh.compute_triangle_normals()
                    
                    objects.append(obj_mesh)
            
            return objects if objects else [self.mesh]
            
        except Exception as e:
            return [self.mesh]
    
    def analyze(self, accuracy: str = 'medium', 
                export_visualizations: bool = False) -> Dict:
        """Perform complete mesh analysis using original geometry for maximum accuracy"""
        analysis_start = time.time()
        
        try:
            # Raycast analysis with timing
            raycast_start = time.time()
            print(f"Performing raycast analysis (accuracy: {accuracy})...")
            self._perform_raycast(accuracy, export_visualizations)
            self.timing.raycast_analysis = time.time() - raycast_start
            print(f"Raycast completed ({self.timing.raycast_analysis:.2f}s)")
            
            # Safety checks
            if not hasattr(self, 'is_watertight'):
                self.is_watertight = False
            if not hasattr(self, 'attempted_repair'):
                self.attempted_repair = False
            
            results = {}
            
            # Dimensional analysis with timing
            dim_start = time.time()
            try:
                results["dimensions"] = self._analyze_dimensions()
            except Exception as e:
                results["dimensions"] = MeshDimensions(0, 0, 0, None, 0, False)
            self.timing.dimensional_analysis = time.time() - dim_start
            
            try:
                results["watertightness"] = self._analyze_watertightness()
            except Exception as e:
                results["watertightness"] = False
            
            try:
                results["watertight_details"] = {
                    "is_watertight": self.is_watertight,
                    "repair_attempted": self.attempted_repair,
                    "repair_successful": self.attempted_repair and self.is_watertight
                }
            except Exception as e:
                results["watertight_details"] = {"is_watertight": False, "repair_attempted": False, "repair_successful": False}
            
            try:
                results["separate_objects"] = self._analyze_objects()
            except Exception as e:
                results["separate_objects"] = {"total_objects": 1, "object_count": 1}
            
            try:
                results["format_conversion"] = self._format_conversion_results()
            except Exception as e:
                results["format_conversion"] = {"performed": False, "error": str(e)}
            
            # Wall thickness analysis with timing
            wall_start = time.time()
            try:
                results["wall_thickness"] = self._analyze_wall_thickness(export_visualizations)
            except Exception as e:
                results["wall_thickness"] = WallAnalysisResult(error=str(e))
            self.timing.wall_analysis = time.time() - wall_start
            
            # Gap analysis with timing
            gap_start = time.time()
            try:
                results["gaps"] = self._analyze_gaps(export_visualizations)
            except Exception as e:
                results["gaps"] = GapAnalysisResult(error=str(e))
            self.timing.gap_analysis = time.time() - gap_start
            
            # Update total timing
            self.timing.total_time = time.time() - analysis_start + self.timing.format_conversion + self.timing.mesh_repair + self.timing.analysis_setup
            
            try:
                results["metadata"] = {
                    "faces": len(self.mesh.triangles),
                    "vertices": len(self.mesh.vertices),
                    "analysis_time": self.timing.total_time,
                    "device_type": self.device_info['type'],
                    "device_name": self.device_info['name']
                }
            except Exception as e:
                results["metadata"] = {
                    "faces": 0,
                    "vertices": 0,
                    "analysis_time": self.timing.total_time,
                    "device_type": "Unknown",
                    "device_name": "Unknown"
                }
            
            # Add timing information
            results["timing"] = {
                "format_conversion": self.timing.format_conversion,
                "mesh_repair": self.timing.mesh_repair,
                "analysis_setup": self.timing.analysis_setup,
                "raycast_analysis": self.timing.raycast_analysis,
                "dimensional_analysis": self.timing.dimensional_analysis,
                "wall_analysis": self.timing.wall_analysis,
                "gap_analysis": self.timing.gap_analysis,
                "total_time": self.timing.total_time
            }
            
            print(f"Analysis completed ({self.timing.total_time:.2f}s total)")
            return results
            
        except Exception as e:
            self.timing.total_time = time.time() - analysis_start + self.timing.format_conversion + self.timing.mesh_repair + self.timing.analysis_setup
            
            return {
                "dimensions": MeshDimensions(0, 0, 0, None, 0, False),
                "watertightness": False,
                "watertight_details": {"is_watertight": False, "repair_attempted": False, "repair_successful": False},
                "separate_objects": {"total_objects": 1, "object_count": 1},
                "wall_thickness": WallAnalysisResult(error="Analysis failed"),
                "gaps": GapAnalysisResult(error="Analysis failed"),
                "metadata": {
                    "faces": 0,
                    "vertices": 0,
                    "analysis_time": self.timing.total_time,
                    "device_type": "Unknown",
                    "device_name": "Unknown",
                    "error": str(e)
                },
                "timing": {
                    "format_conversion": self.timing.format_conversion,
                    "mesh_repair": self.timing.mesh_repair,
                    "analysis_setup": self.timing.analysis_setup,
                    "raycast_analysis": self.timing.raycast_analysis,
                    "dimensional_analysis": self.timing.dimensional_analysis,
                    "wall_analysis": self.timing.wall_analysis,
                    "gap_analysis": self.timing.gap_analysis,
                    "total_time": self.timing.total_time
                }
            }
    
    def _format_conversion_results(self) -> Dict:
        """Format file conversion results for inclusion in analysis output"""
        if self.conversion_info is None:
            return {
                "performed": False,
                "reason": "Input file was already in OBJ format or conversion info not available"
            }
        
        return {
            "performed": True,
            "original_format": self.conversion_info.original_format,
            "format_name": self.conversion_info.format_name,
            "original_file": self.conversion_info.original_file,
            "converted_file": self.conversion_info.converted_file,
            "detected_unit": self.conversion_info.unit,
            "unit_assumed": self.conversion_info.unit_assumed,
            "target_format": ".obj"
        }
    
    def _perform_raycast(self, accuracy: str, export_mode: bool):
        """Perform raycast analysis (cached)"""
        if self._raycast_cache is None:
            sampled_faces = self.sampler.sample(accuracy)
            self._raycast_cache = self.raycast_engine.cast_rays(
                sampled_faces, export_mode
            )
    
    def _analyze_dimensions(self) -> MeshDimensions:
        """Analyze physical dimensions with comprehensive volume handling and additional metrics"""
        bbox = self.mesh.get_axis_aligned_bounding_box()
        extents = bbox.get_extent()
        
        # Calculate surface area
        surface_area = self._calculate_surface_area()
        
        # Calculate volumes
        actual_volume = self._calculate_volume_safe()
        bounding_box_volume = self._calculate_bounding_box_volume()
        convex_hull_volume = self._calculate_convex_hull_volume()
        
        # Calculate ratios
        surface_to_volume_ratio = None
        bounding_to_actual_volume_ratio = None
        concavity_ratio = None
        
        if actual_volume is not None and actual_volume > 1e-12:
            surface_to_volume_ratio = surface_area / actual_volume
            
            if bounding_box_volume > 1e-12:
                bounding_to_actual_volume_ratio = bounding_box_volume / actual_volume
            
            if convex_hull_volume is not None and convex_hull_volume > 1e-12:
                concavity_ratio = convex_hull_volume / actual_volume
        
        return MeshDimensions(
            length=extents[0],
            width=extents[1],
            height=extents[2],
            volume=actual_volume,
            surface_area=surface_area,
            is_watertight=self.is_watertight,
            surface_to_volume_ratio=surface_to_volume_ratio,
            bounding_to_actual_volume_ratio=bounding_to_actual_volume_ratio,
            concavity_ratio=concavity_ratio,
            bounding_box_volume=bounding_box_volume,
            convex_hull_volume=convex_hull_volume
        )
    
    def _calculate_surface_area(self) -> float:
        """Calculate surface area with fallback method"""
        try:
            return self.mesh.get_surface_area()
        except Exception as e:
            try:
                vertices = np.asarray(self.mesh.vertices)
                triangles = np.asarray(self.mesh.triangles)
                
                v0 = vertices[triangles[:, 0]]
                v1 = vertices[triangles[:, 1]]
                v2 = vertices[triangles[:, 2]]
                
                cross = np.cross(v1 - v0, v2 - v0)
                return 0.5 * np.sum(np.linalg.norm(cross, axis=1))
            except Exception as e2:
                return 0.0
    
    def _calculate_volume_safe(self) -> Optional[float]:
        """Calculate volume with error handling and multiple methods"""
        return self._calculate_volume_permissive()
    
    def _analyze_watertightness(self) -> bool:
        """Return the already-determined watertightness status"""
        return self.is_watertight
    
    def _analyze_objects(self) -> Dict:
        """Analyze separate objects"""
        return {
            "total_objects": self.total_objects,
            "object_count": len(self.objects)
        }
    
    def _analyze_wall_thickness(self, export: bool) -> WallAnalysisResult:
        """Analyze wall thickness"""
        try:
            distances = self._raycast_cache.inward_distances
            
            if not distances:
                return WallAnalysisResult(error="No thickness measurements found")
            
            analyzer = thinWallAnalysis.WallAnalyzer()
            
            thinnest = analyzer.analyze(
                ray_data=distances,
                plot=False,
                print_results=False, 
                find_significant=True
            )
            
            if export and self._raycast_cache.inward_face_data is not None:
                try:
                    Path("visualization").mkdir(exist_ok=True)
                    self.visualizer.export_colored_mesh(
                        self._raycast_cache.inward_face_data,
                        "visualization/thickness_model.ply",
                        "visualization/thickness_legend.png",
                        "Wall Thickness (units)",
                        "jet_r"
                    )
                except Exception as viz_error:
                    pass
            
            return WallAnalysisResult(
                thinnest_feature=thinnest,
                vertex_agreement=None
            )
            
        except Exception as e:
            return WallAnalysisResult(error=str(e))
    
    def _analyze_gaps(self, export: bool) -> GapAnalysisResult:
        """Analyze gaps between surfaces"""
        try:
            distances = self._raycast_cache.outward_distances
            
            if not distances:
                return GapAnalysisResult(error="No gap measurements found")
            
            analyzer = thinWallAnalysis.WallAnalyzer()
            
            smallest = analyzer.analyze(
                ray_data=distances,
                plot=False,
                print_results=False,
                find_significant=True
            )
            
            if export and self._raycast_cache.outward_face_data is not None:
                try:
                    Path("visualization").mkdir(exist_ok=True)
                    self.visualizer.export_colored_mesh(
                        self._raycast_cache.outward_face_data,
                        "visualization/gaps_model.ply",
                        "visualization/gaps_legend.png",
                        "Gap Size (units)",
                        "jet_r"
                    )
                except Exception as viz_error:
                    pass
            
            return GapAnalysisResult(
                smallest_gap=smallest,
                vertex_agreement=None
            )
            
        except Exception as e:
            return GapAnalysisResult(error=str(e))


class ResultsFormatter:
    """Formats analysis results for display"""
    
    @staticmethod
    def format_for_display(results: Dict) -> Dict:
        """Format results in a human-readable structure"""
        formatted = {}
        
        # Dimensions with enhanced metrics
        dims = results["dimensions"]
        formatted_dims = {
            "Length": f"{dims.length:.2f} units",
            "Width": f"{dims.width:.2f} units", 
            "Height": f"{dims.height:.2f} units",
            "Surface Area": f"{dims.surface_area:.2f} square units"
        }
        
        # Volume information
        if dims.volume is not None:
            if dims.is_watertight:
                formatted_dims["Volume"] = f"{dims.volume:.6f} cubic units (reliable)"
            else:
                formatted_dims["Volume"] = f"{dims.volume:.6f} cubic units (estimated)"
        else:
            formatted_dims["Volume"] = "Not available"
        
        # Enhanced geometric ratios
        if dims.surface_to_volume_ratio is not None:
            formatted_dims["Surface/Volume Ratio"] = f"{dims.surface_to_volume_ratio:.3f}"
        
        if dims.bounding_to_actual_volume_ratio is not None:
            formatted_dims["Bounding/Actual Volume Ratio"] = f"{dims.bounding_to_actual_volume_ratio:.3f}"
        
        if dims.concavity_ratio is not None:
            formatted_dims["Concavity Ratio"] = f"{dims.concavity_ratio:.3f} (convex hull / actual)"
        
        # Additional volume data
        if dims.bounding_box_volume is not None:
            formatted_dims["Bounding Box Volume"] = f"{dims.bounding_box_volume:.6f} cubic units"
        
        if dims.convex_hull_volume is not None:
            formatted_dims["Convex Hull Volume"] = f"{dims.convex_hull_volume:.6f} cubic units"
            
        formatted["Dimensions"] = formatted_dims
        
        # Basic properties
        watertight_details = results.get("watertight_details", {})
        watertight_status = "Yes" if results["watertightness"] else "No"
        
        if watertight_details.get("repair_attempted", False):
            if watertight_details.get("repair_successful", False):
                watertight_status += " (after repair)"
            else:
                watertight_status += " (repair failed)"
        
        formatted["Properties"] = {
            "Watertight": watertight_status,
            "Separate Objects": results["separate_objects"]["total_objects"]
        }
        
        # Format conversion information
        conversion = results.get("format_conversion", {})
        if conversion.get("performed", False):
            formatted["File Conversion"] = {
                "Original Format": f"{conversion.get('format_name', 'Unknown')} ({conversion.get('original_format', 'unknown')})",
                "Converted To": "Wavefront OBJ (.obj)",
                "Detected Units": f"{conversion.get('detected_unit', 'unknown')} (assumed: {conversion.get('unit_assumed', True)})"
            }
        else:
            formatted["File Conversion"] = {
                "Conversion Performed": "No"
            }
        
        # Manufacturing analysis
        formatted["Manufacturing Analysis"] = {}
        
        # Wall thickness
        wall = results["wall_thickness"]
        if wall.error:
            formatted["Manufacturing Analysis"]["Wall Thickness"] = f"Error: {wall.error}"
        elif wall.thinnest_feature is not None:
            formatted["Manufacturing Analysis"]["Thinnest Wall"] = f"{wall.thinnest_feature:.3f} units"
        else:
            formatted["Manufacturing Analysis"]["Thinnest Wall"] = "No data available"
        
        # Gaps  
        gaps = results["gaps"]
        if gaps.error:
            formatted["Manufacturing Analysis"]["Gaps"] = f"Error: {gaps.error}"
        elif gaps.smallest_gap is not None:
            formatted["Manufacturing Analysis"]["Smallest Gap"] = f"{gaps.smallest_gap:.3f} units"
        else:
            formatted["Manufacturing Analysis"]["Smallest Gap"] = "No data available"
        
        # Metadata
        meta = results["metadata"]
        formatted["Analysis Info"] = {
            "Mesh Complexity": f"{meta.get('faces', 0)} faces, {meta.get('vertices', 0)} vertices",
            "Analysis Time": f"{meta.get('analysis_time', 0):.2f} seconds",
            "Compute Device": f"{meta.get('device_type', 'Unknown')} ({meta.get('device_name', 'Unknown')})"
        }
        
        # Timing breakdown
        timing = results.get("timing", {})
        if timing:
            formatted["Timing Breakdown"] = {
                "Format Conversion": f"{timing.get('format_conversion', 0):.2f}s",
                "Mesh Repair": f"{timing.get('mesh_repair', 0):.2f}s",
                "Analysis Setup": f"{timing.get('analysis_setup', 0):.2f}s",
                "Raycast Analysis": f"{timing.get('raycast_analysis', 0):.2f}s",
                "Dimensional Analysis": f"{timing.get('dimensional_analysis', 0):.2f}s",
                "Wall Analysis": f"{timing.get('wall_analysis', 0):.2f}s",
                "Gap Analysis": f"{timing.get('gap_analysis', 0):.2f}s",
                "Total Time": f"{timing.get('total_time', 0):.2f}s"
            }
        
        return formatted


def main():
    """Command-line interface with automatic format conversion for maximum accuracy analysis"""
    # Get supported formats for help text
    supported_formats = MeshLoader.get_supported_formats()
    format_list = ', '.join(supported_formats.keys())
    
    parser = argparse.ArgumentParser(
        description=f'GPU-accelerated 3D mesh analysis with enhanced geometric calculations. Supports: {format_list}'
    )
    parser.add_argument('--file', '-i', required=True, help=f'Path to 3D model file. Supported formats: {format_list}')
    parser.add_argument('--output', '-o', help='Output JSON file path')
    parser.add_argument(
        '--accuracy', '-a', 
        choices=['low', 'medium', 'high', 'full'],
        default='medium', 
        help='Analysis accuracy level'
    )
    parser.add_argument(
        '--export', '-e', 
        action='store_true',
        help='Export colored visualizations'
    )
    parser.add_argument(
        '--cpu-only', '-c',
        action='store_true',
        help='Force CPU-only processing (disable GPU)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--no-repair',
        action='store_true',
        help='Skip mesh repair (repair is enabled by default)'
    )
    parser.add_argument(
        '--force-volume',
        action='store_true',
        help='Attempt volume calculation even for non-watertight meshes using alternative methods'
    )
    
    args = parser.parse_args()
    
    try:
        # Validate input file
        if not Path(args.file).exists():
            print(f"Error: File '{args.file}' not found")
            sys.exit(1)
        
        file_extension = Path(args.file).suffix.lower()
        supported_formats = MeshLoader.get_supported_formats()
        
        if file_extension not in supported_formats:
            print(f"Error: Unsupported file format '{file_extension}'")
            print(f"Supported formats: {', '.join(supported_formats.keys())}")
            sys.exit(1)
        
        if file_extension != '.obj' and not FORMAT_CONVERTER_AVAILABLE:
            print(f"Error: Format converter not available for {file_extension} files")
            sys.exit(1)
        
        # Setup logging
        if not args.verbose:
            o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        
        # Run analysis
        use_gpu = not args.cpu_only
        repair_mesh = not args.no_repair
        
        try:
            analyzer = MeshAnalyzer(
                args.file, 
                use_gpu=use_gpu,
                repair_mesh=repair_mesh,
                force_volume=args.force_volume
            )
        except Exception as e:
            print(f"Error creating analyzer: {e}")
            analyzer = MeshAnalyzer(args.file, use_gpu=use_gpu)
        
        results = analyzer.analyze(
            accuracy=args.accuracy,
            export_visualizations=args.export
        )
        
        # Format and display results
        formatted = ResultsFormatter.format_for_display(results)
        print("\n" + "="*50)
        print(json.dumps(formatted, indent=2))
        
        # Save if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(formatted, f, indent=2)
            print(f"\nResults saved to: {args.output}")
            
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()