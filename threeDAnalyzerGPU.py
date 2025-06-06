"""
3D Mesh Analyzer - GPU-Accelerated Version with Open3D

A high-performance implementation for analyzing 3D mesh files (.obj format)
with GPU-accelerated raycasting using Open3D.
"""

import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path
import time
import json
import argparse
import sys

# Import the external thin wall analysis module
import thinWallAnalysis


@dataclass
class MeshDimensions:
    """Container for mesh physical dimensions"""
    length: float
    width: float
    height: float
    volume: Optional[float]  # Optional since non-watertight meshes can't have volume
    surface_area: float
    is_watertight: bool


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


class GPUManager:
    """Manages GPU device selection and configuration for Open3D"""
    
    @staticmethod
    def setup_gpu():
        """Setup GPU acceleration if available"""
        try:
            # Check for CUDA support
            if o3d.core.cuda.device_count() > 0:
                device = o3d.core.Device("CUDA:0")
                print(f"GPU acceleration enabled: {device}")
                return device
            else:
                print("CUDA not available, using CPU")
                return o3d.core.Device("CPU:0")
        except Exception as e:
            print(f"GPU setup failed, falling back to CPU: {e}")
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
    """Handles loading 3D mesh files using Open3D"""
    
    @staticmethod
    def load(file_input: Union[str, Path, bytes]) -> o3d.geometry.TriangleMesh:
        """
        Load a mesh from file path or bytes
        
        Args:
            file_input: Path to file or bytes data
            
        Returns:
            Open3D mesh object
            
        Raises:
            ValueError: If mesh cannot be loaded or is invalid
        """
        if isinstance(file_input, bytes):
            return MeshLoader._load_from_bytes(file_input)
        else:
            return MeshLoader._load_from_path(file_input)
    
    @staticmethod
    def _load_from_path(file_path: Union[str, Path]) -> o3d.geometry.TriangleMesh:
        """Load mesh from file path"""
        try:
            mesh = o3d.io.read_triangle_mesh(str(file_path))
            MeshLoader._validate_mesh(mesh)
            MeshLoader._prepare_mesh(mesh)
            return mesh
        except Exception as e:
            raise ValueError(f"Failed to load .obj model: {e}")
    
    @staticmethod
    def _load_from_bytes(file_bytes: bytes) -> o3d.geometry.TriangleMesh:
        """Load mesh from bytes"""
        try:
            # Write bytes to temporary file for Open3D
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as tmp:
                tmp.write(file_bytes)
                tmp.flush()
                mesh = o3d.io.read_triangle_mesh(tmp.name)
                Path(tmp.name).unlink()  # Clean up
            
            MeshLoader._validate_mesh(mesh)
            MeshLoader._prepare_mesh(mesh)
            return mesh
        except Exception as e:
            raise ValueError(f"Failed to load .obj model from bytes: {e}")
    
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
        # Compute vertex normals if not present
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        
        # Compute triangle normals if not present
        if not mesh.has_triangle_normals():
            mesh.compute_triangle_normals()
        
        # Remove duplicated vertices and triangles
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        
        print(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")


class FaceSampler:
    """Handles intelligent face sampling for large meshes"""
    
    def __init__(self, mesh: o3d.geometry.TriangleMesh):
        self.mesh = mesh
        self.total_faces = len(mesh.triangles)
        self._triangle_areas = None
        self._triangle_centers = None
        
    def sample(self, accuracy: str = 'medium') -> np.ndarray:
        """
        Sample faces based on accuracy level
        
        Args:
            accuracy: One of 'low', 'medium', 'high', 'full'
            
        Returns:
            Array of sampled face indices
        """
        if accuracy == 'full':
            return np.arange(self.total_faces)
        
        sample_rate = self._get_sample_rate(accuracy)
        num_samples = max(100, int(self.total_faces * sample_rate))
        
        print(f"Sampling {num_samples} faces from {self.total_faces} total faces")
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
            
            # Get triangle vertices
            v0 = vertices[triangles[:, 0]]
            v1 = vertices[triangles[:, 1]]
            v2 = vertices[triangles[:, 2]]
            
            # Compute areas using cross product
            cross = np.cross(v1 - v0, v2 - v0)
            self._triangle_areas = 0.5 * np.linalg.norm(cross, axis=1)
        
        return self._triangle_areas
    
    def _get_triangle_centers(self) -> np.ndarray:
        """Compute triangle centers if not cached"""
        if self._triangle_centers is None:
            vertices = np.asarray(self.mesh.vertices)
            triangles = np.asarray(self.mesh.triangles)
            
            # Get triangle vertices
            v0 = vertices[triangles[:, 0]]
            v1 = vertices[triangles[:, 1]]
            v2 = vertices[triangles[:, 2]]
            
            # Compute centers
            self._triangle_centers = (v0 + v1 + v2) / 3.0
        
        return self._triangle_centers
    
    def _spatial_sampling(self, num_samples: int) -> np.ndarray:
        """Perform spatially-aware sampling for better coverage"""
        # Create spatial grid
        grid_size = max(2, int(np.ceil(num_samples ** (1.0/3))))
        grid_cells = self._assign_faces_to_grid(grid_size)
        
        # Sample from each cell
        sampled_indices = []
        target_per_cell = max(1, num_samples // len(grid_cells))
        
        for cell_faces in grid_cells.values():
            cell_sampled = self._sample_from_cell(cell_faces, target_per_cell)
            sampled_indices.extend(cell_sampled)
        
        # Ensure minimum samples
        sampled_indices = self._ensure_minimum_samples(
            sampled_indices, num_samples
        )
        
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
        
        # Handle zero weights
        if np.sum(weights) == 0:
            num_samples = min(target_samples, len(indices))
            if num_samples > 0:
                return np.random.choice(
                    indices, size=num_samples, replace=False
                ).tolist()
            return []
        
        # Filter out zero weights for weighted sampling
        non_zero_mask = weights > 0
        non_zero_indices = np.array(indices)[non_zero_mask]
        non_zero_weights = weights[non_zero_mask]
        
        if len(non_zero_indices) == 0:
            num_samples = min(target_samples, len(indices))
            if num_samples > 0:
                return np.random.choice(
                    indices, size=num_samples, replace=False
                ).tolist()
            return []
        
        # Normalize weights and sample
        non_zero_weights = non_zero_weights / np.sum(non_zero_weights)
        num_samples = min(target_samples, len(non_zero_indices))
        
        if num_samples > 0:
            return np.random.choice(
                non_zero_indices, size=num_samples, replace=False, p=non_zero_weights
            ).tolist()
        
        return []
    
    def _ensure_minimum_samples(self, sampled: List[int], 
                               target: int) -> List[int]:
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
            # Create RaycastingScene for GPU-accelerated raycasting
            self.scene = o3d.t.geometry.RaycastingScene()
            
            # Convert legacy mesh to tensor mesh for GPU operations
            mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh, device=self.device)
            
            # Add mesh to scene
            self.scene.add_triangles(mesh_t)
            
            print(f"Raycasting scene initialized on {self.device}")
            
        except Exception as e:
            print(f"GPU raycasting setup failed: {e}")
            # Fallback to CPU
            self.device = o3d.core.Device("CPU:0")
            mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh, device=self.device)
            self.scene = o3d.t.geometry.RaycastingScene()
            self.scene.add_triangles(mesh_t)
    
    def cast_rays(self, face_indices: np.ndarray, 
                  export_mode: bool = False) -> RaycastResult:
        """
        Cast rays inward and outward from specified faces using GPU/CPU
        
        Args:
            face_indices: Indices of faces to cast rays from
            export_mode: Whether to store per-face data for export
            
        Returns:
            RaycastResult containing distance measurements
        """
        max_distance = float(np.max(self.mesh.get_axis_aligned_bounding_box().get_extent()) * 0.5)
        print(f"Using max ray distance: {max_distance:.3f} units")
        
        # Get face centers and normals
        face_centers, face_normals = self._get_face_properties(face_indices)
        
        # Determine device type for progress reporting
        device_type = "GPU" if self.device.get_type() == o3d.core.Device.DeviceType.CUDA else "CPU"
        print(f"Casting rays for {len(face_indices)} faces on {self.device}")
        start_time = time.time()
        
        # Cast rays in batches for memory efficiency
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
            
            # Cast inward rays
            self._cast_batch_rays(
                batch_centers, -batch_normals, batch_indices, max_distance,
                result.inward_distances, result.inward_face_data, export_mode
            )
            
            # Cast outward rays
            self._cast_batch_rays(
                batch_centers, batch_normals, batch_indices, max_distance,
                result.outward_distances, result.outward_face_data, export_mode
            )
            
            # Progress reporting with correct device type
            if num_batches > 10 and batch_idx % max(1, num_batches // 10) == 0:
                progress = (batch_idx / num_batches) * 100
                print(f"{device_type} raycast progress: {progress:.1f}%")
        
        elapsed = time.time() - start_time
        print(f"{device_type} raycasting completed in {elapsed:.2f}s")
        print(f"Found {len(result.inward_distances)} inward and {len(result.outward_distances)} outward measurements")
        
        # Post-process data if needed
        if export_mode:
            result.inward_face_data = self._interpolate_missing_values(
                result.inward_face_data
            )
            result.outward_face_data = self._interpolate_missing_values(
                result.outward_face_data
            )
        
        return result
    
    def _get_optimal_batch_size(self) -> int:
        """Determine optimal batch size based on device memory"""
        if self.device.get_type() == o3d.core.Device.DeviceType.CUDA:
            # GPU: Use larger batches for better GPU utilization
            return min(10000, max(1000, len(self.mesh.triangles) // 10))
        else:
            # CPU: Use smaller batches to avoid memory issues
            return min(5000, max(500, len(self.mesh.triangles) // 20))
    
    def _get_face_properties(self, face_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get face centers and normals for specified faces"""
        vertices = np.asarray(self.mesh.vertices)
        triangles = np.asarray(self.mesh.triangles)
        triangle_normals = np.asarray(self.mesh.triangle_normals)
        
        # Get selected triangles
        selected_triangles = triangles[face_indices]
        
        # Compute face centers
        v0 = vertices[selected_triangles[:, 0]]
        v1 = vertices[selected_triangles[:, 1]]
        v2 = vertices[selected_triangles[:, 2]]
        face_centers = (v0 + v1 + v2) / 3.0
        
        # Get face normals
        face_normals = triangle_normals[face_indices]
        
        return face_centers, face_normals
    
    def _cast_batch_rays(self, origins: np.ndarray, directions: np.ndarray,
                        face_indices: np.ndarray, max_distance: float,
                        measurements: List[float], face_data: Optional[np.ndarray],
                        export_mode: bool):
        """Cast a batch of rays using GPU acceleration"""
        try:
            # Convert to Open3D tensors on the correct device
            ray_origins = o3d.core.Tensor(origins, dtype=o3d.core.float32, device=self.device)
            ray_directions = o3d.core.Tensor(directions, dtype=o3d.core.float32, device=self.device)
            
            # Perform raycasting
            rays = o3d.core.concatenate([ray_origins, ray_directions], axis=1)
            ans = self.scene.cast_rays(rays)
            
            # Extract results
            hit_distances = ans['t_hit'].cpu().numpy()
            geometry_ids = ans['geometry_ids'].cpu().numpy()
            
            # Process hits
            for i, (distance, geom_id) in enumerate(zip(hit_distances, geometry_ids)):
                # Check if ray hit something and distance is valid
                if geom_id != self.scene.INVALID_ID and distance > 1e-3 and distance < max_distance:
                    measurements.append(distance)
                    
                    if export_mode and face_data is not None:
                        global_idx = face_indices[i]
                        face_data[global_idx] = distance
                        
        except Exception as e:
            print(f"Error in GPU ray casting batch: {e}")
            # Fallback to CPU-based raycasting for this batch
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
            # Use Open3D's legacy raycasting on CPU
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
            print(f"Fallback raycasting also failed: {e}")
    
    def _interpolate_missing_values(self, data: np.ndarray) -> np.ndarray:
        """Interpolate missing values using nearest neighbors"""
        if not np.any(np.isnan(data)):
            return data
        
        valid_mask = ~np.isnan(data)
        if not np.any(valid_mask):
            return data
        
        try:
            # Get triangle centers for spatial interpolation
            face_centers = self._get_triangle_centers()
            valid_indices = np.where(valid_mask)[0]
            valid_centers = face_centers[valid_indices]
            valid_values = data[valid_indices]
            
            tree = cKDTree(valid_centers)
            missing_indices = np.where(np.isnan(data))[0]
            
            for idx in missing_indices:
                center = face_centers[idx]
                distances, neighbors = tree.query(center, k=min(3, len(valid_centers)))
                
                # Weighted average
                if np.all(distances > 0):
                    weights = 1.0 / distances
                    data[idx] = np.average(valid_values[neighbors], weights=weights)
                else:
                    data[idx] = valid_values[neighbors[0]]
                    
        except Exception as e:
            print(f"Interpolation failed: {e}, using median")
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
            # Create output directory
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate colors
            face_colors = self._map_values_to_colors(face_data, colormap)
            vertex_colors = self._propagate_face_colors_to_vertices(face_colors)
            
            # Create colored mesh
            colored_mesh = self.mesh.__copy__()
            colored_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors / 255.0)
            
            # Export
            o3d.io.write_triangle_mesh(output_path, colored_mesh)
            print(f"Exported colored mesh: {output_path}")
            
            # Create legend
            self._create_legend(face_data, legend_path, title, colormap)
            
        except Exception as e:
            print(f"Error exporting visualization: {e}")
    
    def _map_values_to_colors(self, values: np.ndarray, 
                             colormap: str) -> np.ndarray:
        """Map scalar values to RGB colors"""
        # Remove NaN values for normalization
        valid_values = values[~np.isnan(values)]
        if len(valid_values) == 0:
            return np.zeros((len(values), 3), dtype=np.uint8)
        
        # Normalize values
        vmin, vmax = np.min(valid_values), np.max(valid_values)
        normalized = np.zeros_like(values)
        valid_mask = ~np.isnan(values)
        
        if vmax > vmin:
            normalized[valid_mask] = (values[valid_mask] - vmin) / (vmax - vmin)
        
        # Apply colormap
        cmap = plt.get_cmap(colormap)
        colors = cmap(normalized)
        
        # Convert to RGB bytes
        return (colors[:, :3] * 255).astype(np.uint8)
    
    def _propagate_face_colors_to_vertices(self, 
                                          face_colors: np.ndarray) -> np.ndarray:
        """Average face colors to vertices"""
        vertices = np.asarray(self.mesh.vertices)
        triangles = np.asarray(self.mesh.triangles)
        
        vertex_colors = np.zeros((len(vertices), 3), dtype=np.float64)
        vertex_counts = np.zeros(len(vertices), dtype=int)
        
        # Accumulate colors
        for face_idx, face in enumerate(triangles):
            for vertex_idx in face:
                vertex_colors[vertex_idx] += face_colors[face_idx]
                vertex_counts[vertex_idx] += 1
        
        # Average
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
        
        # Create colorbar
        norm = Normalize(vmin=np.min(valid_values), vmax=np.max(valid_values))
        sm = cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(colormap))
        sm.set_array([])
        
        cbar = fig.colorbar(sm, ax=ax, orientation='horizontal')
        cbar.set_label(title, fontsize=12)
        
        # Add statistics
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
        
        print(f"Saved legend: {output_path}")


class MeshAnalyzer:
    """Main analyzer class that orchestrates all analysis operations"""
    
    def __init__(self, file_input: Union[str, Path, bytes], 
                 use_gpu: bool = True, repair_mesh: bool = True):
        """
        Initialize analyzer with mesh file
        
        Args:
            file_input: Path to mesh file or bytes data
            use_gpu: Whether to use GPU acceleration
            repair_mesh: Whether to attempt mesh repair for better watertightness
        """
        self.mesh = MeshLoader.load(file_input)
        
        # Attempt mesh repair if requested
        if repair_mesh:
            self.mesh = self._repair_mesh(self.mesh)
        
        self.device = GPUManager.setup_gpu() if use_gpu else o3d.core.Device("CPU:0")
        self.device_info = GPUManager.get_device_info(self.device)
        
        self.sampler = FaceSampler(self.mesh)
        self.raycast_engine = GPURaycastEngine(self.mesh, self.device)
        self.visualizer = VisualizationExporter(self.mesh)
        self._separate_objects()
        self._raycast_cache = None
        
        print(f"Analyzer initialized with {self.device_info['type']}: {self.device_info['name']}")
    
    def _repair_mesh(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """Attempt to repair mesh for better watertightness"""
        try:
            print("Attempting mesh repair...")
            original_faces = len(mesh.triangles)
            original_vertices = len(mesh.vertices)
            
            # Create a copy for repair
            repaired = mesh.__copy__()
            
            # Remove duplicates and degenerate triangles
            repaired.remove_duplicated_vertices()
            repaired.remove_duplicated_triangles()
            repaired.remove_degenerate_triangles()
            
            # Try to remove non-manifold edges
            try:
                repaired.remove_non_manifold_edges()
            except:
                print("  Non-manifold edge removal failed, continuing...")
            
            # Merge close vertices
            repaired.merge_close_vertices(1e-6)
            
            # Recompute normals
            repaired.compute_vertex_normals()
            repaired.compute_triangle_normals()
            
            new_faces = len(repaired.triangles)
            new_vertices = len(repaired.vertices)
            
            print(f"  Mesh repair completed:")
            print(f"    Vertices: {original_vertices} → {new_vertices}")
            print(f"    Faces: {original_faces} → {new_faces}")
            
            # Check if repair improved watertightness
            try:
                is_watertight_before = mesh.is_watertight()
                is_watertight_after = repaired.is_watertight()
                print(f"    Watertight: {is_watertight_before} → {is_watertight_after}")
            except:
                pass
            
            return repaired
            
        except Exception as e:
            print(f"Mesh repair failed: {e}")
            return mesh
        
    def _separate_objects(self):
        """Identify separate objects in the mesh"""
        try:
            # Open3D doesn't have built-in mesh splitting like trimesh
            # We'll use connected components analysis
            self.objects = self._split_connected_components()
            self.total_objects = len(self.objects)
        except:
            self.objects = [self.mesh]
            self.total_objects = 1
    
    def _split_connected_components(self) -> List[o3d.geometry.TriangleMesh]:
        """Split mesh into connected components"""
        try:
            # Get connected components
            triangle_clusters, cluster_n_triangles, cluster_area = (
                self.mesh.cluster_connected_triangles()
            )
            triangle_clusters = np.asarray(triangle_clusters)
            cluster_n_triangles = np.asarray(cluster_n_triangles)
            
            # Create separate meshes for each component
            objects = []
            vertices = np.asarray(self.mesh.vertices)
            triangles = np.asarray(self.mesh.triangles)
            
            for cluster_id in range(len(cluster_n_triangles)):
                if cluster_n_triangles[cluster_id] > 10:  # Filter small components
                    cluster_triangles = triangles[triangle_clusters == cluster_id]
                    
                    # Get unique vertices used by this cluster
                    used_vertices = np.unique(cluster_triangles.flatten())
                    cluster_vertices = vertices[used_vertices]
                    
                    # Remap triangle indices
                    vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}
                    remapped_triangles = np.array([[vertex_map[v] for v in tri] for tri in cluster_triangles])
                    
                    # Create mesh
                    obj_mesh = o3d.geometry.TriangleMesh()
                    obj_mesh.vertices = o3d.utility.Vector3dVector(cluster_vertices)
                    obj_mesh.triangles = o3d.utility.Vector3iVector(remapped_triangles)
                    obj_mesh.compute_vertex_normals()
                    obj_mesh.compute_triangle_normals()
                    
                    objects.append(obj_mesh)
            
            return objects if objects else [self.mesh]
            
        except Exception as e:
            print(f"Failed to split connected components: {e}")
            return [self.mesh]
    
    def analyze(self, accuracy: str = 'medium', 
                export_visualizations: bool = False) -> Dict:
        """
        Perform complete mesh analysis
        
        Args:
            accuracy: Analysis accuracy level
            export_visualizations: Whether to export colored visualizations
            
        Returns:
            Dictionary containing all analysis results
        """
        device_name = "GPU-accelerated" if self.device_info['type'] == 'GPU' else "CPU-based"
        print(f"\nStarting {device_name} mesh analysis (accuracy: {accuracy})")
        start_time = time.time()
        
        try:
            # Perform raycast (cached for efficiency)
            print("Starting raycast analysis...")
            self._perform_raycast(accuracy, export_visualizations)
            print("Raycast analysis completed")
            
            # Safety checks for attributes
            if not hasattr(self, 'is_watertight'):
                self.is_watertight = False
            if not hasattr(self, 'attempted_repair'):
                self.attempted_repair = False
            
            print("Gathering analysis results...")
            
            # Collect all analyses with individual error handling
            results = {}
            
            try:
                print("  Analyzing dimensions...")
                results["dimensions"] = self._analyze_dimensions()
            except Exception as e:
                print(f"  Error in dimensions analysis: {e}")
                results["dimensions"] = MeshDimensions(0, 0, 0, None, 0, False)
            
            try:
                print("  Analyzing watertightness...")
                results["watertightness"] = self._analyze_watertightness()
            except Exception as e:
                print(f"  Error in watertightness analysis: {e}")
                results["watertightness"] = False
            
            try:
                print("  Gathering watertight details...")
                results["watertight_details"] = {
                    "is_watertight": self.is_watertight,
                    "repair_attempted": self.attempted_repair,
                    "repair_successful": self.attempted_repair and self.is_watertight
                }
            except Exception as e:
                print(f"  Error in watertight details: {e}")
                results["watertight_details"] = {"is_watertight": False, "repair_attempted": False, "repair_successful": False}
            
            try:
                print("  Analyzing separate objects...")
                results["separate_objects"] = self._analyze_objects()
            except Exception as e:
                print(f"  Error in objects analysis: {e}")
                results["separate_objects"] = {"total_objects": 1, "object_count": 1}
            
            try:
                print("  Analyzing wall thickness...")
                results["wall_thickness"] = self._analyze_wall_thickness(export_visualizations)
            except Exception as e:
                print(f"  Error in wall thickness analysis: {e}")
                results["wall_thickness"] = WallAnalysisResult(error=str(e))
            
            try:
                print("  Analyzing gaps...")
                results["gaps"] = self._analyze_gaps(export_visualizations)
            except Exception as e:
                print(f"  Error in gaps analysis: {e}")
                results["gaps"] = GapAnalysisResult(error=str(e))
            
            
            try:
                print("  Gathering metadata...")
                results["metadata"] = {
                    "faces": len(self.mesh.triangles),
                    "vertices": len(self.mesh.vertices),
                    "analysis_time": time.time() - start_time,
                    "device_type": self.device_info['type'],
                    "device_name": self.device_info['name']
                }
            except Exception as e:
                print(f"  Error in metadata: {e}")
                results["metadata"] = {
                    "faces": 0,
                    "vertices": 0,
                    "analysis_time": time.time() - start_time,
                    "device_type": "Unknown",
                    "device_name": "Unknown"
                }
            
            print("All analysis components completed")
            print(f"Analysis completed in {results['metadata']['analysis_time']:.2f}s using {self.device_info['type']}")
            return results
            
        except Exception as e:
            print(f"Critical error in analysis: {e}")
            import traceback
            traceback.print_exc()
            
            # Return minimal results to prevent complete failure
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
                    "analysis_time": time.time() - start_time,
                    "device_type": "Unknown",
                    "device_name": "Unknown",
                    "error": str(e)
                }
            }
    
    def _perform_raycast(self, accuracy: str, export_mode: bool):
        """Perform raycast analysis (cached)"""
        if self._raycast_cache is None:
            sampled_faces = self.sampler.sample(accuracy)
            self._raycast_cache = self.raycast_engine.cast_rays(
                sampled_faces, export_mode
            )
    
    def _analyze_dimensions(self) -> MeshDimensions:
        """Analyze physical dimensions with smart volume handling"""
        bbox = self.mesh.get_axis_aligned_bounding_box()
        extents = bbox.get_extent()
        
        # Calculate surface area (always possible)
        surface_area = self._calculate_surface_area()
        
        # Calculate volume only if watertight
        volume = None
        if self.is_watertight:
            volume = self._calculate_volume_safe()
            if volume is not None:
                print(f"Volume calculated: {volume:.3f} cubic units")
            else:
                print("Volume calculation failed despite watertight mesh")
        else:
            print("Skipping volume calculation - mesh is not watertight")
        
        return MeshDimensions(
            length=extents[0],
            width=extents[1],
            height=extents[2],
            volume=volume,
            surface_area=surface_area,
            is_watertight=self.is_watertight
        )
    
    def _calculate_surface_area(self) -> float:
        """Calculate surface area with fallback method"""
        try:
            return self.mesh.get_surface_area()
        except Exception as e:
            print(f"Open3D surface area calculation failed: {e}, using fallback")
            # Fallback manual calculation
            try:
                vertices = np.asarray(self.mesh.vertices)
                triangles = np.asarray(self.mesh.triangles)
                
                v0 = vertices[triangles[:, 0]]
                v1 = vertices[triangles[:, 1]]
                v2 = vertices[triangles[:, 2]]
                
                cross = np.cross(v1 - v0, v2 - v0)
                return 0.5 * np.sum(np.linalg.norm(cross, axis=1))
            except Exception as e2:
                print(f"Fallback surface area calculation failed: {e2}")
                return 0.0
    
    def _calculate_volume_safe(self) -> Optional[float]:
        """Calculate volume with error handling - only call if watertight"""
        try:
            volume = self.mesh.get_volume()
            return abs(volume) if volume != 0 else None
        except Exception as e:
            print(f"Volume calculation failed: {e}")
            return None
    
    def _calculate_volume_robust(self) -> float:
        """Calculate volume with multiple fallback methods"""
        # Method 1: Try Open3D's built-in volume calculation
        try:
            volume = self.mesh.get_volume()
            if volume > 0:  # Valid volume
                return abs(volume)  # Ensure positive
        except Exception as e:
            print(f"Open3D volume calculation failed: {e}")
        
        # Method 2: Try after mesh repair
        try:
            # Create a copy and try to repair it
            repaired_mesh = self.mesh.__copy__()
            repaired_mesh.remove_duplicated_vertices()
            repaired_mesh.remove_duplicated_triangles()
            repaired_mesh.remove_degenerate_triangles()
            repaired_mesh.remove_non_manifold_edges()
            
            volume = repaired_mesh.get_volume()
            if volume > 0:
                return abs(volume)
        except Exception as e:
            print(f"Repaired mesh volume calculation failed: {e}")
        
        # Method 3: Manual volume calculation using divergence theorem
        try:
            volume = self._calculate_volume_manual()
            if volume > 0:
                return abs(volume)
        except Exception as e:
            print(f"Manual volume calculation failed: {e}")
        
        # Method 4: Approximate volume using bounding box and density
        try:
            bbox = self.mesh.get_axis_aligned_bounding_box()
            bbox_volume = bbox.volume()
            
            # Estimate density by sampling points
            vertices = np.asarray(self.mesh.vertices)
            bbox_min = bbox.min_bound
            bbox_max = bbox.max_bound
            
            # Sample points in bounding box
            n_samples = 10000
            sample_points = np.random.uniform(bbox_min, bbox_max, (n_samples, 3))
            
            # Check which points are inside the mesh
            inside_count = 0
            for point in sample_points[:1000]:  # Limit for performance
                if self._point_in_mesh(point):
                    inside_count += 1
            
            density = inside_count / 1000
            approximate_volume = bbox_volume * density
            
            if approximate_volume > 0:
                print(f"Using approximate volume calculation: {approximate_volume:.3f}")
                return approximate_volume
                
        except Exception as e:
            print(f"Approximate volume calculation failed: {e}")
        
        # Method 5: Return 0 if all methods fail
        print("Warning: All volume calculation methods failed, returning 0")
        return 0.0
    
    def _calculate_volume_manual(self) -> float:
        """Manual volume calculation using divergence theorem"""
        vertices = np.asarray(self.mesh.vertices)
        triangles = np.asarray(self.mesh.triangles)
        
        if len(triangles) == 0:
            return 0.0
        
        # Get triangle vertices
        v0 = vertices[triangles[:, 0]]
        v1 = vertices[triangles[:, 1]]
        v2 = vertices[triangles[:, 2]]
        
        # Calculate triangle centers and normals
        centers = (v0 + v1 + v2) / 3.0
        
        # Calculate cross product for normal and area
        cross = np.cross(v1 - v0, v2 - v0)
        normals = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + 1e-12)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        
        # Volume calculation using divergence theorem
        # V = (1/3) * sum(dot(center, normal) * area)
        volume = np.sum(np.sum(centers * normals, axis=1) * areas) / 3.0
        
        return abs(volume)
    
    def _point_in_mesh(self, point: np.ndarray) -> bool:
        """Check if a point is inside the mesh using ray casting"""
        try:
            # Cast a ray from the point in a random direction
            direction = np.array([1.0, 0.0, 0.0])  # Simple direction
            
            # Use Open3D's raycasting scene
            ray_origins = o3d.core.Tensor([point], dtype=o3d.core.float32, device=o3d.core.Device("CPU:0"))
            ray_directions = o3d.core.Tensor([direction], dtype=o3d.core.float32, device=o3d.core.Device("CPU:0"))
            
            rays = o3d.core.concatenate([ray_origins, ray_directions], axis=1)
            
            # Create a simple raycasting scene
            scene = o3d.t.geometry.RaycastingScene()
            mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh, device=o3d.core.Device("CPU:0"))
            scene.add_triangles(mesh_t)
            
            ans = scene.cast_rays(rays)
            hit_distances = ans['t_hit'].cpu().numpy()
            geometry_ids = ans['geometry_ids'].cpu().numpy()
            
            # Count intersections
            valid_hits = np.sum((geometry_ids != scene.INVALID_ID) & (hit_distances > 1e-6))
            
            # Point is inside if odd number of intersections
            return (valid_hits % 2) == 1
            
        except:
            return False
    
    def _analyze_watertightness(self) -> bool:
        """Return the already-determined watertightness status"""
        return self.is_watertight
    
    def _analyze_objects(self) -> Dict:
        """Analyze separate objects (no volume calculation)"""
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
            
            print(f"  Found {len(distances)} inward ray measurements")
            
            # Use external analyzer with proper error handling
            analyzer = thinWallAnalysis.WallAnalyzer()
            
            # Disable plotting to avoid Windows matplotlib errors
            thinnest = analyzer.analyze(
                ray_data=distances,
                plot=True,  # Disable plotting to avoid directory/matplotlib errors
                print_results=False, 
                find_significant=True
            )
            
            print(f"  Thinnest wall analysis result: {thinnest}")
            
            # Export visualization if requested
            if export and self._raycast_cache.inward_face_data is not None:
                try:
                    # Ensure visualization directory exists
                    Path("visualization").mkdir(exist_ok=True)
                    self.visualizer.export_colored_mesh(
                        self._raycast_cache.inward_face_data,
                        "visualization/thickness_model.ply",
                        "visualization/thickness_legend.png",
                        "Wall Thickness (units)",
                        "jet_r"
                    )
                except Exception as viz_error:
                    print(f"    Warning: Visualization export failed: {viz_error}")
            
            return WallAnalysisResult(
                thinnest_feature=thinnest,
                vertex_agreement=None
            )
            
        except Exception as e:
            print(f"  Wall thickness analysis failed: {e}")
            return WallAnalysisResult(error=str(e))
    
    def _analyze_gaps(self, export: bool) -> GapAnalysisResult:
        """Analyze gaps between surfaces"""
        try:
            distances = self._raycast_cache.outward_distances
            
            if not distances:
                return GapAnalysisResult(error="No gap measurements found")
            
            print(f"  Found {len(distances)} outward ray measurements")
            
            # Use external analyzer with proper error handling
            analyzer = thinWallAnalysis.WallAnalyzer()
            
            # Disable plotting to avoid Windows matplotlib errors
            smallest = analyzer.analyze(
                ray_data=distances,
                plot=False,  # Disable plotting to avoid directory/matplotlib errors
                print_results=True,
                find_significant=True
            )
            
            print(f"  Smallest gap analysis result: {smallest}")
            
            # Export visualization if requested
            if export and self._raycast_cache.outward_face_data is not None:
                try:
                    # Ensure visualization directory exists
                    Path("visualization").mkdir(exist_ok=True)
                    self.visualizer.export_colored_mesh(
                        self._raycast_cache.outward_face_data,
                        "visualization/gaps_model.ply",
                        "visualization/gaps_legend.png",
                        "Gap Size (units)",
                        "jet_r"
                    )
                except Exception as viz_error:
                    print(f"    Warning: Visualization export failed: {viz_error}")
            
            return GapAnalysisResult(
                smallest_gap=smallest,
                vertex_agreement=None
            )
            
        except Exception as e:
            print(f"  Gap analysis failed: {e}")
            return GapAnalysisResult(error=str(e))
    


class ResultsFormatter:
    """Formats analysis results for display"""
    
    @staticmethod
    def format_for_display(results: Dict) -> Dict:
        """Format results in a human-readable structure"""
        formatted = {}
        
        # Dimensions
        dims = results["dimensions"]
        formatted_dims = {
            "Length": f"{dims.length:.2f} units",
            "Width": f"{dims.width:.2f} units", 
            "Height": f"{dims.height:.2f} units",
            "Surface Area": f"{dims.surface_area:.2f} square units"
        }
        
        # Add volume only if available
        if dims.volume is not None:
            formatted_dims["Volume"] = f"{dims.volume:.2f} cubic units"
        else:
            formatted_dims["Volume"] = "Not available (mesh not watertight)"
            
        formatted["Dimensions"] = formatted_dims
        
        # Basic properties with detailed watertightness info
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
            "Mesh Complexity": f"{meta['faces']} faces, {meta['vertices']} vertices",
            "Analysis Time": f"{meta['analysis_time']:.2f} seconds",
            "Compute Device": f"{meta['device_type']} ({meta['device_name']})"
        }
        
        return formatted


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='GPU-accelerated 3D mesh analysis using Open3D'
    )
    parser.add_argument('--file', '-i', required=True, help='Path to .obj file')
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
    
    args = parser.parse_args()
    
    try:
        # Validate input file
        if not Path(args.file).exists():
            print(f"Error: File '{args.file}' not found")
            sys.exit(1)
        
        print(f"Analyzing: {args.file}")
        
        # Setup logging level
        if not args.verbose:
            # Suppress Open3D warnings in non-verbose mode
            o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        
        # Run analysis
        use_gpu = not args.cpu_only
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
            
        # Print GPU info if used
        if use_gpu:
            device_info = analyzer.device_info
            print(f"\nProcessing completed using: {device_info['type']} - {device_info['name']}")
            
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()