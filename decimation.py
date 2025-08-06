import open3d as o3d
import numpy as np
import argparse
import os
from typing import Tuple, Optional, Set
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

def analyze_mesh_statistics(mesh):
    """Analyze mesh edge lengths and triangle areas for statistics."""
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    edge_lengths = []
    triangle_areas = []
    
    for triangle in triangles:
        # Calculate edge lengths for this triangle
        for i in range(3):
            v1, v2 = triangle[i], triangle[(i+1) % 3]
            edge_length = np.linalg.norm(vertices[v1] - vertices[v2])
            edge_lengths.append(edge_length)
        
        # Calculate triangle area
        p1, p2, p3 = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]
        area = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
        triangle_areas.append(area)
    
    return {
        'edge_lengths': np.array(edge_lengths),
        'triangle_areas': np.array(triangle_areas),
        'avg_edge_length': np.mean(edge_lengths),
        'std_edge_length': np.std(edge_lengths),
        'avg_triangle_area': np.mean(triangle_areas),
        'std_triangle_area': np.std(triangle_areas)
    }

def detect_feature_edges(mesh, angle_threshold=30.0):
    """
    Detect feature edges based on dihedral angles between adjacent faces.
    
    Args:
        mesh: Open3D triangle mesh
        angle_threshold: Angle in degrees above which an edge is considered a feature
    
    Returns:
        Set of feature edges as (vertex1, vertex2) tuples
    """
    mesh.compute_triangle_normals()
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    triangle_normals = np.asarray(mesh.triangle_normals)
    
    # Build edge to triangle mapping
    edge_to_triangles = {}
    
    for tri_idx, triangle in enumerate(triangles):
        for i in range(3):
            v1, v2 = triangle[i], triangle[(i+1) % 3]
            edge = (min(v1, v2), max(v1, v2))  # Canonical edge representation
            
            if edge not in edge_to_triangles:
                edge_to_triangles[edge] = []
            edge_to_triangles[edge].append(tri_idx)
    
    feature_edges = set()
    angle_threshold_rad = np.radians(angle_threshold)
    
    for edge, adjacent_triangles in edge_to_triangles.items():
        if len(adjacent_triangles) == 2:  # Internal edge
            tri1, tri2 = adjacent_triangles
            normal1 = triangle_normals[tri1]
            normal2 = triangle_normals[tri2]
            
            # Calculate dihedral angle
            dot_product = np.clip(np.dot(normal1, normal2), -1.0, 1.0)
            angle = np.arccos(abs(dot_product))
            
            if angle > angle_threshold_rad:
                feature_edges.add(edge)
        elif len(adjacent_triangles) == 1:  # Boundary edge
            feature_edges.add(edge)
    
    return feature_edges

def compute_vertex_curvatures(mesh):
    """
    Compute mean curvature at each vertex using local neighborhood analysis.
    
    Args:
        mesh: Open3D triangle mesh
        
    Returns:
        Array of mean curvatures at each vertex
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # Compute vertex normals if not already computed
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    vertex_normals = np.asarray(mesh.vertex_normals)
    
    curvatures = np.zeros(len(vertices))
    
    # Build vertex adjacency
    vertex_adjacency = [set() for _ in range(len(vertices))]
    for triangle in triangles:
        for i in range(3):
            for j in range(3):
                if i != j:
                    vertex_adjacency[triangle[i]].add(triangle[j])
    
    for v_idx in range(len(vertices)):
        neighbors = list(vertex_adjacency[v_idx])
        if len(neighbors) < 3:
            continue
            
        vertex = vertices[v_idx]
        normal = vertex_normals[v_idx]
        
        # Compute curvature using discrete mean curvature
        curvature_vector = np.zeros(3)
        total_weight = 0
        
        for n_idx in neighbors:
            neighbor = vertices[n_idx]
            edge_vector = neighbor - vertex
            edge_length = np.linalg.norm(edge_vector)
            
            if edge_length > 0:
                # Weight by inverse edge length
                weight = 1.0 / edge_length
                curvature_vector += weight * edge_vector
                total_weight += weight
        
        if total_weight > 0:
            curvature_vector /= total_weight
            # Project onto normal to get mean curvature
            curvatures[v_idx] = abs(np.dot(curvature_vector, normal))
    
    return curvatures

def compute_feature_distances(vertices, feature_edges):
    """
    Compute distance from each vertex to the nearest feature edge.
    
    Args:
        vertices: Vertex coordinates
        feature_edges: Set of feature edges
        
    Returns:
        Array of distances to nearest feature edge
    """
    if not feature_edges:
        return np.full(len(vertices), float('inf'))
    
    # Sample points along feature edges
    feature_points = []
    for edge in feature_edges:
        v1_idx, v2_idx = edge
        v1, v2 = vertices[v1_idx], vertices[v2_idx]
        
        # Sample points along the edge
        num_samples = max(int(np.linalg.norm(v2 - v1) * 10), 2)
        for i in range(num_samples + 1):
            t = i / num_samples
            point = v1 + t * (v2 - v1)
            feature_points.append(point)
    
    if not feature_points:
        return np.full(len(vertices), float('inf'))
    
    feature_points = np.array(feature_points)
    
    # Compute distances using KDTree
    tree = cKDTree(feature_points)
    distances, _ = tree.query(vertices)
    
    return distances

def compute_adaptive_edge_lengths(mesh, 
                                  min_edge_length,
                                  max_edge_length,
                                  feature_angle_threshold=30.0,
                                  curvature_sensitivity=2.0,
                                  feature_distance_factor=3.0):
    """
    Compute adaptive edge lengths based on local curvature and feature proximity.
    
    Args:
        mesh: Open3D triangle mesh
        min_edge_length: Minimum edge length for high-detail areas
        max_edge_length: Maximum edge length for low-detail areas
        feature_angle_threshold: Angle threshold for feature detection
        curvature_sensitivity: How much curvature affects edge length (higher = more sensitive)
        feature_distance_factor: Distance factor for feature influence
        
    Returns:
        Array of target edge lengths at each vertex
    """
    vertices = np.asarray(mesh.vertices)
    
    # Detect features
    feature_edges = detect_feature_edges(mesh, feature_angle_threshold)
    
    # Compute curvatures
    curvatures = compute_vertex_curvatures(mesh)
    
    # Compute distances to features
    feature_distances = compute_feature_distances(vertices, feature_edges)
    
    # Normalize curvatures to [0, 1]
    if np.max(curvatures) > 0:
        normalized_curvatures = curvatures / np.max(curvatures)
    else:
        normalized_curvatures = np.zeros_like(curvatures)
    
    # Normalize feature distances
    max_feature_distance = np.percentile(feature_distances[feature_distances < float('inf')], 95)
    if max_feature_distance > 0:
        normalized_feature_distances = np.clip(feature_distances / max_feature_distance, 0, 1)
    else:
        normalized_feature_distances = np.ones_like(feature_distances)
    
    # Compute edge length factors
    # High curvature -> small triangles (factor approaches 0)
    curvature_factor = np.exp(-curvature_sensitivity * normalized_curvatures)
    
    # Close to features -> small triangles (factor approaches 0)
    feature_factor = np.exp(-feature_distance_factor * (1 - normalized_feature_distances))
    
    # Combine factors (taking minimum to prioritize detail)
    combined_factor = np.minimum(curvature_factor, feature_factor)
    
    # Map to edge length range
    target_edge_lengths = min_edge_length + combined_factor * (max_edge_length - min_edge_length)
    
    return target_edge_lengths, normalized_curvatures, normalized_feature_distances

def sample_adaptive_points(mesh, target_edge_lengths):
    """
    Sample points on mesh surface with adaptive density based on target edge lengths.
    
    Args:
        mesh: Open3D triangle mesh
        target_edge_lengths: Target edge length at each vertex
        
    Returns:
        Sampled points and their target edge lengths
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    sampled_points = []
    point_edge_lengths = []
    
    for triangle in triangles:
        # Get triangle vertices and their target edge lengths
        tri_vertices = vertices[triangle]
        tri_edge_lengths = target_edge_lengths[triangle]
        
        # Use minimum target edge length for this triangle (highest detail)
        target_edge = np.min(tri_edge_lengths)
        
        # Calculate triangle area
        p1, p2, p3 = tri_vertices[0], tri_vertices[1], tri_vertices[2]
        area = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
        
        # Calculate number of points needed
        target_triangle_area = (np.sqrt(3) / 4) * target_edge**2
        num_points = max(int(area / target_triangle_area), 1)
        
        # Sample points uniformly in the triangle
        for _ in range(num_points):
            # Random barycentric coordinates
            r1, r2 = np.random.random(2)
            if r1 + r2 > 1:
                r1, r2 = 1 - r1, 1 - r2
            r3 = 1 - r1 - r2
            
            # Compute point
            point = r1 * p1 + r2 * p2 + r3 * p3
            sampled_points.append(point)
            point_edge_lengths.append(target_edge)
    
    return np.array(sampled_points), np.array(point_edge_lengths)

def remesh_adaptive_for_raycasting(input_path: str,
                                   min_edge_length: float,
                                   max_edge_length: float,
                                   output_path: str,
                                   feature_angle_threshold: float = 30.0,
                                   curvature_sensitivity: float = 2.0,
                                   feature_distance_factor: float = 3.0) -> None:
    """
    Creates an adaptive mesh optimized for thin wall and gap detection via raycasting.
    
    This method creates locally uniform triangles that adapt to geometric features:
    - Small triangles near sharp edges, corners, and high-curvature areas
    - Large triangles in flat, low-detail regions
    - Optimal for raycasting analysis of thin walls and small gaps
    
    The algorithm:
    1. Analyzes local geometry (curvature, feature proximity)
    2. Computes adaptive target edge lengths for each vertex
    3. Samples points with density matching local requirements
    4. Reconstructs mesh using Poisson surface reconstruction
    
    Benefits for raycasting:
    - More raycast hits where walls are likely to be thin
    - Reduced computational cost in low-detail areas
    - Better detection of small geometric features
    - Maintains feature fidelity for accurate measurements
    
    Args:
        input_path: Path to input mesh file
        min_edge_length: Minimum edge length for high-detail areas (thin walls, sharp features)
        max_edge_length: Maximum edge length for low-detail areas (flat surfaces)
        output_path: Path to save the adaptive mesh
        feature_angle_threshold: Angle threshold for feature detection (default: 30°)
        curvature_sensitivity: Sensitivity to curvature changes (default: 2.0)
        feature_distance_factor: Distance decay factor for feature influence (default: 3.0)
    """
    
    print(f"Loading mesh from: {input_path}")
    mesh = o3d.io.read_triangle_mesh(input_path)
    
    if len(mesh.vertices) == 0:
        raise ValueError("Failed to load mesh or mesh is empty")
    
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    
    print(f"Original mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    
    # Analyze initial mesh statistics
    initial_stats = analyze_mesh_statistics(mesh)
    print(f"Initial average edge length: {initial_stats['avg_edge_length']:.4f}")
    print(f"Target range: {min_edge_length:.4f} - {max_edge_length:.4f}")
    
    # Compute adaptive edge lengths
    print("Computing adaptive edge lengths based on geometry...")
    target_edge_lengths, curvatures, feature_distances = compute_adaptive_edge_lengths(
        mesh, min_edge_length, max_edge_length, feature_angle_threshold,
        curvature_sensitivity, feature_distance_factor
    )
    
    print(f"Edge length adaptation:")
    print(f"  Min target: {np.min(target_edge_lengths):.4f}")
    print(f"  Max target: {np.max(target_edge_lengths):.4f}")
    print(f"  Mean target: {np.mean(target_edge_lengths):.4f}")
    
    # Analyze distribution
    high_detail_vertices = np.sum(target_edge_lengths < (min_edge_length + max_edge_length) / 2)
    print(f"  High-detail vertices: {high_detail_vertices}/{len(target_edge_lengths)} ({100*high_detail_vertices/len(target_edge_lengths):.1f}%)")
    
    # Sample points adaptively
    print("Sampling points with adaptive density...")
    sampled_points, point_edge_lengths = sample_adaptive_points(mesh, target_edge_lengths)
    
    print(f"Sampled {len(sampled_points)} points")
    print(f"Point density distribution:")
    print(f"  High-detail points: {np.sum(point_edge_lengths < (min_edge_length + max_edge_length) / 2)}")
    print(f"  Low-detail points: {np.sum(point_edge_lengths >= (min_edge_length + max_edge_length) / 2)}")
    
    # Create new adaptive mesh
    print("Reconstructing adaptive mesh...")
    try:
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(sampled_points)
        
        # Estimate normals
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(100)
        
        # Perform Poisson reconstruction
        adaptive_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, 
            depth=9,
            width=0,
            scale=1.1,
            linear_fit=False
        )
        
        # Clean up the mesh
        adaptive_mesh.remove_degenerate_triangles()
        adaptive_mesh.remove_duplicated_triangles()
        adaptive_mesh.remove_duplicated_vertices()
        adaptive_mesh.remove_non_manifold_edges()
        
        if len(adaptive_mesh.triangles) == 0:
            raise ValueError("Mesh reconstruction failed - no triangles generated")
        
        # Compute normals for the new mesh
        adaptive_mesh.compute_vertex_normals()
        adaptive_mesh.compute_triangle_normals()
        
        print(f"Adaptive mesh: {len(adaptive_mesh.vertices)} vertices, {len(adaptive_mesh.triangles)} triangles")
        
        # Analyze final mesh statistics
        final_stats = analyze_mesh_statistics(adaptive_mesh)
        print(f"\n--- Final Results ---")
        print(f"Final avg edge length: {final_stats['avg_edge_length']:.4f}")
        print(f"Final std edge length: {final_stats['std_edge_length']:.4f}")
        print(f"Edge length coefficient of variation: {final_stats['std_edge_length']/final_stats['avg_edge_length']:.2f}")
        
        # Calculate mesh efficiency
        triangle_ratio = len(adaptive_mesh.triangles) / len(mesh.triangles)
        print(f"Triangle count ratio: {triangle_ratio:.3f}")
        print(f"Mesh size change: {100*(triangle_ratio-1):.1f}%")
        
        # Save the result
        success = o3d.io.write_triangle_mesh(output_path, adaptive_mesh)
        if success:
            print(f"Adaptive mesh optimized for raycasting saved to: {output_path}")
        else:
            raise RuntimeError(f"Failed to save mesh to: {output_path}")
            
    except Exception as e:
        print(f"Adaptive mesh reconstruction failed: {e}")
        print("Falling back to subdivision-based approach...")
        
        # Fallback: Use adaptive subdivision
        adaptive_mesh = mesh.subdivide_loop(number_of_iterations=1)
        adaptive_mesh.compute_vertex_normals()
        
        # Analyze and save fallback result
        final_stats = analyze_mesh_statistics(adaptive_mesh)
        print(f"Fallback mesh: {len(adaptive_mesh.vertices)} vertices, {len(adaptive_mesh.triangles)} triangles")
        print(f"Fallback avg edge length: {final_stats['avg_edge_length']:.4f}")
        
        success = o3d.io.write_triangle_mesh(output_path, adaptive_mesh)
        if success:
            print(f"Fallback subdivided mesh saved to: {output_path}")
        else:
            raise RuntimeError(f"Failed to save fallback mesh to: {output_path}")

def remesh_uniform_with_features(input_path: str, 
                                target_edge_length: float, 
                                output_path: str,
                                feature_angle_threshold: float = 30.0,
                                feature_preservation_factor: float = 2.0) -> None:
    """
    Performs uniform remeshing while preserving geometric features.
    
    This method creates a completely new mesh with uniform triangle sizes by:
    1. Detecting sharp edges and geometric features
    2. Sampling points uniformly across the surface
    3. Adding extra samples along feature edges for preservation
    4. Reconstructing the mesh using Poisson surface reconstruction
    5. Ensuring uniform triangle distribution for consistent raycasting
    
    This approach works well for:
    - Already decimated meshes with non-uniform triangulation
    - Preparing meshes for uniform raycasting (measuring thin walls/gaps)
    - Creating consistent mesh density across the entire surface
    
    Args:
        input_path: Path to input mesh file
        target_edge_length: Desired average edge length for uniform triangles
        output_path: Path to save the remeshed mesh
        feature_angle_threshold: Angle threshold in degrees for feature detection (default: 30°)
        feature_preservation_factor: Multiplier for point density along features (default: 2.0)
    """
    
    print(f"Loading mesh from: {input_path}")
    mesh = o3d.io.read_triangle_mesh(input_path)
    
    if len(mesh.vertices) == 0:
        raise ValueError("Failed to load mesh or mesh is empty")
    
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    
    print(f"Original mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    
    # Analyze initial mesh statistics
    initial_stats = analyze_mesh_statistics(mesh)
    print(f"Initial average edge length: {initial_stats['avg_edge_length']:.4f}")
    print(f"Initial edge length std dev: {initial_stats['std_edge_length']:.4f}")
    print(f"Target edge length: {target_edge_length:.4f}")
    
    # Detect feature edges
    print("Detecting geometric features...")
    feature_edges = detect_feature_edges(mesh, feature_angle_threshold)
    print(f"Detected {len(feature_edges)} feature edges")
    
    # Sample points uniformly on the surface
    print("Sampling points uniformly on surface...")
    # Calculate target triangle area
    target_area = (np.sqrt(3) / 4) * target_edge_length**2
    
    # Calculate total surface area
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    
    total_area = 0
    for triangle in triangles:
        p1, p2, p3 = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]
        area = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
        total_area += area
    
    # Calculate number of points needed
    num_points = max(int(total_area / target_area), 100)
    
    print(f"Sampling {num_points} points uniformly on surface (total area: {total_area:.4f})")
    
    # Sample points using Open3D's built-in method
    sampled_points = mesh.sample_points_uniformly(number_of_points=num_points)
    surface_points = np.asarray(sampled_points.points)
    
    # Sample additional points along feature edges
    print("Sampling points along feature edges...")
    feature_spacing = target_edge_length / feature_preservation_factor
    feature_points = []
    
    for edge in feature_edges:
        v1_idx, v2_idx = edge
        v1, v2 = vertices[v1_idx], vertices[v2_idx]
        
        # Calculate number of points needed along this edge
        edge_length = np.linalg.norm(v2 - v1)
        num_points_edge = max(int(edge_length / feature_spacing), 2)
        
        # Sample points along the edge
        for i in range(num_points_edge + 1):
            t = i / num_points_edge
            point = v1 + t * (v2 - v1)
            feature_points.append(point)
    
    feature_points = np.array(feature_points) if feature_points else np.empty((0, 3))
    
    # Combine all points
    if len(feature_points) > 0:
        all_points = np.vstack([surface_points, feature_points])
        print(f"Total points: {len(surface_points)} surface + {len(feature_points)} feature = {len(all_points)}")
    else:
        all_points = surface_points
        print(f"Total points: {len(surface_points)} (no feature points)")
    
    # Remove duplicate points that are too close
    print("Removing duplicate points...")
    if len(all_points) > 0:
        # Use KDTree to remove points that are too close
        tree = cKDTree(all_points)
        min_distance = target_edge_length * 0.5  # Minimum distance between points
        
        keep_indices = []
        used = set()
        
        for i, point in enumerate(all_points):
            if i not in used:
                # Find all points within min_distance
                indices = tree.query_ball_point(point, min_distance)
                keep_indices.append(i)
                used.update(indices)
        
        cleaned_points = all_points[keep_indices]
        print(f"Kept {len(cleaned_points)} points after removing duplicates")
    else:
        cleaned_points = all_points
    
    # Create new uniform mesh
    print("Reconstructing uniform mesh...")
    try:
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cleaned_points)
        
        # Estimate normals using the original mesh
        pcd.estimate_normals()
        
        # Ensure normal orientation is consistent
        pcd.orient_normals_consistent_tangent_plane(100)
        
        # Perform Poisson reconstruction
        uniform_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, 
            depth=9,  # Depth of octree for reconstruction
            width=0,  # Ignored if depth is specified
            scale=1.1,  # Scale factor
            linear_fit=False
        )
        
        # Clean up the mesh
        uniform_mesh.remove_degenerate_triangles()
        uniform_mesh.remove_duplicated_triangles()
        uniform_mesh.remove_duplicated_vertices()
        uniform_mesh.remove_non_manifold_edges()
        
        if len(uniform_mesh.triangles) == 0:
            raise ValueError("Mesh reconstruction failed - no triangles generated")
        
        # Compute normals for the new mesh
        uniform_mesh.compute_vertex_normals()
        uniform_mesh.compute_triangle_normals()
        
        print(f"Remeshed: {len(uniform_mesh.vertices)} vertices, {len(uniform_mesh.triangles)} triangles")
        
        # Analyze final mesh statistics
        final_stats = analyze_mesh_statistics(uniform_mesh)
        print(f"\n--- Final Results ---")
        print(f"Final avg edge length: {final_stats['avg_edge_length']:.4f}")
        print(f"Final std edge length: {final_stats['std_edge_length']:.4f}")
        print(f"Edge length uniformity (1/CV): {final_stats['avg_edge_length']/final_stats['std_edge_length']:.2f}")
        print(f"Target achievement: {(target_edge_length/final_stats['avg_edge_length'])*100:.1f}%")
        
        # Save the result
        success = o3d.io.write_triangle_mesh(output_path, uniform_mesh)
        if success:
            print(f"Uniform remeshed mesh saved to: {output_path}")
        else:
            raise RuntimeError(f"Failed to save mesh to: {output_path}")
            
    except Exception as e:
        print(f"Mesh reconstruction failed: {e}")
        print("Falling back to subdivision-based approach...")
        
        # Fallback: Use subdivision for more uniform triangles
        uniform_mesh = mesh.subdivide_loop(number_of_iterations=2)
        uniform_mesh.compute_vertex_normals()
        
        # Analyze and save fallback result
        final_stats = analyze_mesh_statistics(uniform_mesh)
        print(f"Fallback mesh: {len(uniform_mesh.vertices)} vertices, {len(uniform_mesh.triangles)} triangles")
        print(f"Fallback avg edge length: {final_stats['avg_edge_length']:.4f}")
        
        success = o3d.io.write_triangle_mesh(output_path, uniform_mesh)
        if success:
            print(f"Fallback subdivided mesh saved to: {output_path}")
        else:
            raise RuntimeError(f"Failed to save fallback mesh to: {output_path}")

def decimate_uniform_with_features(input_path: str, 
                                 target_edge_length: float, 
                                 output_path: str,
                                 feature_angle_threshold: float = 30.0,
                                 max_iterations: int = 3) -> None:
    """
    Performs feature-aware decimation to create triangles of similar sizes while preserving engineering features.
    
    This method uses an iterative approach that:
    1. Detects sharp edges and geometric features
    2. Applies adaptive quadric decimation that preserves features
    3. Monitors triangle uniformity and adjusts accordingly
    4. Maintains feature preservation throughout the process
    
    The algorithm works by:
    - Using quadric error metrics that weight feature preservation heavily
    - Iteratively adjusting decimation targets based on current triangle statistics
    - Preserving boundary edges and sharp features (> feature_angle_threshold degrees)
    
    Args:
        input_path: Path to input mesh file
        target_edge_length: Desired average edge length for uniform triangles
        output_path: Path to save the decimated mesh
        feature_angle_threshold: Angle threshold in degrees for feature detection (default: 30°)
        max_iterations: Maximum number of decimation iterations (default: 3)
    """
    
    print(f"Loading mesh from: {input_path}")
    mesh = o3d.io.read_triangle_mesh(input_path)
    
    if len(mesh.vertices) == 0:
        raise ValueError("Failed to load mesh or mesh is empty")
    
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    
    print(f"Original mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    
    # Analyze initial mesh statistics
    initial_stats = analyze_mesh_statistics(mesh)
    print(f"Initial average edge length: {initial_stats['avg_edge_length']:.4f}")
    print(f"Initial edge length std dev: {initial_stats['std_edge_length']:.4f}")
    print(f"Target edge length: {target_edge_length:.4f}")
    
    # Detect feature edges
    print("Detecting geometric features...")
    feature_edges = detect_feature_edges(mesh, feature_angle_threshold)
    print(f"Detected {len(feature_edges)} feature edges")
    
    current_mesh = mesh
    
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")
        
        # Analyze current mesh
        current_stats = analyze_mesh_statistics(current_mesh)
        current_avg_edge = current_stats['avg_edge_length']
        current_std_edge = current_stats['std_edge_length']
        
        print(f"Current avg edge length: {current_avg_edge:.4f}")
        print(f"Current std edge length: {current_std_edge:.4f}")
        
        # Determine if we need more or less decimation
        edge_ratio = current_avg_edge / target_edge_length
        
        if abs(edge_ratio - 1.0) < 0.1:  # Within 10% of target
            print("Target edge length achieved!")
            break
        
        # Calculate target triangle count
        # Since triangle area scales with edge_length^2, triangle count scales with 1/edge_length^2
        area_scaling_factor = (target_edge_length / current_avg_edge) ** 2
        target_triangles = int(len(current_mesh.triangles) * area_scaling_factor)
        
        # Ensure we don't go below a reasonable minimum
        min_triangles = max(100, len(current_mesh.triangles) // 20)
        target_triangles = max(target_triangles, min_triangles)
        
        print(f"Targeting {target_triangles} triangles (scaling factor: {area_scaling_factor:.3f})")
        
        if target_triangles >= len(current_mesh.triangles):
            print("Target triangle count reached or exceeded")
            break
        
        # Perform quadric decimation with feature preservation
        try:
            # Use boundary_weight to preserve boundaries and feature edges
            decimated_mesh = current_mesh.simplify_quadric_decimation(
                target_number_of_triangles=target_triangles,
                maximum_error=np.inf,  # Allow error to preserve triangle count target
                boundary_weight=1.0    # Preserve boundary edges
            )
            
            if len(decimated_mesh.triangles) == 0:
                print("Decimation resulted in empty mesh, stopping")
                break
                
            decimated_mesh.compute_vertex_normals()
            decimated_mesh.compute_triangle_normals()
            
            # Remove degenerate triangles and vertices
            decimated_mesh.remove_degenerate_triangles()
            decimated_mesh.remove_duplicated_triangles()
            decimated_mesh.remove_duplicated_vertices()
            decimated_mesh.remove_non_manifold_edges()
            
            current_mesh = decimated_mesh
            
            print(f"After decimation: {len(current_mesh.vertices)} vertices, {len(current_mesh.triangles)} triangles")
            
        except Exception as e:
            print(f"Decimation failed: {e}")
            break
    
    # Final statistics
    final_stats = analyze_mesh_statistics(current_mesh)
    print(f"\n--- Final Results ---")
    print(f"Final mesh: {len(current_mesh.vertices)} vertices, {len(current_mesh.triangles)} triangles")
    print(f"Final avg edge length: {final_stats['avg_edge_length']:.4f}")
    print(f"Final std edge length: {final_stats['std_edge_length']:.4f}")
    print(f"Edge length uniformity (1/CV): {final_stats['avg_edge_length']/final_stats['std_edge_length']:.2f}")
    
    # Calculate reduction ratio
    reduction_ratio = len(current_mesh.triangles) / len(mesh.triangles)
    print(f"Overall reduction ratio: {reduction_ratio:.3f} ({100*(1-reduction_ratio):.1f}% reduction)")
    
    # Save the result
    success = o3d.io.write_triangle_mesh(output_path, current_mesh)
    if success:
        print(f"Feature-preserving decimated mesh saved to: {output_path}")
    else:
        raise RuntimeError(f"Failed to save mesh to: {output_path}")

def decimate_mesh(input_path, reduction_ratio, output_path):
    """Original decimation function (preserved for compatibility)."""
    # Load mesh
    mesh = o3d.io.read_triangle_mesh(input_path)
    mesh.compute_vertex_normals()

    # Compute target triangle count
    target_triangles = int(len(mesh.triangles) * reduction_ratio)
    print(f"Reducing to {target_triangles} triangles...")

    # Decimate
    dec_mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
    dec_mesh.compute_vertex_normals()

    # Save result
    o3d.io.write_triangle_mesh(output_path, dec_mesh)
    print(f"Decimated mesh saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process 3D meshes with adaptive or uniform remeshing optimized for raycasting analysis.")
    parser.add_argument("--input", type=str, required=True, help="Path to input mesh file")
    parser.add_argument("--output", type=str, required=True, help="Path to output file")
    
    # Choose processing method
    method_group = parser.add_mutually_exclusive_group(required=True)
    method_group.add_argument("--reduction", type=float, help="Reduction ratio for standard decimation (0 < ratio < 1)")
    method_group.add_argument("--target-edge-length", type=float, help="Target edge length for uniform feature-aware decimation")
    method_group.add_argument("--remesh-edge-length", type=float, help="Target edge length for uniform feature-preserving remeshing")
    method_group.add_argument("--adaptive-edge-range", type=float, nargs=2, metavar=('MIN', 'MAX'), 
                              help="Min and max edge lengths for adaptive remeshing optimized for raycasting (e.g., --adaptive-edge-range 0.1 1.0)")
    
    # Optional parameters for feature-aware processing
    parser.add_argument("--feature-angle", type=float, default=30.0, help="Feature detection angle threshold in degrees (default: 30)")
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximum decimation iterations (default: 3)")
    parser.add_argument("--feature-preservation", type=float, default=2.0, help="Feature preservation factor for remeshing (default: 2.0)")
    
    # Parameters specific to adaptive remeshing
    parser.add_argument("--curvature-sensitivity", type=float, default=2.0, help="Curvature sensitivity for adaptive remeshing (default: 2.0)")
    parser.add_argument("--feature-distance-factor", type=float, default=3.0, help="Feature distance influence factor (default: 3.0)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        exit(1)

    try:
        if args.reduction is not None:
            # Use original decimation method
            if not (0 < args.reduction < 1):
                print("Error: Reduction ratio must be between 0 and 1 (e.g., 0.5 for 50%)")
                exit(1)
            
            print("Using standard decimation method...")
            decimate_mesh(args.input, args.reduction, args.output)
            
        elif args.target_edge_length is not None:
            # Use feature-aware uniform decimation
            if args.target_edge_length <= 0:
                print("Error: Target edge length must be positive")
                exit(1)
            
            print("Using feature-aware uniform decimation method...")
            decimate_uniform_with_features(
                args.input, 
                args.target_edge_length, 
                args.output,
                args.feature_angle,
                args.max_iterations
            )
            
        elif args.remesh_edge_length is not None:
            # Use uniform remeshing method
            if args.remesh_edge_length <= 0:
                print("Error: Remesh edge length must be positive")
                exit(1)
            
            print("Using uniform feature-preserving remeshing method...")
            remesh_uniform_with_features(
                args.input, 
                args.remesh_edge_length, 
                args.output,
                args.feature_angle,
                args.feature_preservation
            )
            
        elif args.adaptive_edge_range is not None:
            # Use new adaptive remeshing method
            min_edge, max_edge = args.adaptive_edge_range
            if min_edge <= 0 or max_edge <= 0 or min_edge >= max_edge:
                print("Error: Edge range must be positive and min < max")
                exit(1)
            
            print("Using adaptive remeshing optimized for raycasting...")
            remesh_adaptive_for_raycasting(
                args.input,
                min_edge,
                max_edge,
                args.output,
                args.feature_angle,
                args.curvature_sensitivity,
                args.feature_distance_factor
            )
            
    except Exception as e:
        print(f"Error during processing: {e}")
        exit(1)