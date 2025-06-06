import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ezdxf
import os
from collections import defaultdict
from scipy.spatial import ConvexHull

def load_obj(file_path):
    """
    Load an OBJ file and extract vertices and faces
    """
    vertices = []
    faces = []
    
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):  # vertex
                parts = line.strip().split(' ')
                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                vertices.append(vertex)
            elif line.startswith('f '):  # face
                parts = line.strip().split(' ')[1:]
                # OBJ indices start from 1, so we subtract 1
                face = [int(p.split('/')[0]) - 1 for p in parts]
                faces.append(face)
    
    return np.array(vertices), faces

def get_model_dimensions(vertices):
    """
    Get the dimensions of the model and calculate min/max values for each axis
    """
    min_vals = np.min(vertices, axis=0)
    max_vals = np.max(vertices, axis=0)
    dimensions = max_vals - min_vals
    
    return min_vals, max_vals, dimensions

def calculate_cross_section_value(min_val, max_val, section_value, is_percentage):
    """
    Calculate the actual cross-section position based on input
    """
    if is_percentage:
        # Convert percentage to actual value
        if section_value < 0:  # negative percentage means from max
            return max_val + (section_value * (max_val - min_val))
        else:  # positive percentage means from min
            return min_val + (section_value * (max_val - min_val))
    else:
        # Absolute value
        if section_value < 0:  # negative means from max
            return max_val + section_value
        else:  # positive means from min
            return min_val + section_value

def extract_edges(faces):
    """
    Extract unique edges from faces
    """
    # Dictionary to count how many faces each edge belongs to
    edge_face_count = defaultdict(int)
    edge_to_faces = defaultdict(list)
    
    # Extract all edges from faces
    for face_idx, face in enumerate(faces):
        for i in range(len(face)):
            # Get the two vertices that form this edge
            v1, v2 = face[i], face[(i + 1) % len(face)]
            # Store edge as tuple with smaller vertex index first
            edge = tuple(sorted([v1, v2]))
            edge_face_count[edge] += 1
            edge_to_faces[edge].append(face_idx)
    
    # Find edges that appear only once (outline edges)
    outline_edges = [edge for edge, count in edge_face_count.items() if count == 1]
    
    # Also collect edges that might be important feature lines (shared by exactly 2 faces)
    feature_edges = [edge for edge, count in edge_face_count.items() if count == 2]
    
    return outline_edges, feature_edges, edge_to_faces

def project_to_2d(vertices, projection_plane='xy'):
    """
    Project 3D vertices onto a 2D plane
    """
    if projection_plane == 'xy':
        return vertices[:, 0:2]  # x, y coordinates (top view)
    elif projection_plane == 'yz':
        return vertices[:, 1:3]  # y, z coordinates (front view)
    elif projection_plane == 'xz':
        return vertices[:, [0, 2]]  # x, z coordinates (side view)
    else:
        raise ValueError("Projection plane must be 'xy', 'yz', or 'xz'")

def find_visible_edges(vertices_3d, faces, projection_plane):
    """
    Find edges that would be visible in the given projection
    by identifying silhouette edges
    """
    # Extract normal vector for the projection plane
    if projection_plane == 'xy':
        normal = np.array([0, 0, 1])  # z-axis
    elif projection_plane == 'yz':
        normal = np.array([1, 0, 0])  # x-axis
    elif projection_plane == 'xz':
        normal = np.array([0, 1, 0])  # y-axis
    
    # Calculate face normals
    face_normals = []
    face_visibility = []
    
    for face in faces:
        if len(face) >= 3:
            # Get three points from the face
            p0 = vertices_3d[face[0]]
            p1 = vertices_3d[face[1]]
            p2 = vertices_3d[face[2]]
            
            # Calculate face normal using cross product
            v1 = p1 - p0
            v2 = p2 - p0
            face_normal = np.cross(v1, v2)
            
            # Normalize
            norm = np.linalg.norm(face_normal)
            if norm > 0:
                face_normal = face_normal / norm
            
            # Determine if face is visible (dot product with viewing direction)
            dot_product = np.dot(face_normal, normal)
            is_visible = dot_product < 0  # Face is visible if it faces the camera
            
            face_normals.append(face_normal)
            face_visibility.append(is_visible)
    
    # Extract outline edges and feature edges
    outline_edges, feature_edges, edge_to_faces = extract_edges(faces)
    
    # For each feature edge, determine if it's a silhouette edge
    # (edge between a visible and a non-visible face)
    silhouette_edges = []
    
    # Find silhouette edges
    for edge, face_indices in edge_to_faces.items():
        if len(face_indices) == 2:
            face1, face2 = face_indices
            if face1 < len(face_visibility) and face2 < len(face_visibility):
                if face_visibility[face1] != face_visibility[face2]:
                    silhouette_edges.append(edge)
    
    # Combine outline edges and silhouette edges for visibility
    visible_edges = outline_edges + silhouette_edges
    
    return visible_edges

def generate_cross_section_edges(vertices_3d, faces, axis, section_value, is_percentage=True, tolerance=0.001):
    """
    Generate cross-section edges at the specified position along an axis
    
    Parameters:
    - vertices_3d: 3D vertices of the model
    - faces: List of faces from the model
    - axis: Axis for cross-section ('x', 'y', or 'z')
    - section_value: Position along the axis (percentage or absolute)
    - is_percentage: Whether section_value is a percentage (True) or absolute (False)
    - tolerance: Tolerance for considering a point on the section plane
    
    Returns:
    - List of edge segments (pairs of 2D points) representing the cross-section
    """
    # Get model dimensions
    min_vals, max_vals, dimensions = get_model_dimensions(vertices_3d)
    
    # Determine the axis index
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis.lower()]
    
    # Calculate the actual section position
    section_pos = calculate_cross_section_value(
        min_vals[axis_idx], max_vals[axis_idx], section_value, is_percentage
    )
    
    # Store the intersection segments
    intersection_segments = []
    
    # For each face
    for face in faces:
        # Skip faces with less than 3 vertices
        if len(face) < 3:
            continue
            
        # Get all vertices of this face
        face_vertices = [vertices_3d[i] for i in face]
        
        # Check if the face intersects with the section plane
        min_val = min(v[axis_idx] for v in face_vertices)
        max_val = max(v[axis_idx] for v in face_vertices)
        
        # Skip if the face doesn't cross the section plane
        if section_pos < min_val - tolerance or section_pos > max_val + tolerance:
            continue
        
        # Find all edge intersections for this face
        face_intersections = []
        
        for i in range(len(face)):
            v1_idx = face[i]
            v2_idx = face[(i + 1) % len(face)]
            
            p1 = vertices_3d[v1_idx]
            p2 = vertices_3d[v2_idx]
            
            # Check if the edge crosses the section plane
            if ((p1[axis_idx] < section_pos - tolerance and p2[axis_idx] > section_pos + tolerance) or
                (p2[axis_idx] < section_pos - tolerance and p1[axis_idx] > section_pos + tolerance)):
                
                # Calculate the intersection point using linear interpolation
                t = (section_pos - p1[axis_idx]) / (p2[axis_idx] - p1[axis_idx])
                
                # Calculate the 3D intersection point
                intersection = p1 + t * (p2 - p1)
                
                # Determine the 2D projection based on section axis
                if axis.lower() == 'x':
                    intersection_2d = [intersection[1], intersection[2]]  # yz plane
                elif axis.lower() == 'y':
                    intersection_2d = [intersection[0], intersection[2]]  # xz plane
                else:  # z
                    intersection_2d = [intersection[0], intersection[1]]  # xy plane
                
                face_intersections.append(intersection_2d)
        
        # If we found exactly two intersections for this face,
        # it means the face forms a line segment in the cross-section
        if len(face_intersections) == 2:
            intersection_segments.append((face_intersections[0], face_intersections[1]))
        
        # For faces with more than 2 intersections (rare, but possible),
        # we need to determine which points to connect
        elif len(face_intersections) > 2:
            # This is a simplification - ideally we'd have more logic here
            # to handle complex intersections correctly
            
            # Sort points based on their position
            # (this is a simple approach, might need refinement)
            points = np.array(face_intersections)
            
            # Try to find the convex hull of the intersection points
            try:
                hull = ConvexHull(points)
                hull_vertices = hull.vertices
                
                # Connect the hull vertices to form line segments
                for i in range(len(hull_vertices)):
                    idx1 = hull_vertices[i]
                    idx2 = hull_vertices[(i + 1) % len(hull_vertices)]
                    intersection_segments.append((points[idx1], points[idx2]))
            except:
                # Fallback if convex hull fails (e.g., collinear points)
                # Just connect consecutive points
                for i in range(len(points) - 1):
                    intersection_segments.append((points[i], points[i + 1]))
                
    return intersection_segments

def create_dxf_clean(vertices_2d, vertices_3d, faces, output_path, projection_plane):
    """
    Create a DXF file with clean edges (outline/silhouette)
    """
    # Find visible edges for this projection
    visible_edges = find_visible_edges(vertices_3d, faces, projection_plane)
    
    # Create DXF document
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    # Add visible edges to the drawing
    for edge in visible_edges:
        v1_idx, v2_idx = edge
        p1 = vertices_2d[v1_idx]
        p2 = vertices_2d[v2_idx]
        
        # Add line
        msp.add_line(p1, p2)
    
    # Save the drawing
    doc.saveas(output_path)
    
    return output_path, visible_edges

def create_dxf_cross_section(cross_section_segments, output_path):
    """
    Create a DXF file with the cross-section edges
    """
    # Create DXF document
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    # Add each segment as a line
    for segment in cross_section_segments:
        p1, p2 = segment
        msp.add_line(p1, p2)
    
    # Save the drawing
    doc.saveas(output_path)
    
    return output_path

def visualize_clean_projection(vertices_2d, visible_edges, projection_plane, output_path=None):
    """
    Visualize the clean 2D projection using matplotlib
    """
    plt.figure(figsize=(10, 10))
    
    # Plot each visible edge
    for edge in visible_edges:
        v1_idx, v2_idx = edge
        p1 = vertices_2d[v1_idx]
        p2 = vertices_2d[v2_idx]
        
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=2)
    
    # Set axis equal and labels
    plt.axis('equal')
    plt.title(f'Clean 2D Projection ({projection_plane} plane)')
    
    if projection_plane == 'xy':
        plt.xlabel('X')
        plt.ylabel('Y')
    elif projection_plane == 'yz':
        plt.xlabel('Y')
        plt.ylabel('Z')
    elif projection_plane == 'xz':
        plt.xlabel('X')
        plt.ylabel('Z')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def visualize_cross_section_segments(cross_section_segments, axis, section_value, output_path=None):
    """
    Visualize the cross-section segments using matplotlib
    """
    plt.figure(figsize=(10, 10))
    
    # Plot each segment
    for segment in cross_section_segments:
        p1, p2 = segment
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=2)
    
    # Set axis equal and labels
    plt.axis('equal')
    
    # Determine labels based on the section axis
    if axis.lower() == 'x':
        plt.xlabel('Y')
        plt.ylabel('Z')
        section_type = "X"
    elif axis.lower() == 'y':
        plt.xlabel('X')
        plt.ylabel('Z')
        section_type = "Y"
    else:  # z
        plt.xlabel('X')
        plt.ylabel('Y')
        section_type = "Z"
    
    plt.title(f'Cross-section at {section_type}={section_value:.3f}')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def obj_to_clean_cad(obj_file, output_dir, generate_views=None, cross_sections=None):
    """
    Convert OBJ to clean CAD drawings with optional cross-sections
    
    Parameters:
    - obj_file: Path to input OBJ file
    - output_dir: Directory to save output files
    - generate_views: List of views to generate (default: ['xy', 'yz', 'xz'])
    - cross_sections: List of cross-sections to generate, each as a tuple (axis, value, is_percentage)
      e.g., [('z', 0.5, True), ('x', 10, False)]
    """
    if generate_views is None:
        generate_views = ['xy', 'yz', 'xz']
    
    if cross_sections is None:
        cross_sections = []
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load OBJ file
    print(f"Loading OBJ file: {obj_file}")
    vertices_3d, faces = load_obj(obj_file)
    
    # Get model dimensions
    min_vals, max_vals, dimensions = get_model_dimensions(vertices_3d)
    print(f"Model dimensions: {dimensions}")
    print(f"X range: {min_vals[0]:.3f} to {max_vals[0]:.3f}")
    print(f"Y range: {min_vals[1]:.3f} to {max_vals[1]:.3f}")
    print(f"Z range: {min_vals[2]:.3f} to {max_vals[2]:.3f}")
    
    # Process each projection plane
    for plane in generate_views:
        print(f"Generating clean {plane} projection...")
        
        # Project vertices to 2D
        vertices_2d = project_to_2d(vertices_3d, plane)
        
        # Create DXF file with clean edges
        dxf_path = os.path.join(output_dir, f"{os.path.basename(obj_file).split('.')[0]}_{plane}.dxf")
        _, visible_edges = create_dxf_clean(vertices_2d, vertices_3d, faces, dxf_path, plane)
        print(f"Clean DXF file saved: {dxf_path}")
        
        # Visualize clean projection
        img_path = os.path.join(output_dir, f"{os.path.basename(obj_file).split('.')[0]}_{plane}.png")
        visualize_clean_projection(vertices_2d, visible_edges, plane, img_path)
        print(f"Clean preview image saved: {img_path}")
    
    # Process each cross-section
    for i, (axis, value, is_percentage) in enumerate(cross_sections):
        print(f"Generating cross-section {i+1} along {axis}-axis at {'{}%'.format(value*100) if is_percentage else value}...")
        
        # Calculate actual position for display purposes
        actual_value = value
        if is_percentage:
            actual_value = value * 100  # Convert to percentage for display
        
        # Generate cross-section segments
        cross_section_segments = generate_cross_section_edges(
            vertices_3d, faces, axis, value, is_percentage
        )
        
        if cross_section_segments:
            # Format the value for filename
            value_str = f"{value*100:.0f}pct" if is_percentage else f"{value:.1f}units"
            
            # Create DXF file with cross-section
            dxf_path = os.path.join(
                output_dir, 
                f"{os.path.basename(obj_file).split('.')[0]}_section_{axis}_{value_str}.dxf"
            )
            create_dxf_cross_section(cross_section_segments, dxf_path)
            print(f"Cross-section DXF file saved: {dxf_path}")
            
            # Visualize cross-section
            img_path = os.path.join(
                output_dir,
                f"{os.path.basename(obj_file).split('.')[0]}_section_{axis}_{value_str}.png"
            )
            visualize_cross_section_segments(cross_section_segments, axis, actual_value, img_path)
            print(f"Cross-section preview image saved: {img_path}")
        else:
            print(f"No intersection found for cross-section {i+1}.")
    
    print("Conversion completed!")

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert 3D OBJ model to clean 2D CAD drawings with cross-sections')
    parser.add_argument('obj_file', help='Path to input OBJ file')
    parser.add_argument('--output-dir', default='output', help='Directory to save output files')
    parser.add_argument('--views', nargs='+', default=['xy', 'yz', 'xz'], 
                        choices=['xy', 'yz', 'xz'], help='Projection views to generate')
    
    # Cross-section arguments
    section_group = parser.add_argument_group('Cross-section options')
    section_group.add_argument('--section-x', nargs='+', type=float, action='append', default=[],
                              help='X-axis cross-section positions (can specify multiple)')
    section_group.add_argument('--section-y', nargs='+', type=float, action='append', default=[],
                              help='Y-axis cross-section positions (can specify multiple)')
    section_group.add_argument('--section-z', nargs='+', type=float, action='append', default=[],
                              help='Z-axis cross-section positions (can specify multiple)')
    section_group.add_argument('--percentage', action='store_true',
                              help='Interpret section values as percentages (0.0-1.0) instead of absolute units')
    
    args = parser.parse_args()
    
    # Build cross-sections list
    cross_sections = []
    
    # Process all section arguments
    for axis, sections in [('x', args.section_x), ('y', args.section_y), ('z', args.section_z)]:
        for section_values in sections:
            for value in section_values:
                cross_sections.append((axis, value, args.percentage))
    
    obj_to_clean_cad(args.obj_file, args.output_dir, args.views, cross_sections)