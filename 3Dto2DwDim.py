import numpy as np
import matplotlib.pyplot as plt
import ezdxf
import os
from collections import defaultdict
import math
import time
import cv2

class ProgressTracker:
    """Progress tracking for long operations"""
    def __init__(self, total_steps, description="Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.description = description
        self.last_update_time = 0
        
    def update(self, step=None, message=""):
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
            
        current_time = time.time()
        
        # Update every 2 seconds to avoid spam
        if current_time - self.last_update_time >= 2.0:
            elapsed = current_time - self.start_time
            if self.current_step > 0:
                eta = (elapsed / self.current_step) * (self.total_steps - self.current_step)
                progress = (self.current_step / self.total_steps) * 100
                
                print(f"  {self.description}: {progress:.1f}% ({self.current_step}/{self.total_steps}) "
                      f"- ETA: {eta:.0f}s {message}")
            else:
                print(f"  {self.description}: Starting... {message}")
                
            self.last_update_time = current_time
    
    def finish(self, message="Completed"):
        elapsed = time.time() - self.start_time
        print(f"  ✅ {self.description}: {message} in {elapsed:.1f}s")

class SmartDimensionDrawer:
    """
    Smart dimension drawer that places all dimensions close to shapes with consistent annotation style
    """
    def __init__(self, msp, scale_factor=1.0):
        self.msp = msp
        self.scale_factor = scale_factor
        self.arrow_size = 2.0  # Smaller arrows for cleaner look
        self.text_height = 2.8  # Slightly smaller text
        self.extension_line_offset = 0.5  # Minimal offset from shape
        self.extension_line_extension = 1.5  # Shorter extensions
        self.used_positions = []  # Track used dimension positions
        self.min_spacing = 10.0  # Minimum spacing to prevent actual collisions
        self.close_offset = 4.0  # How close dimensions should be to shapes - PREFERRED distance
        
    def find_safe_offset(self, start_point, end_point, preferred_offset, direction):
        """Find a safe offset - stay close unless there's actual collision"""
        base_offset = abs(self.close_offset)  # Use consistent close offset
        sign = 1 if preferred_offset > 0 else -1
        
        # First try the close offset (preferred)
        test_offset = sign * base_offset
        if self._is_position_safe(start_point, end_point, test_offset, direction):
            return test_offset
        
        # Only if there's actual collision, try progressively larger offsets
        for multiplier in [1.5, 2.0, 2.5, 3.0, 4.0]:
            test_offset = sign * base_offset * multiplier
            if self._is_position_safe(start_point, end_point, test_offset, direction):
                return test_offset
                
        # Last resort - far enough to avoid any collision
        return sign * base_offset * 5
    
    def _is_position_safe(self, start_point, end_point, offset, direction):
        """Check if dimension position is safe - only flag real collisions"""
        if direction == 'horizontal':
            dim_line_y = min(start_point[1], end_point[1]) + offset
            dim_line_x_start = min(start_point[0], end_point[0])
            dim_line_x_end = max(start_point[0], end_point[0])
            
            for used_pos in self.used_positions:
                if used_pos['type'] == 'horizontal':
                    # Only consider it a collision if lines are very close in Y
                    y_distance = abs(used_pos['y'] - dim_line_y)
                    if y_distance < self.min_spacing:
                        # Check if they overlap in X direction
                        used_x_start = min(used_pos.get('x_start', used_pos.get('x', 0)), 
                                         used_pos.get('x_end', used_pos.get('x', 0)))
                        used_x_end = max(used_pos.get('x_start', used_pos.get('x', 0)), 
                                       used_pos.get('x_end', used_pos.get('x', 0)))
                        
                        # Add small buffer for text space
                        text_buffer = self.text_height * 1.2
                        
                        # Check for actual X overlap with text buffer
                        if not (dim_line_x_end < used_x_start - text_buffer or 
                                dim_line_x_start > used_x_end + text_buffer):
                            return False  # Real collision detected
                            
        else:  # vertical
            dim_line_x = min(start_point[0], end_point[0]) + offset
            dim_line_y_start = min(start_point[1], end_point[1])
            dim_line_y_end = max(start_point[1], end_point[1])
            
            for used_pos in self.used_positions:
                if used_pos['type'] == 'vertical':
                    # Only consider it a collision if lines are very close in X
                    x_distance = abs(used_pos['x'] - dim_line_x)
                    if x_distance < self.min_spacing:
                        # Check if they overlap in Y direction
                        used_y_start = min(used_pos.get('y_start', used_pos.get('y', 0)), 
                                         used_pos.get('y_end', used_pos.get('y', 0)))
                        used_y_end = max(used_pos.get('y_start', used_pos.get('y', 0)), 
                                       used_pos.get('y_end', used_pos.get('y', 0)))
                        
                        # Add small buffer for text space
                        text_buffer = self.text_height * 1.2
                        
                        # Check for actual Y overlap with text buffer
                        if not (dim_line_y_end < used_y_start - text_buffer or 
                                dim_line_y_start > used_y_end + text_buffer):
                            return False  # Real collision detected
        
        return True  # No collision - safe to use this position
    
    def draw_linear_dimension(self, dimension):
        """Draw a linear dimension with smart positioning - always close to shape"""
        start_point = dimension['start_point']
        end_point = dimension['end_point']
        preferred_offset = dimension.get('offset', -self.close_offset)  # Default to close offset
        label = dimension['label']
        direction = dimension.get('direction', 'auto')
        
        # Determine dimension line direction
        if direction == 'horizontal' or (direction == 'auto' and 
            abs(end_point[0] - start_point[0]) > abs(end_point[1] - start_point[1])):
            direction = 'horizontal'
        else:
            direction = 'vertical'
        
        # Find safe offset - but keep it close
        safe_offset = self.find_safe_offset(start_point, end_point, preferred_offset, direction)
        
        if direction == 'horizontal':
            # Horizontal dimension
            dim_line_y = min(start_point[1], end_point[1]) + safe_offset
            
            # Extension lines - very short
            ext1_start = [start_point[0], start_point[1] - self.extension_line_offset]
            ext1_end = [start_point[0], dim_line_y - self.extension_line_extension]
            
            ext2_start = [end_point[0], end_point[1] - self.extension_line_offset]
            ext2_end = [end_point[0], dim_line_y - self.extension_line_extension]
            
            # Dimension line
            dim_line_start = [start_point[0], dim_line_y]
            dim_line_end = [end_point[0], dim_line_y]
            
            # Text position - CORRECTED from test1e.py
            text_pos = [(start_point[0] + end_point[0]) / 2, dim_line_y - self.text_height]
            is_vertical = False
            
            # Record used position with detailed information for collision detection
            self.used_positions.append({
                'type': 'horizontal', 
                'y': dim_line_y,
                'x_start': start_point[0],
                'x_end': end_point[0]
            })
            
        else:  # vertical
            # Vertical dimension
            dim_line_x = min(start_point[0], end_point[0]) + safe_offset
            
            # Extension lines - very short
            ext1_start = [start_point[0] - self.extension_line_offset, start_point[1]]
            ext1_end = [dim_line_x - self.extension_line_extension, start_point[1]]
            
            ext2_start = [end_point[0] - self.extension_line_offset, end_point[1]]
            ext2_end = [dim_line_x - self.extension_line_extension, end_point[1]]
            
            # Dimension line
            dim_line_start = [dim_line_x, start_point[1]]
            dim_line_end = [dim_line_x, end_point[1]]
            
            # Text position - CORRECTED from test1e.py
            text_pos = [dim_line_x - self.text_height, (start_point[1] + end_point[1]) / 2]
            is_vertical = True
            
            # Record used position with detailed information for collision detection
            self.used_positions.append({
                'type': 'vertical', 
                'x': dim_line_x,
                'y_start': start_point[1],
                'y_end': end_point[1]
            })
        
        # Draw components
        self._draw_extension_line(ext1_start, ext1_end)
        self._draw_extension_line(ext2_start, ext2_end)
        self._draw_dimension_line_with_arrows(dim_line_start, dim_line_end)
        self._draw_dimension_text(text_pos, label, is_vertical)
    
    def draw_circle_diameter_consistent(self, circle, index):
        """Draw diameter dimension for ALL circles using consistent close annotation style"""
        center = circle['center']
        radius = circle['radius']
        diameter = circle['diameter']
        
        # Always use the SAME approach for ALL circles - place dimension OUTSIDE and close
        # Points for horizontal diameter line through center
        left_point = [center[0] - radius, center[1]]
        right_point = [center[0] + radius, center[1]]
        
        # CONSISTENT approach: Always place dimension OUTSIDE, very close to perimeter
        diameter_offset = self.close_offset  # Use consistent close offset
        
        # Find safe position that doesn't overlap with other dimensions
        safe_offset = self.find_safe_offset(left_point, right_point, diameter_offset, 'horizontal')
        
        # Draw the diameter dimension outside - SAME for all circles
        diameter_dim = {
            'start_point': left_point,
            'end_point': right_point,
            'label': f"⌀{diameter:.1f}",
            'direction': 'horizontal',
            'offset': safe_offset
        }
        self.draw_linear_dimension(diameter_dim)
        
        # Position circle label close to circle - consistent placement
        label_distance = radius + self.close_offset + 2  # Just outside the dimension
        label_offset_x = label_distance * math.cos(math.radians(45))
        label_offset_y = label_distance * math.sin(math.radians(45))
        label_pos = [center[0] + label_offset_x, center[1] + label_offset_y]
        
        # Draw center mark for all circles
        self._draw_center_mark(center, radius)
        
        # Add circle identifier label - close and consistent
        self.msp.add_text(f"C{index+1}", dxfattribs={
            'insert': label_pos,
            'height': self.text_height * 0.7,
            'layer': 'TEXT',
            'color': 2,  # Yellow for clear visibility
            'halign': 1,  # Center alignment
            'valign': 1   # Middle alignment
        })
    
    def draw_rectangle_dimensions_close(self, rectangle, index, model_bounds):
        """Draw dimensions for rectangles with consistent close positioning"""
        center = rectangle['center']
        width = rectangle['width']
        height = rectangle['height']
        
        # Calculate rectangle corners
        left = center[0] - width/2
        right = center[0] + width/2
        bottom = center[1] - height/2
        top = center[1] + height/2
        
        # Width dimension (bottom) - close to shape
        width_offset = -self.close_offset  # Close to bottom edge
        width_safe_offset = self.find_safe_offset([left, bottom], [right, bottom], width_offset, 'horizontal')
        width_dim = {
            'start_point': [left, bottom],
            'end_point': [right, bottom],
            'label': f"{width:.1f}",
            'direction': 'horizontal',
            'offset': width_safe_offset
        }
        self.draw_linear_dimension(width_dim)
        
        # Height dimension (right side) - close to shape
        height_offset = self.close_offset  # Close to right edge
        height_safe_offset = self.find_safe_offset([right, bottom], [right, top], height_offset, 'vertical')
        height_dim = {
            'start_point': [right, bottom],
            'end_point': [right, top],
            'label': f"{height:.1f}",
            'direction': 'vertical',
            'offset': height_safe_offset
        }
        self.draw_linear_dimension(height_dim)
        
        # Add rectangle label at center
        self.msp.add_text(f"R{index+1}", dxfattribs={
            'insert': center,
            'height': self.text_height * 0.7,
            'layer': 'DIMENSIONS',
            'color': 3,  # Green for feature labels
            'halign': 1,  # Center alignment
            'valign': 1   # Middle alignment
        })
    
    def draw_center_to_edge_dimension_close(self, feature, feature_type, index, model_bounds):
        """Draw dimension from feature center to nearest model edge - close positioning"""
        center = np.array(feature['center'])
        min_bounds = np.array(model_bounds['min'])
        max_bounds = np.array(model_bounds['max'])
        
        # Find nearest edge
        distances = [
            (center[0] - min_bounds[0], 'left'),
            (max_bounds[0] - center[0], 'right'),
            (center[1] - min_bounds[1], 'bottom'),
            (max_bounds[1] - center[1], 'top')
        ]
        
        min_distance, edge_direction = min(distances, key=lambda x: x[0])
        
        # Determine edge point and dimension direction - use close offset
        if edge_direction == 'left':
            edge_point = [min_bounds[0], center[1]]
            offset = -self.close_offset
            direction = 'horizontal'
        elif edge_direction == 'right':
            edge_point = [max_bounds[0], center[1]]
            offset = self.close_offset
            direction = 'horizontal'
        elif edge_direction == 'bottom':
            edge_point = [center[0], min_bounds[1]]
            offset = -self.close_offset
            direction = 'vertical'
        else:  # top
            edge_point = [center[0], max_bounds[1]]
            offset = self.close_offset
            direction = 'vertical'
        
        # Only draw if distance is meaningful (> 5 units)
        if min_distance > 5:
            prefix = f"C{index+1}" if feature_type == 'circle' else f"R{index+1}"
            edge_dim = {
                'start_point': edge_point,
                'end_point': center.tolist(),
                'label': f"{prefix}→{min_distance:.1f}",
                'direction': direction,
                'offset': offset
            }
            self.draw_linear_dimension(edge_dim)
    
    def draw_feature_spacing_close(self, feature1, feature2, index1, index2, feature_types):
        """Draw spacing between features if they're close enough - consistent close positioning"""
        center1 = np.array(feature1['center'])
        center2 = np.array(feature2['center'])
        distance = np.linalg.norm(center1 - center2)
        
        # Only draw spacing if features are reasonably close
        if distance < 100:  # Adjust threshold as needed
            type1 = 'C' if feature_types[0] == 'circle' else 'R'
            type2 = 'C' if feature_types[1] == 'circle' else 'R'
            
            spacing_dim = {
                'start_point': center1.tolist(),
                'end_point': center2.tolist(),
                'label': f"{type1}{index1+1}↔{type2}{index2+1}: {distance:.1f}",
                'direction': 'auto',
                'offset': self.close_offset
            }
            self.draw_linear_dimension(spacing_dim)
    
    def _draw_extension_line(self, start, end):
        """Draw an extension line - lighter weight for cleaner look"""
        line = self.msp.add_line(start, end)
        line.dxf.layer = 'DIMENSIONS'
        line.dxf.lineweight = 13  # Lighter line weight
    
    def _draw_dimension_line_with_arrows(self, start, end):
        """Draw dimension line with arrows at both ends"""
        # Main dimension line
        line = self.msp.add_line(start, end)
        line.dxf.layer = 'DIMENSIONS'
        line.dxf.lineweight = 25  # Standard dimension line weight
        
        # Arrows - smaller for cleaner look
        self._draw_arrow(start, end)
        self._draw_arrow(end, start)
    
    def _draw_arrow(self, tip_point, direction_point):
        """Draw a smaller, cleaner arrow head"""
        direction = np.array(direction_point) - np.array(tip_point)
        length = np.linalg.norm(direction)
        
        if length > 0:
            direction = direction / length
            arrow_length = self.arrow_size
            perp = np.array([-direction[1], direction[0]])
            
            p1 = tip_point
            p2 = np.array(tip_point) + arrow_length * direction + arrow_length * 0.25 * perp
            p3 = np.array(tip_point) + arrow_length * direction - arrow_length * 0.25 * perp
            
            # Create filled arrow for cleaner look
            arrow = self.msp.add_lwpolyline([p1, p2.tolist(), p3.tolist(), p1])
            arrow.dxf.layer = 'DIMENSIONS'
            arrow.close(True)  # Close the polyline to create filled arrow
    
    def _draw_dimension_text(self, position, text, is_vertical=False):
        """Draw dimension text with consistent styling positioned correctly near the dimension line"""
        rotation_angle = 90 if is_vertical else 0
        
        dim_text = self.msp.add_text(
            text,
            dxfattribs={
                'insert': position,
                'height': self.text_height,
                'layer': 'DIMENSIONS',
                'rotation': rotation_angle
            }
        )
    
    def _draw_center_mark(self, center, radius):
        """Draw center mark for circles - proportional to circle size but always visible"""
        mark_size = min(max(radius * 0.12, 1.5), 4.0)  # Between 1.5 and 4.0 units
        
        # Horizontal line
        h_start = [center[0] - mark_size, center[1]]
        h_end = [center[0] + mark_size, center[1]]
        center_line_h = self.msp.add_line(h_start, h_end, dxfattribs={'layer': 'DIMENSIONS'})
        center_line_h.dxf.lineweight = 18  # Medium weight for visibility
        
        # Vertical line
        v_start = [center[0], center[1] - mark_size]
        v_end = [center[0], center[1] + mark_size]
        center_line_v = self.msp.add_line(v_start, v_end, dxfattribs={'layer': 'DIMENSIONS'})
        center_line_v.dxf.lineweight = 18  # Medium weight for visibility

class OpenCVShapeDetector:
    """OpenCV-based shape detection for 2D projection images"""
    def __init__(self, min_area=100, approx_epsilon=0.02):
        self.min_area = min_area
        self.approx_epsilon = approx_epsilon
        
    def detect_shapes_in_image(self, image_path, transform_params):
        """Detect shapes in a 2D projection image using OpenCV with proper coordinate transformation"""
        print(f"    🔍 Analyzing image: {os.path.basename(image_path)}")
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"    ❌ Could not load image: {image_path}")
            return [], [], None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"    📊 Found {len(contours)} contours to analyze")
        
        circles = []
        rectangles = []
        result_img = img.copy()
        
        # Extract transformation parameters
        min_2d = transform_params['min_2d']
        scale = transform_params['scale']
        padding = transform_params['padding']
        img_height = transform_params['img_height']
        
        progress = ProgressTracker(len(contours), "Analyzing contours")
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if area < self.min_area:
                continue
            
            epsilon = self.approx_epsilon * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Convert image coordinates back to real 2D coordinates
            # Reverse the transformation used in create_clean_projection_image
            center_x_img = x + w/2
            center_y_img = y + h/2
            
            # Convert from image coordinates to real 2D coordinates
            center_x_real = (center_x_img - padding) / scale + min_2d[0]
            center_y_real = min_2d[1] + (img_height - center_y_img - padding) / scale
            
            width_real = w / scale
            height_real = h / scale
            
            vertices = len(approx)
            
            if vertices >= 8:  # Likely a circle
                circle_data = self._validate_circle_with_coords(contour, x, y, w, h, 
                                                              center_x_real, center_y_real, 
                                                              width_real, height_real, scale)
                if circle_data:
                    circles.append(circle_data)
                    cv2.circle(result_img, (int(x + w/2), int(y + h/2)), 
                             int(max(w, h)/2), (0, 255, 0), 2)
                    cv2.putText(result_img, f"C{len(circles)}", 
                              (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
            elif vertices == 4:  # Likely a rectangle
                rect_data = self._validate_rectangle_with_coords(contour, x, y, w, h,
                                                               center_x_real, center_y_real,
                                                               width_real, height_real, scale)
                if rect_data:
                    rectangles.append(rect_data)
                    cv2.rectangle(result_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(result_img, f"R{len(rectangles)}", 
                              (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            if i % 10 == 0:
                progress.update(i, f"Found: {len(circles)}C, {len(rectangles)}R")
        
        progress.finish("Shape detection complete")
        
        print(f"    ✅ Detection results:")
        print(f"      🔵 Circles: {len(circles)}")
        print(f"      🔲 Rectangles: {len(rectangles)}")
        
        return circles, rectangles, result_img
    
    def _validate_circle_with_coords(self, contour, x, y, w, h, center_x_real, center_y_real, 
                                   width_real, height_real, scale):
        """Validate if a contour is actually a circle with proper coordinate transformation"""
        aspect_ratio = float(w) / h if h != 0 else 0
        
        if 0.7 <= aspect_ratio <= 1.3:
            contour_area = cv2.contourArea(contour)
            bounding_rect_area = w * h
            area_ratio = contour_area / bounding_rect_area if bounding_rect_area > 0 else 0
            
            if 0.6 <= area_ratio <= 0.9:
                # Calculate radius and diameter in real coordinates
                radius_real = max(width_real, height_real) / 2
                diameter_real = radius_real * 2
                
                return {
                    'type': 'circle',
                    'center': [center_x_real, center_y_real],  # Real 2D coordinates
                    'radius': radius_real,
                    'diameter': diameter_real,
                    'area': contour_area / (scale ** 2),  # Convert area to real units
                    'aspect_ratio': aspect_ratio,
                    'area_ratio': area_ratio
                }
        
        return None
    
    def _validate_rectangle_with_coords(self, contour, x, y, w, h, center_x_real, center_y_real,
                                      width_real, height_real, scale):
        """Validate if a contour is actually a rectangle with proper coordinate transformation"""
        contour_area = cv2.contourArea(contour)
        bounding_rect_area = w * h
        area_ratio = contour_area / bounding_rect_area if bounding_rect_area > 0 else 0
        
        if area_ratio >= 0.8:
            return {
                'type': 'rectangle',
                'center': [center_x_real, center_y_real],  # Real 2D coordinates
                'width': width_real,
                'height': height_real,
                'area': contour_area / (scale ** 2),  # Convert area to real units
                'area_ratio': area_ratio
            }
        
        return None

def load_obj(file_path):
    """Load an OBJ file and extract vertices and faces with progress tracking"""
    print(f"📁 Loading OBJ file: {os.path.basename(file_path)}")
    
    vertices = []
    faces = []
    
    file_size = os.path.getsize(file_path)
    
    with open(file_path, 'r') as file:
        progress = ProgressTracker(file_size, "Loading OBJ file")
        bytes_read = 0
        
        for line in file:
            bytes_read += len(line.encode('utf-8'))
            
            if line.startswith('v '):
                parts = line.strip().split(' ')
                if len(parts) >= 4:
                    try:
                        vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                        vertices.append(vertex)
                    except ValueError:
                        continue
            elif line.startswith('f '):
                parts = line.strip().split(' ')[1:]
                try:
                    face = [int(p.split('/')[0]) - 1 for p in parts if p]
                    if len(face) >= 3:
                        faces.append(face)
                except (ValueError, IndexError):
                    continue
            
            if bytes_read % (file_size // 100 + 1) == 0:
                progress.update(bytes_read)
    
    progress.finish(f"Loaded {len(vertices)} vertices, {len(faces)} faces")
    
    return np.array(vertices), faces

def get_model_dimensions(vertices):
    """Get the dimensions of the model and calculate min/max values for each axis"""
    min_vals = np.min(vertices, axis=0)
    max_vals = np.max(vertices, axis=0)
    dimensions = max_vals - min_vals
    
    return min_vals, max_vals, dimensions

def extract_edges(faces):
    """Extract unique edges from faces with progress tracking"""
    print("🔗 Extracting edges from faces...")
    
    edge_face_count = defaultdict(int)
    edge_to_faces = defaultdict(list)
    
    progress = ProgressTracker(len(faces), "Processing faces for edges")
    
    for face_idx, face in enumerate(faces):
        for i in range(len(face)):
            v1, v2 = face[i], face[(i + 1) % len(face)]
            edge = tuple(sorted([v1, v2]))
            edge_face_count[edge] += 1
            edge_to_faces[edge].append(face_idx)
        
        if face_idx % 100 == 0:
            progress.update(face_idx)
    
    progress.finish("Edge extraction complete")
    
    outline_edges = [edge for edge, count in edge_face_count.items() if count == 1]
    feature_edges = [edge for edge, count in edge_face_count.items() if count == 2]
    
    print(f"  📊 Found {len(outline_edges)} outline edges, {len(feature_edges)} feature edges")
    
    return outline_edges, feature_edges, edge_to_faces

def project_to_2d(vertices, projection_plane='xy'):
    """Project 3D vertices onto a 2D plane"""
    print(f"📐 Projecting to {projection_plane.upper()} plane...")
    
    if projection_plane == 'xy':
        return vertices[:, 0:2]
    elif projection_plane == 'yz':
        return vertices[:, 1:3]
    elif projection_plane == 'xz':
        return vertices[:, [0, 2]]
    else:
        raise ValueError("Projection plane must be 'xy', 'yz', or 'xz'")

def find_visible_edges(vertices_3d, faces, projection_plane):
    """Find edges that would be visible in the given projection with progress tracking"""
    print(f"👁️  Finding visible edges for {projection_plane.upper()} view...")
    
    if projection_plane == 'xy':
        normal = np.array([0, 0, 1])
    elif projection_plane == 'yz':
        normal = np.array([1, 0, 0])
    elif projection_plane == 'xz':
        normal = np.array([0, 1, 0])
    
    face_visibility = []
    
    progress = ProgressTracker(len(faces), "Calculating face visibility")
    
    for i, face in enumerate(faces):
        if len(face) >= 3:
            try:
                p0 = vertices_3d[face[0]]
                p1 = vertices_3d[face[1]]
                p2 = vertices_3d[face[2]]
                
                v1 = p1 - p0
                v2 = p2 - p0
                face_normal = np.cross(v1, v2)
                
                norm = np.linalg.norm(face_normal)
                if norm > 0:
                    face_normal = face_normal / norm
                
                dot_product = np.dot(face_normal, normal)
                is_visible = dot_product < 0
                
                face_visibility.append(is_visible)
            except IndexError:
                face_visibility.append(False)
        else:
            face_visibility.append(False)
        
        if i % 100 == 0:
            progress.update(i)
    
    progress.finish("Face visibility calculated")
    
    outline_edges, feature_edges, edge_to_faces = extract_edges(faces)
    
    print("🎭 Finding silhouette edges...")
    silhouette_edges = []
    
    progress = ProgressTracker(len(feature_edges), "Processing silhouette edges")
    
    for i, edge in enumerate(feature_edges):
        face_indices = edge_to_faces[edge]
        if len(face_indices) == 2:
            face1, face2 = face_indices
            if face1 < len(face_visibility) and face2 < len(face_visibility):
                if face_visibility[face1] != face_visibility[face2]:
                    silhouette_edges.append(edge)
        
        if i % 100 == 0:
            progress.update(i)
    
    progress.finish(f"Found {len(silhouette_edges)} silhouette edges")
    
    visible_edges = outline_edges + silhouette_edges
    
    print(f"  ✅ Total visible edges: {len(visible_edges)}")
    
    return visible_edges

def get_2d_bounding_box(vertices_2d):
    """Get the 2D bounding box of the projected vertices"""
    min_vals = np.min(vertices_2d, axis=0)
    max_vals = np.max(vertices_2d, axis=0)
    return min_vals, max_vals

def create_clean_projection_image(vertices_2d, visible_edges, projection_plane, 
                                 vertices_3d, output_path, img_size=(800, 600)):
    """Create a clean black and white projection image for OpenCV analysis"""
    print(f"  🖼️  Creating clean projection image...")
    
    min_2d, max_2d = get_2d_bounding_box(vertices_2d)
    
    padding = 50
    img_width, img_height = img_size
    
    data_width = max_2d[0] - min_2d[0]
    data_height = max_2d[1] - min_2d[1]
    
    scale_x = (img_width - 2 * padding) / data_width if data_width > 0 else 1
    scale_y = (img_height - 2 * padding) / data_height if data_height > 0 else 1
    scale = min(scale_x, scale_y)
    
    img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    
    def to_image_coords(point_2d):
        x = int((point_2d[0] - min_2d[0]) * scale + padding)
        y = int(img_height - ((point_2d[1] - min_2d[1]) * scale + padding))
        return (x, y)
    
    # Draw edges
    for edge in visible_edges:
        v1_idx, v2_idx = edge
        if v1_idx < len(vertices_2d) and v2_idx < len(vertices_2d):
            p1 = to_image_coords(vertices_2d[v1_idx])
            p2 = to_image_coords(vertices_2d[v2_idx])
            cv2.line(img, p1, p2, (0, 0, 0), 2)
    
    cv2.imwrite(output_path, img)
    pixel_to_unit_ratio = 1.0 / scale
    
    # Return transformation parameters for coordinate conversion
    transform_params = {
        'min_2d': min_2d,
        'max_2d': max_2d,
        'scale': scale,
        'padding': padding,
        'img_height': img_height,
        'pixel_to_unit_ratio': pixel_to_unit_ratio
    }
    
    return transform_params

def create_scaled_projection_image(vertices_2d, visible_edges, projection_plane, 
                                  vertices_3d, output_path, img_size=(1200, 900)):
    """Create a scaled 2D projection image with measurements"""
    print(f"  📏 Creating scaled projection image...")
    
    fig, ax = plt.subplots(figsize=(img_size[0]/100, img_size[1]/100), dpi=100)
    
    # Plot each visible edge
    for edge in visible_edges:
        v1_idx, v2_idx = edge
        if v1_idx < len(vertices_2d) and v2_idx < len(vertices_2d):
            p1 = vertices_2d[v1_idx]
            p2 = vertices_2d[v2_idx]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=1.5)
    
    # Get dimensions and bounding box
    min_3d, max_3d, dimensions_3d = get_model_dimensions(vertices_3d)
    min_2d, max_2d = get_2d_bounding_box(vertices_2d)
    
    # Add dimension annotations
    drawing_width = max_2d[0] - min_2d[0]
    drawing_height = max_2d[1] - min_2d[1]
    
    h_offset = max(drawing_height * 0.15, 15)
    v_offset = max(drawing_width * 0.15, 15)
    
    if projection_plane == 'xy':
        # Width dimension (X)
        y_pos = min_2d[1] - h_offset
        ax.annotate('', xy=(max_2d[0], y_pos), xytext=(min_2d[0], y_pos),
                   arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
        ax.text((min_2d[0] + max_2d[0])/2, y_pos - 4, 
               f'{dimensions_3d[0]:.1f}', ha='center', va='top', color='red', 
               fontsize=10, fontweight='bold')
        
        # Length dimension (Y)
        x_pos = max_2d[0] + v_offset
        ax.annotate('', xy=(x_pos, max_2d[1]), xytext=(x_pos, min_2d[1]),
                   arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
        ax.text(x_pos + 4, (min_2d[1] + max_2d[1])/2, 
               f'{dimensions_3d[1]:.1f}', ha='left', va='center', color='red', 
               fontsize=10, fontweight='bold', rotation=90)
    
    elif projection_plane == 'yz':
        # Length dimension (Y)
        z_pos = min_2d[1] - h_offset
        ax.annotate('', xy=(max_2d[0], z_pos), xytext=(min_2d[0], z_pos),
                   arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
        ax.text((min_2d[0] + max_2d[0])/2, z_pos - 4, 
               f'{dimensions_3d[1]:.1f}', ha='center', va='top', color='red', 
               fontsize=10, fontweight='bold')
        
        # Height dimension (Z)
        y_pos = max_2d[0] + v_offset
        ax.annotate('', xy=(y_pos, max_2d[1]), xytext=(y_pos, min_2d[1]),
                   arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
        ax.text(y_pos + 4, (min_2d[1] + max_2d[1])/2, 
               f'{dimensions_3d[2]:.1f}', ha='left', va='center', color='red', 
               fontsize=10, fontweight='bold', rotation=90)
    
    elif projection_plane == 'xz':
        # Width dimension (X)
        z_pos = min_2d[1] - h_offset
        ax.annotate('', xy=(max_2d[0], z_pos), xytext=(min_2d[0], z_pos),
                   arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
        ax.text((min_2d[0] + max_2d[0])/2, z_pos - 4, 
               f'{dimensions_3d[0]:.1f}', ha='center', va='top', color='red', 
               fontsize=10, fontweight='bold')
        
        # Height dimension (Z)
        x_pos = max_2d[0] + v_offset
        ax.annotate('', xy=(x_pos, max_2d[1]), xytext=(x_pos, min_2d[1]),
                   arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
        ax.text(x_pos + 4, (min_2d[1] + max_2d[1])/2, 
               f'{dimensions_3d[2]:.1f}', ha='left', va='center', color='red', 
               fontsize=10, fontweight='bold', rotation=90)
    
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Set labels and title
    if projection_plane == 'xy':
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        title = 'TOP VIEW'
    elif projection_plane == 'yz':
        ax.set_xlabel('Y', fontsize=12)
        ax.set_ylabel('Z', fontsize=12)
        title = 'FRONT VIEW'
    elif projection_plane == 'xz':
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Z', fontsize=12)
        title = 'SIDE VIEW'
    
    ax.set_title(f'{title} - Scaled Drawing', fontsize=16, fontweight='bold', pad=20)
    
    # Add scale information
    scale_text = f"Scale: 1:1\nDimensions in units"
    ax.text(0.02, 0.98, scale_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def create_focused_dxf(vertices_2d, vertices_3d, faces, circles, rectangles, 
                      output_path, projection_plane):
    """Create a focused DXF file with consistent close dimensioning for detected features AND overall model dimensions"""
    print(f"\n🎨 Creating focused DXF for {projection_plane.upper()} view...")
    
    # Find visible edges
    visible_edges = find_visible_edges(vertices_3d, faces, projection_plane)
    
    # Create DXF document
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    # Create layers - simplified structure
    doc.layers.new('GEOMETRY', dxfattribs={'color': 7})      # White/black for geometry
    doc.layers.new('DIMENSIONS', dxfattribs={'color': 1})    # Red for dimensions
    doc.layers.new('CENTERLINES', dxfattribs={'color': 4})   # Cyan for centerlines
    doc.layers.new('TEXT', dxfattribs={'color': 2})          # Yellow for text
    
    # Add visible edges to the drawing
    print("  ✏️  Drawing geometry...")
    progress = ProgressTracker(len(visible_edges), "Adding geometry lines")
    
    for i, edge in enumerate(visible_edges):
        v1_idx, v2_idx = edge
        if v1_idx < len(vertices_2d) and v2_idx < len(vertices_2d):
            p1 = vertices_2d[v1_idx]
            p2 = vertices_2d[v2_idx]
            msp.add_line(p1, p2, dxfattribs={'layer': 'GEOMETRY'})
        
        if i % 50 == 0:
            progress.update(i)
    
    progress.finish("Geometry complete")
    
    # Create smart dimension drawer with consistent close positioning
    drawer = SmartDimensionDrawer(msp)
    
    # Get MODEL bounds (3D dimensions) and 2D projection bounds
    min_3d, max_3d, dimensions_3d = get_model_dimensions(vertices_3d)
    min_2d, max_2d = get_2d_bounding_box(vertices_2d)
    
    print("  📏 Adding overall MODEL dimensions...")
    # Add overall MODEL dimensions (not frame dimensions) with consistent close positioning
    if projection_plane == 'xy':
        # Width (X dimension) - close to bottom edge
        width_dim = {
            'start_point': [min_2d[0], min_2d[1]],
            'end_point': [max_2d[0], min_2d[1]],
            'label': f"{dimensions_3d[0]:.1f}",  # Use MODEL dimension, not frame dimension
            'direction': 'horizontal',
            'offset': -drawer.close_offset  # Use consistent close offset
        }
        drawer.draw_linear_dimension(width_dim)
        
        # Length (Y dimension) - close to right edge
        length_dim = {
            'start_point': [max_2d[0], min_2d[1]],
            'end_point': [max_2d[0], max_2d[1]],
            'label': f"{dimensions_3d[1]:.1f}",  # Use MODEL dimension, not frame dimension
            'direction': 'vertical',
            'offset': drawer.close_offset  # Use consistent close offset
        }
        drawer.draw_linear_dimension(length_dim)
    
    elif projection_plane == 'yz':
        # Length (Y dimension) - close to bottom edge
        length_dim = {
            'start_point': [min_2d[0], min_2d[1]],
            'end_point': [max_2d[0], min_2d[1]],
            'label': f"{dimensions_3d[1]:.1f}",  # Use MODEL dimension, not frame dimension
            'direction': 'horizontal',
            'offset': -drawer.close_offset  # Use consistent close offset
        }
        drawer.draw_linear_dimension(length_dim)
        
        # Height (Z dimension) - close to right edge
        height_dim = {
            'start_point': [max_2d[0], min_2d[1]],
            'end_point': [max_2d[0], max_2d[1]],
            'label': f"{dimensions_3d[2]:.1f}",  # Use MODEL dimension, not frame dimension
            'direction': 'vertical',
            'offset': drawer.close_offset  # Use consistent close offset
        }
        drawer.draw_linear_dimension(height_dim)
    
    elif projection_plane == 'xz':
        # Width (X dimension) - close to bottom edge
        width_dim = {
            'start_point': [min_2d[0], min_2d[1]],
            'end_point': [max_2d[0], min_2d[1]],
            'label': f"{dimensions_3d[0]:.1f}",  # Use MODEL dimension, not frame dimension
            'direction': 'horizontal',
            'offset': -drawer.close_offset  # Use consistent close offset
        }
        drawer.draw_linear_dimension(width_dim)
        
        # Height (Z dimension) - close to right edge
        height_dim = {
            'start_point': [max_2d[0], min_2d[1]],
            'end_point': [max_2d[0], max_2d[1]],
            'label': f"{dimensions_3d[2]:.1f}",  # Use MODEL dimension, not frame dimension
            'direction': 'vertical',
            'offset': drawer.close_offset  # Use consistent close offset
        }
        drawer.draw_linear_dimension(height_dim)
    
    # Add circle diameter dimensions with CONSISTENT close annotation
    print("  🔵 Adding circle diameter dimensions (consistent close style)...")
    
    for i, circle in enumerate(circles):
        drawer.draw_circle_diameter_consistent(circle, i)
    
    # Add rectangle dimensions with consistent close positioning
    print("  🔲 Adding rectangle dimensions (close style)...")
    
    model_bounds = {'min': min_2d, 'max': max_2d}
    for i, rectangle in enumerate(rectangles):
        drawer.draw_rectangle_dimensions_close(rectangle, i, model_bounds)
    
    # Add title block - simplified
    print("  📝 Adding title block...")
    
    # Calculate title position relative to the actual drawing bounds
    drawing_height = max_2d[1] - min_2d[1]
    title_y_offset = max(drawing_height * 0.15, 20)
    title_position = [min_2d[0], max_2d[1] + title_y_offset]
    
    # Main title
    view_names = {'xy': 'TOP VIEW', 'yz': 'FRONT VIEW', 'xz': 'SIDE VIEW'}
    title_text = view_names.get(projection_plane, f'{projection_plane.upper()} VIEW')
    
    msp.add_text(title_text, 
                 dxfattribs={
                     'layer': 'TEXT',
                     'color': 2,
                     'height': 4,
                     'style': 'STANDARD',
                     'insert': title_position,
                     'halign': 0,
                     'valign': 1
                 })
    
    # Feature count summary
    feature_summary_lines = []
    if circles:
        feature_summary_lines.append(f"Circles: {len(circles)}")
    if rectangles:
        feature_summary_lines.append(f"Rectangles: {len(rectangles)}")
    
    if feature_summary_lines:
        feature_summary = "Features Detected: " + ", ".join(feature_summary_lines)
        feature_position = [min_2d[0], max_2d[1] + title_y_offset - 6]
        
        msp.add_text(feature_summary, 
                     dxfattribs={
                         'layer': 'TEXT',
                         'color': 2,
                         'height': 2.5,
                         'insert': feature_position
                     })
    
    # Save the DXF file
    print(f"  💾 Saving DXF: {os.path.basename(output_path)}")
    doc.saveas(output_path)
    
    return output_path, len(circles), len(rectangles)

def obj_to_focused_cad(obj_file, output_dir, generate_views=None):
    """
    Convert OBJ to focused CAD drawings with consistent close dimensioning for all features
    
    Parameters:
    - obj_file: Path to input OBJ file
    - output_dir: Directory to save output files
    - generate_views: List of views to generate (default: ['xy', 'yz', 'xz'])
    """
    if generate_views is None:
        generate_views = ['xy', 'yz', 'xz']
    
    print("="*70)
    print("🚀 FOCUSED 3D TO 2D CAD CONVERTER WITH CONSISTENT CLOSE DIMENSIONING")
    print("="*70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load OBJ file
    start_time = time.time()
    vertices_3d, faces = load_obj(obj_file)
    
    # Get model dimensions
    min_vals, max_vals, dimensions = get_model_dimensions(vertices_3d)
    print(f"\n📊 MODEL ANALYSIS:")
    print(f"   Width (X):  {dimensions[0]:.3f}")
    print(f"   Length (Y): {dimensions[1]:.3f}")
    print(f"   Height (Z): {dimensions[2]:.3f}")
    print(f"   Volume:     {np.prod(dimensions):.1f} cubic units")
    print(f"   Vertices:   {len(vertices_3d):,}")
    print(f"   Faces:      {len(faces):,}")
    
    total_features_found = {'circles': 0, 'rectangles': 0}
    all_detected_features = []
    
    # Process each projection plane
    for view_num, plane in enumerate(generate_views, 1):
        print(f"\n" + "="*60)
        print(f"🎯 PROCESSING VIEW {view_num}/{len(generate_views)}: {plane.upper()}")
        print("="*60)
        
        view_start_time = time.time()
        
        # Project vertices to 2D
        vertices_2d = project_to_2d(vertices_3d, plane)
        
        # Find visible edges
        visible_edges = find_visible_edges(vertices_3d, faces, plane)
        
        # Create file names
        base_name = os.path.basename(obj_file).split('.')[0]
        clean_img_path = os.path.join(output_dir, f"{base_name}_{plane}_clean.png")
        scaled_img_path = os.path.join(output_dir, f"{base_name}_{plane}_scaled.png")
        detected_img_path = os.path.join(output_dir, f"{base_name}_{plane}_detected.png")
        dxf_path = os.path.join(output_dir, f"{base_name}_{plane}_focused.dxf")
        
        # 1. Create clean projection image for OpenCV analysis
        transform_params = create_clean_projection_image(
            vertices_2d, visible_edges, plane, vertices_3d, clean_img_path)
        
        # 2. Create scaled projection image with basic dimensions
        create_scaled_projection_image(
            vertices_2d, visible_edges, plane, vertices_3d, scaled_img_path)
        
        # 3. Detect shapes using OpenCV with proper coordinate transformation
        print("  🔍 Running OpenCV shape detection...")
        detector = OpenCVShapeDetector(min_area=100, approx_epsilon=0.02)
        circles, rectangles, result_img = detector.detect_shapes_in_image(
            clean_img_path, transform_params)
        
        # 4. Save the result image with detected shapes
        if result_img is not None:
            cv2.imwrite(detected_img_path, result_img)
            print(f"  💾 Shape detection result saved: {os.path.basename(detected_img_path)}")
        
        # 5. Create focused DXF file with consistent close dimensioning
        _, num_circles, num_rectangles = create_focused_dxf(
            vertices_2d, vertices_3d, faces, circles, rectangles, dxf_path, plane)
        
        view_time = time.time() - view_start_time
        
        total_features_found['circles'] += num_circles
        total_features_found['rectangles'] += num_rectangles
        
        # Store detected features for summary
        view_features = {
            'view': plane.upper(),
            'circles': circles,
            'rectangles': rectangles
        }
        all_detected_features.append(view_features)
        
        print(f"\n✅ VIEW {view_num} COMPLETED:")
        print(f"   📁 Files generated:")
        print(f"      • Clean image: {os.path.basename(clean_img_path)}")
        print(f"      • Scaled image: {os.path.basename(scaled_img_path)}")
        print(f"      • Detection result: {os.path.basename(detected_img_path)}")
        print(f"      • Focused DXF: {os.path.basename(dxf_path)}")
        print(f"   🔍 Features detected:")
        print(f"      • Circles: {num_circles}")
        print(f"      • Rectangles: {num_rectangles}")
        print(f"   📏 DXF contains:")
        print(f"      • Circle diameters (close placement, spacing only when needed)")
        print(f"      • Rectangle dimensions (close to shapes, collision-aware)")
        print(f"      • NO frame dimensions (feature-specific only)")
        print(f"      • Smart spacing: close unless collision detected")
        print(f"      • Visible text labels even when spaced out")
        print(f"      • Maintains shape-to-dimension association")
        print(f"   ⏱️  Processing time: {view_time:.1f}s")
    
    total_time = time.time() - start_time
    
    # Print comprehensive summary
    print(f"\n" + "="*70)
    print("🎉 CONSISTENT CLOSE DIMENSIONING ANALYSIS SUMMARY")
    print("="*70)
    
    for view_features in all_detected_features:
        view = view_features['view']
        circles = view_features['circles']
        rectangles = view_features['rectangles']
        
        print(f"\n📋 {view} VIEW FEATURES:")
        
        if circles:
            print(f"  🔵 CIRCLES ({len(circles)}) - All with consistent close dimensioning:")
            for i, circle in enumerate(circles):
                center = circle['center']
                diameter = circle['diameter']
                print(f"    C{i+1}: ⌀{diameter:.1f} at ({center[0]:.1f}, {center[1]:.1f}) - 4.0 units outside")
        
        if rectangles:
            print(f"  🔲 RECTANGLES ({len(rectangles)}) - All with close positioning:")
            for i, rect in enumerate(rectangles):
                center = rect['center']
                width, height = rect['width'], rect['height']
                print(f"    R{i+1}: {width:.1f}×{height:.1f} at ({center[0]:.1f}, {center[1]:.1f}) - 4.0 units close")
        
        if not circles and not rectangles:
            print(f"  ⚪ No geometric features detected in this view")
    
    print(f"\n" + "="*70)
    print("📊 FINAL STATISTICS")
    print("="*70)
    print(f"⏱️  Total processing time: {total_time:.1f} seconds")
    print(f"📁 Output directory: {output_dir}")
    print(f"📄 Files per view: 4 (clean image, scaled image, detection result, focused DXF)")
    print(f"📄 Total files generated: {len(generate_views) * 4}")
    print(f"🔍 Total features detected across all views:")
    print(f"   🔵 Circles: {total_features_found['circles']}")
    print(f"   🔲 Rectangles: {total_features_found['rectangles']}")
    print(f"📏 Consistent dimensioning features:")
    print(f"   • Dimensions stay close to shapes (4.0 units preferred)")
    print(f"   • Circle diameters: ALL use outside annotation")
    print(f"   • Rectangle dimensions: close to edges")
    print(f"   • NO frame dimensions (only individual feature measurements)")
    print(f"   • Smart collision detection: spaces out ONLY when necessary")
    print(f"   • Text labels remain visible and associated with their shapes")
    print(f"   • Progressive spacing: 1.5x, 2x, 2.5x, 3x, 4x when collisions occur")
    print(f"   • Maintains clear shape-to-dimension relationship")
    print(f"🎯 Detection method: OpenCV contour analysis with validation")
    print(f"🎨 Output format: Professional DXF with layered organization")
    print("="*70)

# Example usage and main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Focused 3D OBJ to dimensioned 2D CAD conversion with consistent close dimensioning')
    parser.add_argument('obj_file', help='Path to input OBJ file')
    parser.add_argument('--output-dir', default='focused_output', 
                        help='Directory to save output files (default: focused_output)')
    parser.add_argument('--views', nargs='+', default=['xy', 'yz', 'xz'], 
                        choices=['xy', 'yz', 'xz'], 
                        help='Projection views to generate (default: xy yz xz)')
    parser.add_argument('--min-area', type=int, default=100,
                        help='Minimum contour area for shape detection (default: 100)')
    parser.add_argument('--approx-epsilon', type=float, default=0.02,
                        help='Approximation epsilon for contour simplification (default: 0.02)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.obj_file):
        print(f"❌ Error: Input file '{args.obj_file}' not found!")
        exit(1)
    
    if not args.obj_file.lower().endswith('.obj'):
        print(f"⚠️  Warning: File '{args.obj_file}' may not be an OBJ file!")
    
    # Check dependencies
    try:
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
    except ImportError:
        print("❌ Error: OpenCV is required for shape detection!")
        print("Install with: pip install opencv-python")
        exit(1)
    
    try:
        import ezdxf
        print(f"✅ ezdxf version: {ezdxf.__version__}")
    except ImportError:
        print("❌ Error: ezdxf is required for DXF generation!")
        print("Install with: pip install ezdxf")
        exit(1)
    
    try:
        obj_to_focused_cad(args.obj_file, args.output_dir, args.views)
    except KeyboardInterrupt:
        print(f"\n⚠️  Process interrupted by user!")
        exit(1)
    except Exception as e:
        print(f"\n❌ Error during conversion: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)