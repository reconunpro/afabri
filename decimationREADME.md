# Enhanced Mesh Decimation Tool

A Python script for intelligent mesh decimation with two specialized methods: standard reduction decimation and feature-aware uniform decimation optimized for engineering models and raycasting applications.

## Features

- **Standard Decimation**: Traditional percentage-based mesh reduction
- **Feature-Aware Uniform Decimation**: Creates triangles of similar sizes while preserving engineering features
- **Automatic Feature Detection**: Identifies and preserves sharp edges, thin walls, and geometric details
- **Raycasting Optimization**: Generates uniform triangle distributions ideal for thin-wall analysis
- **Engineering Model Support**: Specialized for CAD/engineering meshes with critical small features

## Installation

### Requirements
```bash
pip install open3d numpy
```

### Supported File Formats
- Input: OBJ, PLY, STL, OFF, and other formats supported by Open3D
- Output: Same formats as input

## Usage

### Feature-Aware Uniform Decimation (Recommended for Engineering Models)

**Basic usage:**
```bash
python decimation_script.py --input model.obj --target-edge-length 0.05 --output uniform_model.obj
```

**Advanced usage with custom parameters:**
```bash
python decimation_script.py \
    --input complex_part.obj \
    --target-edge-length 0.02 \
    --feature-angle 25 \
    --max-iterations 5 \
    --output decimated_part.obj
```

### Standard Decimation (Original Method)

```bash
python decimation_script.py --input model.obj --reduction 0.3 --output reduced_model.obj
```

## Parameters

### Common Parameters
- `--input`: Path to input mesh file (required)
- `--output`: Path to output mesh file (required)

### Feature-Aware Decimation Parameters
- `--target-edge-length`: Desired average edge length for uniform triangles (required for this method)
- `--feature-angle`: Angle threshold in degrees for feature detection (default: 30°)
- `--max-iterations`: Maximum number of refinement iterations (default: 3)

### Standard Decimation Parameters
- `--reduction`: Reduction ratio between 0 and 1 (e.g., 0.5 = 50% reduction)

## When to Use Each Method

### Use Feature-Aware Uniform Decimation When:
- ✅ **Raycasting Analysis**: Detecting thin walls, gaps, or performing distance measurements
- ✅ **Engineering Models**: CAD parts with critical small features
- ✅ **Uniform Sampling**: Need consistent triangle density across the mesh
- ✅ **Feature Preservation**: Sharp edges and details must be maintained
- ✅ **Quality Control**: Wall thickness analysis or geometric validation

### Use Standard Decimation When:
- ✅ **Simple Reduction**: Just need to reduce file size by a specific percentage
- ✅ **Organic Models**: Models without critical engineering features
- ✅ **Quick Processing**: Fast reduction without feature analysis
- ✅ **Legacy Compatibility**: Maintaining existing workflows

## Algorithm Details

### Feature-Aware Uniform Decimation Process

1. **Feature Detection**
   - Analyzes dihedral angles between adjacent triangles
   - Identifies edges with angles > threshold as geometric features
   - Automatically detects boundary edges and sharp corners

2. **Statistical Analysis**
   - Computes current edge length distribution
   - Calculates uniformity metrics (mean, standard deviation)
   - Determines optimal triangle count for target edge length

3. **Iterative Refinement**
   - Uses geometric scaling: triangle_count ∝ 1/edge_length²
   - Applies quadric decimation with boundary preservation
   - Monitors convergence to target edge length (±10% tolerance)

4. **Quality Assurance**
   - Removes degenerate triangles and duplicate vertices
   - Eliminates non-manifold edges
   - Validates mesh topology integrity

### Key Benefits for Raycasting

- **Uniform Ray Density**: Similar triangle sizes ensure consistent sampling
- **Feature Preservation**: Critical thin walls and edges are maintained
- **Predictable Results**: Target edge length gives controllable triangle size
- **Quality Metrics**: Reports uniformity statistics for validation

## Examples for Engineering Applications

### Thin Wall Analysis
```bash
# Create uniform mesh for detecting 2mm thin walls
python decimation_script.py \
    --input housing.obj \
    --target-edge-length 0.5 \
    --feature-angle 20 \
    --output housing_raycast.obj
```

### Heat Exchanger Modeling
```bash
# Preserve fine channels while reducing mesh size
python decimation_script.py \
    --input heat_exchanger.obj \
    --target-edge-length 0.1 \
    --feature-angle 35 \
    --max-iterations 4 \
    --output heat_exchanger_optimized.obj
```

### Mechanical Part Inspection
```bash
# Optimize for gap detection and clearance analysis
python decimation_script.py \
    --input assembly.obj \
    --target-edge-length 0.03 \
    --feature-angle 25 \
    --output assembly_inspection.obj
```

## Output Information

The script provides detailed statistics:

```
--- Final Results ---
Final mesh: 15,432 vertices, 28,756 triangles
Final avg edge length: 0.0501
Final std edge length: 0.0087
Edge length uniformity (1/CV): 5.76
Overall reduction ratio: 0.342 (65.8% reduction)
```

### Interpreting Results
- **Edge length uniformity**: Higher values indicate more uniform triangles (>3 is good, >5 is excellent)
- **Standard deviation**: Lower values mean more consistent triangle sizes
- **Reduction ratio**: Actual percentage of triangles retained

## Troubleshooting

### Common Issues

**"Target edge length achieved!" after 1 iteration**
- ✅ Normal behavior when mesh is already close to target size

**High standard deviation in edge lengths**
- Try smaller `--target-edge-length` value
- Increase `--max-iterations` for more refinement

**Important features being removed**
- Decrease `--feature-angle` to detect more features (try 15-25°)
- Check if input mesh has sufficient resolution for features

**Mesh becomes too dense**
- Increase `--target-edge-length` value
- Use standard decimation with `--reduction` instead

### Performance Notes

- **Large meshes**: Processing time scales with mesh complexity
- **Very small target edge lengths**: May result in minimal reduction
- **Complex geometries**: Feature detection adds processing time but improves quality

## License

This tool is designed for engineering and scientific applications. Please ensure your use case complies with Open3D's license terms.

## Contributing

For feature requests or issues specifically related to engineering mesh processing, please provide:
- Input mesh characteristics (triangle count, feature types)
- Target application (raycasting, analysis type)
- Current parameter settings and results