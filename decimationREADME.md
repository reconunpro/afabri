# Mesh Decimation Tool

A Python script for reducing the polygon count of 3D mesh files using quadric decimation. This tool is useful for optimizing 3D models for real-time rendering, web applications, or reducing file sizes while preserving visual quality.

## Features

- **Quadric Decimation**: Uses advanced quadric error metrics for high-quality mesh simplification
- **Configurable Reduction**: Specify exact reduction ratios (e.g., 0.5 for 50% reduction)
- **Automatic Normal Computation**: Recalculates vertex normals for proper lighting
- **Progress Feedback**: Shows target triangle count and confirms successful processing
- **Input Validation**: Checks file existence and validates reduction parameters

## Requirements

- Python 3.6+
- Open3D library

## Installation

Install the required dependency:

```bash
pip install open3d
```

## Usage

```bash
python decimation.py --input <input_file> --reduction <ratio> --output <output_file>
```

### Parameters

- `--input`: Path to the input OBJ mesh file
- `--reduction`: Reduction ratio (float between 0 and 1)
  - `0.5` = reduce to 50% of original triangles
  - `0.1` = reduce to 10% of original triangles
- `--output`: Path for the output file (supports .obj, .ply, .stl formats)

### Examples

Reduce a mesh to 50% of its original triangle count:
```bash
python decimation.py --input model.obj --reduction 0.5 --output model_reduced.obj
```

Create a low-poly version with 10% of original triangles:
```bash
python decimation.py --input high_poly.obj --reduction 0.1 --output low_poly.obj
```

## How It Works

1. **Loading**: Reads the input mesh file and computes initial vertex normals
2. **Target Calculation**: Determines the target number of triangles based on the reduction ratio
3. **Decimation**: Applies quadric decimation algorithm to simplify the mesh while preserving shape
4. **Normal Recomputation**: Recalculates vertex normals for the simplified mesh
5. **Export**: Saves the decimated mesh to the specified output file

## Supported Formats

- **Input**: OBJ files (other formats supported by Open3D may work)
- **Output**: OBJ, PLY, STL, and other formats supported by Open3D

## Error Handling

The script includes validation for:
- Input file existence
- Reduction ratio bounds (must be between 0 and 1)
- Clear error messages for common issues

## Tips for Best Results

- **Moderate Reductions**: Start with ratios like 0.5 or 0.3 for good quality preservation
- **Aggressive Reductions**: Use ratios below 0.2 only when file size is critical
- **Preview Results**: Always check the output mesh in a 3D viewer before using in production
- **Backup Originals**: Keep copies of original high-resolution meshes

## Technical Details

The script uses Open3D's quadric decimation algorithm, which:
- Minimizes geometric error during simplification
- Preserves important geometric features
- Maintains mesh topology where possible
- Provides better results than simple vertex removal methods

## Troubleshooting

**"Input file not found"**: Check the file path and ensure the file exists
**"Reduction ratio must be between 0 and 1"**: Use decimal values like 0.5, not percentages like 50
**Poor quality results**: Try a less aggressive reduction ratio or check if the input mesh has issues

## License

This script uses the Open3D library. Please refer to Open3D's license for usage terms.