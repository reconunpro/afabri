# 3Dto2DConverter

A Python utility for converting 3D OBJ models into clean 2D CAD drawings with support for standard orthographic projections and cross-sections.

## Features

- **Clean Edge Detection**: Creates clean outlines and silhouettes instead of wireframes
- **Standard Views**: Generates top (XY), front (YZ), and side (XZ) orthographic projections
- **Cross-Sections**: Creates precise cross-section views at specified positions along any axis
- **DXF Export**: Outputs industry-standard DXF files compatible with most CAD software
- **Preview Images**: Generates PNG preview images for quick reference

## Installation

### Prerequisites

- Python 3.7+
- Required libraries:
  - NumPy
  - Matplotlib
  - ezdxf
  - SciPy (for cross-section generation)

```bash
pip install numpy matplotlib ezdxf scipy
```

## Usage

### Basic Usage

```bash
python 3Dto2DConverter.py path/to/model.obj
```

This will:
1. Create an `output` directory (if it doesn't exist)
2. Generate top (XY), front (YZ), and side (XZ) views
3. Export DXF files and PNG previews for each view

### Command Line Options

```bash
python 3Dto2DConverter.py model.obj [options]
```

#### Basic Options

| Option | Description |
|--------|-------------|
| `--output-dir DIR` | Specify output directory (default: 'output') |
| `--views VIEW1 [VIEW2...]` | Choose specific views to generate (choices: xy, yz, xz) |

#### Cross-Section Options

| Option | Description |
|--------|-------------|
| `--percentage` | Interpret section values as percentages (0.0-1.0) instead of absolute units |
| `--section-x VAL1 [VAL2...]` | Generate cross-sections along X-axis at specified positions |
| `--section-y VAL1 [VAL2...]` | Generate cross-sections along Y-axis at specified positions |
| `--section-z VAL1 [VAL2...]` | Generate cross-sections along Z-axis at specified positions |

### Examples

#### Generate Standard Views

```bash
python 3Dto2DConverter.py model.obj
```

#### Generate Only Top and Front Views

```bash
python 3Dto2DConverter.py model.obj --views xy yz
```

#### Generate Cross-Sections as Percentages

```bash
python 3Dto2DConverter.py model.obj --percentage --section-z 0.25 0.5 0.75
```
This generates cross-sections at 25%, 50%, and 75% along the Z-axis.

#### Generate Cross-Sections as Absolute Units

```bash
python 3Dto2DConverter.py model.obj --section-y 5 10 --section-y -5
```
This generates cross-sections at 5 and 10 units from the bottom, and 5 units from the top along the Y-axis.

#### Comprehensive Example

```bash
python 3Dto2DConverter.py model.obj --output-dir cad_drawings --views xy yz --percentage --section-x 0.5 --section-z 0.25 0.75
```
This will:
- Generate top (XY) and front (YZ) views
- Create a cross-section at 50% along the X-axis
- Create cross-sections at 25% and 75% along the Z-axis
- Save all files to "cad_drawings" directory

## Understanding Cross-Section Values

- **Positive values with `--percentage`**: Percentage from the minimum value (0.0 to 1.0)
  - Example: `--percentage --section-z 0.25` is 25% from the bottom along Z-axis

- **Negative values with `--percentage`**: Percentage from the maximum value
  - Example: `--percentage --section-z -0.25` is 25% from the top along Z-axis

- **Positive values without `--percentage`**: Absolute units from the minimum value
  - Example: `--section-y 10` is 10 units from the minimum Y-coordinate

- **Negative values without `--percentage`**: Absolute units from the maximum value
  - Example: `--section-y -5` is 5 units from the maximum Y-coordinate

## Output Files

For an input file named `model.obj`, the generated files will be:

### Standard Views
- `output/model_xy.dxf` - Top view DXF
- `output/model_xy.png` - Top view preview
- `output/model_yz.dxf` - Front view DXF
- `output/model_yz.png` - Front view preview
- `output/model_xz.dxf` - Side view DXF
- `output/model_xz.png` - Side view preview

### Cross-Sections
- `output/model_section_z_50pct.dxf` - Cross-section at 50% along Z-axis (DXF)
- `output/model_section_z_50pct.png` - Cross-section at 50% along Z-axis (preview)

## Advanced Usage

The script can also be imported and used in other Python programs:

```python
from 3Dto2DConverter import obj_to_clean_cad

# Generate standard views
obj_to_clean_cad('model.obj', 'output')

# Generate specific views and cross-sections
obj_to_clean_cad(
    'model.obj',
    'output',
    generate_views=['xy', 'yz'],
    cross_sections=[
        ('x', 0.5, True),  # X-axis at 50% (percentage)
        ('y', 10, False),  # Y-axis at 10 units (absolute)
        ('z', -5, False)   # Z-axis at 5 units from max (absolute)
    ]
)
```

## Limitations

- Works best with manifold (watertight) 3D models
- Complex cross-sections with multiple intersecting shapes may require manual cleanup
- Very detailed models may have performance implications

## License

This tool is released under the MIT License.
