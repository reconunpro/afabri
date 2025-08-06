# 3D to 2D CAD Converter with Smart Dimensioning

A Python script that converts 3D OBJ files into professional 2D CAD drawings with automatic feature detection and intelligent dimensioning.

## Overview

This tool transforms 3D models into dimensioned 2D technical drawings by:

- **Projecting 3D geometry** onto 2D planes (Top, Front, Side views)
- **Detecting geometric features** (circles, rectangles) using OpenCV
- **Creating professional DXF files** with smart dimensioning
- **Maintaining consistent annotation style** with dimensions placed close to shapes
- **Avoiding dimension collisions** through intelligent spacing algorithms

### Key Features

- ✅ **Multi-view generation**: XY (Top), YZ (Front), XZ (Side) projections
- ✅ **Automatic shape detection**: Circles and rectangles with validation
- ✅ **Smart dimensioning**: Dimensions placed close to shapes with collision avoidance
- ✅ **Professional output**: Layered DXF files ready for CAD software
- ✅ **Progress tracking**: Real-time feedback for large models
- ✅ **Coordinate accuracy**: Proper transformation between 3D model and 2D projections

## Setup

### Required Libraries

Install the following Python packages:

```bash
pip install numpy matplotlib ezdxf opencv-python
```

**Detailed Requirements:**
- `numpy` - Array operations and mathematical calculations
- `matplotlib` - Image generation and plotting
- `ezdxf` - DXF file creation and manipulation
- `opencv-python` - Computer vision and shape detection
- Standard library modules (included with Python): `os`, `collections`, `math`, `time`, `argparse`

### Verify Installation

The script will automatically check for required dependencies and display version information:

```bash
python 3D-to-2D-With-Dimensions.py --help
```

If successful, you should see:
```
✅ OpenCV version: 4.x.x
✅ ezdxf version: 1.x.x
```

### File Structure

Ensure your project structure looks like this:

```
your_project/
├── 3D-to-2D-With-Dimensions.py
├── your_model.obj
└── focused_output/          # Created automatically
    ├── model_xy_clean.png
    ├── model_xy_scaled.png
    ├── model_xy_detected.png
    ├── model_xy_focused.dxf
    └── ... (similar files for yz and xz views)
```

### Input File Requirements

- **File format**: OBJ files only
- **File structure**: Must contain vertices (`v`) and faces (`f`) data
- **Coordinate system**: Standard 3D coordinates (X, Y, Z)
- **Model requirements**: Closed meshes work best for accurate projections

## Execution

### Basic Usage

Convert an OBJ file to CAD drawings with default settings:

```bash
python 3D-to-2D-With-Dimensions.py path/to/your/model.obj
```

This creates all three standard views (Top, Front, Side) in the `focused_output` directory.

### Command Line Arguments

#### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `obj_file` | Path to input OBJ file | `models/part.obj` |

#### Optional Arguments

| Argument | Default | Description | Example |
|----------|---------|-------------|---------|
| `--output-dir` | `focused_output` | Directory for output files | `--output-dir results` |
| `--views` | `xy yz xz` | Views to generate | `--views xy xz` |
| `--min-area` | `100` | Minimum contour area for detection | `--min-area 50` |
| `--approx-epsilon` | `0.02` | Contour approximation precision | `--approx-epsilon 0.015` |

### Advanced Examples

**Generate only top and side views:**
```bash
python 3D-to-2D-With-Dimensions.py model.obj --views xy xz
```

**Use custom output directory:**
```bash
python 3D-to-2D-With-Dimensions.py model.obj --output-dir my_drawings
```

**Fine-tune shape detection:**
```bash
python 3D-to-2D-With-Dimensions.py model.obj --min-area 200 --approx-epsilon 0.01
```

**Complete custom configuration:**
```bash
python 3D-to-2D-With-Dimensions.py models/complex_part.obj \
    --output-dir technical_drawings \
    --views xy yz \
    --min-area 150 \
    --approx-epsilon 0.025
```

### View Options Explained

| View | Projection | Shows | Typical Use |
|------|------------|-------|-------------|
| `xy` | Top View | X-Y plane (looking down Z-axis) | Plan view, footprint |
| `yz` | Front View | Y-Z plane (looking along X-axis) | Elevation, height |
| `xz` | Side View | X-Z plane (looking along Y-axis) | Profile, cross-section |

### Output Files (per view)

Each view generates 4 files:

1. **`*_clean.png`** - Black/white image for shape detection
2. **`*_scaled.png`** - Scaled drawing with basic dimensions
3. **`*_detected.png`** - Shape detection results visualization
4. **`*_focused.dxf`** - Professional CAD file with smart dimensioning

## Understanding the Output

### DXF File Contents

The generated DXF files contain:

- **GEOMETRY layer**: Model outline and visible edges
- **DIMENSIONS layer**: Smart-placed dimensions and extension lines
- **CENTERLINES layer**: Circle center marks
- **TEXT layer**: Feature labels and annotations

### Dimensioning Logic

- **Circles**: Diameter dimensions placed outside with ⌀ symbol
- **Rectangles**: Width/height dimensions close to edges
- **Overall dimensions**: Model extents (not frame dimensions)
- **Smart spacing**: 4.0 units close, progressive spacing if collisions occur
- **Feature labels**: C1, C2... for circles; R1, R2... for rectangles

### Detection Parameters

| Parameter | Purpose | Effect of Increasing | Effect of Decreasing |
|-----------|---------|---------------------|---------------------|
| `--min-area` | Minimum feature size | Ignores smaller features | Detects more small features |
| `--approx-epsilon` | Contour simplification | More approximate shapes | More precise contours |

## Troubleshooting

### Common Issues

**"Input file not found"**
- Verify the OBJ file path is correct
- Use absolute paths if relative paths don't work

**"OpenCV/ezdxf not found"**
- Install missing dependencies: `pip install opencv-python ezdxf`

**"No shapes detected"**
- Try lowering `--min-area` parameter
- Check if your model has clear geometric features
- Verify the model projects meaningful shapes in the selected view

**Large processing time**
- Use fewer views: `--views xy`
- Consider model complexity and file size
- Progress indicators show estimated completion time

### File Size Considerations

- **Small models** (< 10MB): Process quickly with all views
- **Medium models** (10-100MB): Consider selective views
- **Large models** (> 100MB): Expect longer processing times, monitor progress

### Performance Tips

- Start with one view (`--views xy`) to test settings
- Use `--min-area 200` for faster processing of large models
- Close other applications for memory-intensive models

## Output Integration

### CAD Software Compatibility

Generated DXF files work with:
- AutoCAD
- FreeCAD
- LibreCAD  
- DraftSight
- Fusion 360
- SolidWorks (import)

### Recommended Workflow

1. **Generate all views** with default settings
2. **Review detection results** in `*_detected.png` files
3. **Adjust parameters** if needed and regenerate
4. **Import DXF files** into your preferred CAD software
5. **Add additional annotations** as needed

---

## Example Output

For a model named `bracket.obj`, the script generates:

```
focused_output/
├── bracket_xy_clean.png      # Top view analysis image
├── bracket_xy_scaled.png     # Top view with basic dimensions  
├── bracket_xy_detected.png   # Top view detection results
├── bracket_xy_focused.dxf    # Top view CAD file
├── bracket_yz_clean.png      # Front view analysis image
├── bracket_yz_scaled.png     # Front view with basic dimensions
├── bracket_yz_detected.png   # Front view detection results
├── bracket_yz_focused.dxf    # Front view CAD file
├── bracket_xz_clean.png      # Side view analysis image
├── bracket_xz_scaled.png     # Side view with basic dimensions
├── bracket_xz_detected.png   # Side view detection results
└── bracket_xz_focused.dxf    # Side view CAD file
```

The DXF files contain professional technical drawings with intelligent dimensioning that maintains clear associations between geometric features and their measurements.