# Tolerance Extraction Tool

A Python tool for automatically extracting and analyzing tolerances, dimensions, and ISO fit specifications from PDF engineering drawings.

## Features

- **Tolerance Detection**: Automatically identifies and categorizes different types of tolerances:
  - Plus/minus (±) tolerances
  - Upper/lower limit tolerances (+/-)
  - ISO fit tolerances (H7, g6, etc.)
  - General tolerances from drawing tables

- **GD&T Symbol Recognition**: Converts suspicious characters to proper geometric dimensioning and tolerancing symbols:
  - Diameter (⌀), degree (°), depth (↧), countersink (⌵), counterbore (⌴), etc.

- **Complete Dimension Notation**: Reconstructs full dimension specifications by finding nearby symbols and text elements

- **ISO Fit Analysis**: Supports 44 standard ISO fit codes and calculates actual tolerance values when possible

- **General Tolerance Assignment**: Automatically assigns general tolerances to dimensions without specific tolerances based on decimal places

## Requirements

### Required Dependencies
- Python 3.6+
- PyMuPDF (`fitz`) - For PDF processing

### Optional Dependencies
- `isofits` - For precise ISO fit tolerance calculations

## Installation

1. **Install required dependencies:**
```bash
pip install PyMuPDF
```

2. **Install optional dependency for enhanced ISO fit calculations:**
```bash
pip install isofits
```

3. **Download the script:**
Save `Tolerance-Extraction-Tool.py` to your desired location.

## Usage

### Basic Usage
```bash
python Tolerance-Extraction-Tool.py your_drawing.pdf
```

### Advanced Usage with Options
```bash
python Tolerance-Extraction-Tool.py drawing.pdf --tolerance-font-ratio 0.8 --max-distance 25
```

### Command Line Arguments

- `pdf_file` (required): Path to the PDF file to analyze
- `--tolerance-font-ratio` (optional): Font size ratio for distinguishing main dimensions from tolerances (default: 0.7)
- `--max-distance` (optional): Maximum distance for associating tolerances with dimensions (default: 30)

### Examples

```bash
# Analyze a drawing in the current directory
python Tolerance-Extraction-Tool.py mechanical_part.pdf

# Analyze with custom parameters
python Tolerance-Extraction-Tool.py "C:\Drawings\assembly.pdf" --tolerance-font-ratio 0.6

# Linux/Mac path example
python Tolerance-Extraction-Tool.py /home/user/drawings/part.pdf --max-distance 40
```

## Supported ISO Fit Codes

### Hole Fits (Uppercase)
- **Standard**: E7, F6, F7, F8, G6, G7, H6, H7, H8, H9, H10
- **Transition**: JS6, JS7, K6, K7, M6, M7, N6, N7
- **Interference**: P6, P7, R7

### Shaft Fits (Lowercase)
- **Clearance**: f6, f7, g5, g6, h4, h5, h6, h7, h8, h9
- **Transition**: js5, js6, js7, k5, k6, m5, m6, n5, n6
- **Interference**: p6, r6

## Output Explanation

The tool provides comprehensive analysis results organized into categories:

### 1. Dimensions with ± Tolerances
Dimensions that include plus/minus tolerances in a single specification:
```
⌀25.4 ± 0.1
```

### 2. Dimensions with Upper/Lower Tolerances
Dimensions with separate upper and lower limit tolerances:
```
50.0 → Upper: +0.2 Lower: -0.1
```

### 3. Dimensions with ISO Fit Tolerances
Dimensions using standard ISO fit specifications:
```
⌀16 g6 → g6 (Shaft): +6/-7 μm → 15.993 to 16.006 mm
```

### 4. Dimensions with General Tolerances
Dimensions automatically assigned general tolerances based on decimal places:
```
25.0 → ±0.1 (1 decimal) 
```

## Understanding the Output

- **Complete Notation**: Shows reconstructed dimensions with detected GD&T symbols
- **Page Numbers**: Indicates which page of the PDF contains each dimension
- **Distance Calculations**: When `isofits` is available, shows precise tolerance calculations
- **Symbol Translation**: Maps detected characters to proper engineering symbols

## General Tolerance Detection

The tool automatically detects general tolerance tables with patterns like:
- X = ±0.2 mm (whole numbers)
- X.X = ±0.1 mm (1 decimal place)
- X.XX = ±0.05 mm (2 decimal places)
- X.XXX = ±0.02 mm (3 decimal places)
- ANGULAR = ±0.5° (angular tolerances)

## Troubleshooting

### Common Issues

1. **"No dimension-like texts found"**
   - PDF may contain scanned images rather than text
   - Try OCR processing the PDF first

2. **Poor symbol detection**
   - Adjust `--tolerance-font-ratio` parameter
   - Some PDFs may use non-standard character encodings

3. **Missing ISO fit calculations**
   - Install the `isofits` library: `pip install isofits`
   - Some custom fit codes may not be supported

4. **Incorrect tolerance grouping**
   - Adjust `--max-distance` parameter to change proximity sensitivity
   - Complex drawings may require manual verification

### Performance Tips

- Larger PDFs may take longer to process
- The tool works best with vector-based PDF drawings
- Multiple pages are supported but may increase processing time

## File Format Support

- **Input**: PDF files containing vector text (not scanned images)
- **Text Encoding**: UTF-8 and common PDF text encodings
- **Drawing Standards**: ISO, ANSI, and similar engineering drawing standards

## Limitations

- Requires text-based PDFs (not scanned images)
- GD&T symbol detection depends on character mapping accuracy
- Complex drawings with overlapping elements may need manual verification
- Custom tolerance formats may not be automatically recognized

## Contributing

When reporting issues, please include:
- Sample PDF file (if possible)
- Complete error message
- Python version and dependency versions
- Expected vs. actual output

## License

This tool is provided as-is for engineering analysis purposes. Users should verify results for critical applications.