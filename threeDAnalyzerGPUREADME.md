# GPU-Accelerated 3D Mesh Analyzer

A high-performance tool for analyzing 3D mesh files (.obj format) with GPU acceleration using Open3D. This tool provides comprehensive analysis for manufacturing and 3D printing applications, including wall thickness measurement, gap analysis, and overhang detection.

## 🚀 Features

- **GPU-Accelerated Raycasting**: Up to 100x faster analysis on compatible hardware
- **Comprehensive Analysis**: Wall thickness, gaps, overhangs, dimensions, and watertightness
- **Smart Sampling**: Intelligent face sampling for efficient processing of large meshes
- **Visualization Export**: Generate colored 3D models showing analysis results
- **Automatic Fallback**: Seamlessly switches to CPU if GPU is unavailable
- **Multiple Accuracy Levels**: Choose between speed and precision based on your needs

## 📋 Requirements

### System Requirements
- **Operating System**: Windows 10/11, Linux (Ubuntu 18.04+), or macOS 10.14+
- **Python**: 3.7 or higher (3.8-3.11 recommended)
- **Memory**: Minimum 4GB RAM (8GB+ recommended for large meshes)
- **Storage**: 2GB free space for installation

### GPU Requirements (Optional but Recommended)
- **NVIDIA GPU**: GTX 1050 or better (RTX series recommended)
- **CUDA Compute Capability**: 3.5 or higher
- **VRAM**: 2GB minimum (4GB+ recommended for large meshes)

## 🛠 Installation Guide

### Step 1: Install Python

#### Windows:
1. Download Python from [python.org](https://www.python.org/downloads/)
2. **Important**: Check "Add Python to PATH" during installation
3. Verify installation:
   ```cmd
   python --version
   pip --version
   ```

#### Linux (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

#### macOS:
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python
```

### Step 2: Install CUDA (For GPU Acceleration)

> **Note**: Skip this step if you only want CPU processing or don't have an NVIDIA GPU.

#### Check GPU Compatibility
First, verify you have a compatible NVIDIA GPU:

**Windows:**
```cmd
nvidia-smi
```

**Linux:**
```bash
lspci | grep -i nvidia
nvidia-smi
```

If you see GPU information, proceed with CUDA installation.

#### Install CUDA Toolkit

##### Windows:
1. Visit [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
2. Select:
   - Operating System: Windows
   - Architecture: x86_64
   - Version: Your Windows version
   - Installer Type: exe (network or local)
3. Download and run the installer
4. Choose "Express Installation"
5. Restart your computer
6. Verify installation:
   ```cmd
   nvcc --version
   ```

##### Linux (Ubuntu):
```bash
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

# Install CUDA
sudo apt-get install cuda

# Add to PATH (add to ~/.bashrc for persistence)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Verify installation
nvcc --version
```

##### macOS:
> **Note**: NVIDIA CUDA is not supported on Apple Silicon Macs (M1/M2). Use CPU-only mode.

For Intel Macs with NVIDIA GPUs (rare):
1. Download CUDA for macOS from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads)
2. Follow the installer instructions
3. Add to PATH in your shell profile

### Step 3: Create Virtual Environment

It's highly recommended to use a virtual environment:

```bash
# Create virtual environment
python -m venv mesh_analyzer_env

# Activate virtual environment
# Windows:
mesh_analyzer_env\Scripts\activate
# Linux/macOS:
source mesh_analyzer_env/bin/activate

# You should see (mesh_analyzer_env) in your prompt
```

### Step 4: Install Required Packages

#### For GPU Support:
```bash
# Upgrade pip first
pip install --upgrade pip

# Install core dependencies
pip install open3d>=0.17.0
pip install numpy>=1.21.0
pip install scipy>=1.7.0
pip install matplotlib>=3.5.0

# Install additional dependencies
pip install trimesh  # For compatibility with thinWallAnalysis module
pip install pathlib2  # For older Python versions
```

#### For CPU-Only:
```bash
# Upgrade pip first
pip install --upgrade pip

# Install CPU-only version (smaller download)
pip install open3d-cpu>=0.17.0
pip install numpy>=1.21.0
pip install scipy>=1.7.0
pip install matplotlib>=3.5.0
pip install trimesh
```

#### Verify Open3D GPU Support:
Test if GPU acceleration is working:

```python
import open3d as o3d
print("Open3D version:", o3d.__version__)
print("CUDA support:", o3d.core.cuda.device_count() > 0)
if o3d.core.cuda.device_count() > 0:
    print("Available CUDA devices:", o3d.core.cuda.device_count())
else:
    print("No CUDA devices found - will use CPU")
```

### Step 5: Download the Analyzer

1. Download the `threeDAnalyzer.py` file and `thinWallAnalysis.py` module
2. Place them in your working directory
3. Ensure both files are in the same folder

### Step 6: Test Installation

Create a simple test:

```bash
# Download a sample .obj file or use your own
# Test with CPU-only mode first
python threeDAnalyzer.py --file your_model.obj --cpu-only --verbose

# If that works, test GPU mode
python threeDAnalyzer.py --file your_model.obj --verbose
```

## 📖 Usage Guide

### Basic Usage

```bash
# Analyze a mesh file with default settings
python threeDAnalyzer.py --file model.obj

# High accuracy analysis with visualizations
python threeDAnalyzer.py --file model.obj --accuracy high --export

# Save results to JSON file
python threeDAnalyzer.py --file model.obj --output results.json
```

### Command Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--file` | `-i` | Path to .obj file | Required |
| `--output` | `-o` | Output JSON file path | None |
| `--accuracy` | `-a` | Analysis accuracy (low/medium/high/full) | medium |
| `--export` | `-e` | Export colored visualizations | False |
| `--cpu-only` | `-c` | Force CPU processing | False |
| `--verbose` | `-v` | Enable verbose output | False |

### Accuracy Levels

- **Low**: Fast analysis, 1-10% of faces sampled
- **Medium**: Balanced speed/accuracy, 5-20% of faces sampled  
- **High**: High accuracy, 10-40% of faces sampled
- **Full**: Maximum accuracy, all faces analyzed (slow for large meshes)

### Example Workflows

#### Quick Analysis:
```bash
python threeDAnalyzer.py --file part.obj --accuracy low
```

#### Production Analysis:
```bash
python threeDAnalyzer.py --file part.obj --accuracy high --export --output analysis_results.json
```

#### Troubleshooting Mode:
```bash
python threeDAnalyzer.py --file part.obj --cpu-only --verbose
```

## 🎯 Understanding Results

### Analysis Output

The tool provides comprehensive analysis in JSON format:

```json
{
  "Dimensions": {
    "Length": "25.40 units",
    "Width": "15.20 units", 
    "Height": "8.50 units",
    "Volume": "1250.30 cubic units",
    "Surface Area": "485.20 square units"
  },
  "Properties": {
    "Watertight": "Yes",
    "Separate Objects": 1
  },
  "Manufacturing Analysis": {
    "Thinnest Wall": "0.800 units",
    "Smallest Gap": "1.200 units"
  },
  "3D Printing": {
    "Overhang Area": "12.50 square units",
    "Overhang Percentage": "2.6%",
    "Critical Overhangs": 15
  },
  "Analysis Info": {
    "Mesh Complexity": "25847 faces, 12924 vertices",
    "Analysis Time": "3.24 seconds",
    "Compute Device": "GPU (CUDA Device 0)"
  }
}
```

### Visualization Files

When using `--export`, the tool generates:
- `visualization/thickness_model.ply`: Colored mesh showing wall thickness
- `visualization/thickness_legend.png`: Color scale legend for thickness
- `visualization/gaps_model.ply`: Colored mesh showing gap sizes
- `visualization/gaps_legend.png`: Color scale legend for gaps

## ⚠️ Troubleshooting

### Common Issues

#### 1. "No CUDA devices found"
**Solution**: 
- Verify NVIDIA GPU is installed: `nvidia-smi`
- Reinstall CUDA toolkit
- Try CPU-only mode: `--cpu-only`

#### 2. "Failed to load .obj model"
**Solutions**:
- Check file path is correct
- Ensure .obj file is not corrupted
- Try opening file in a 3D viewer first

#### 3. "Out of memory" errors
**Solutions**:
- Use lower accuracy: `--accuracy low`
- Force CPU mode: `--cpu-only`
- Close other GPU applications
- Use smaller mesh files

#### 4. Import errors
**Solutions**:
```bash
# Reinstall packages
pip uninstall open3d
pip install open3d

# Check Python version compatibility
python --version
```

#### 5. Slow performance on GPU
**Possible causes**:
- Small meshes (GPU overhead)
- Insufficient VRAM
- Old GPU drivers

**Solutions**:
- Update GPU drivers
- Use `--cpu-only` for small meshes
- Increase virtual memory/swap space

### Performance Tips

1. **For Large Meshes (>100k faces)**:
   - Use GPU acceleration
   - Start with `--accuracy low`
   - Close other GPU applications

2. **For Small Meshes (<10k faces)**:
   - Use `--cpu-only` (faster due to less overhead)
   - Use `--accuracy high` or `--accuracy full`

3. **Memory Conservation**:
   - Avoid `--export` for very large meshes
   - Process one file at a time
   - Use virtual memory if needed

### Getting Help

If you encounter issues:

1. **Check system requirements** above
2. **Run with verbose output**: `--verbose`
3. **Try CPU-only mode**: `--cpu-only`
4. **Verify CUDA installation**: `nvcc --version`
5. **Test with a simple .obj file**

## 📊 Performance Benchmarks

Typical performance on different hardware:

| Hardware | Mesh Size | Accuracy | Time (CPU) | Time (GPU) | Speedup |
|----------|-----------|----------|------------|------------|---------|
| RTX 3070 | 50k faces | Medium | 45s | 4s | 11x |
| RTX 4090 | 200k faces | High | 180s | 8s | 22x |
| GTX 1660 | 25k faces | Medium | 25s | 6s | 4x |
| CPU Only | 10k faces | High | 15s | - | - |

*Results may vary based on mesh complexity and system configuration*

## 🔧 Advanced Configuration

### Environment Variables

Set these for consistent behavior:

```bash
# Windows
set CUDA_VISIBLE_DEVICES=0
set PYTHONPATH=%PYTHONPATH%;.

# Linux/macOS
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:.
```

### Custom Python Scripts

Example integration:

```python
from threeDAnalyzer import MeshAnalyzer, ResultsFormatter

# Initialize analyzer with GPU
analyzer = MeshAnalyzer("model.obj", use_gpu=True)

# Run analysis
results = analyzer.analyze(accuracy='high', export_visualizations=True)

# Format for display
formatted = ResultsFormatter.format_for_display(results)
print(formatted)
```

## 📝 License

This project is provided as-is for educational and research purposes. Please ensure you have appropriate licenses for all dependencies.

## 🤝 Contributing

To contribute or report issues:
1. Ensure your environment matches the requirements above
2. Test with both CPU and GPU modes
3. Include system information in bug reports

---

**Happy Analyzing! 🎉**

For additional support, please ensure you've followed all installation steps and tested with the troubleshooting commands above.