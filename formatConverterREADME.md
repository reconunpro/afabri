# formatConverter

**formatConverter** is a Python-based tool to convert 3D files to the `.obj` format. It supports multiple 3D file types including STL, STEP, IGES, PLY, GLB, and 3MF, and can operate with either local file storage or Django's file storage system.

## Features

- ✅ Convert common 3D formats to `.obj`
- ✅ Automatically detect and preserve units (e.g., mm, cm, in)
- ✅ Supports both local and Django storage backends
- ✅ Command-line interface for quick conversions
- ✅ Handles temporary files securely
- ✅ Logging support for debugging

## Supported Input Formats

- `.obj`
- `.stl`
- `.ply`
- `.glb`
- `.3mf`
- `.step` / `.stp`
- `.iges` / `.igs`

## Installation

Install the required packages using pip:

```bash
pip install trimesh cadquery pygltflib pythonocc-core
```

If using Django storage backend:

```bash
pip install django
```

## Usage

### As a Python Module

```python
from formatConverter import convert_file

result = convert_file("input.stl", "output.obj", use_local=True)
print(result)
```

### From the Command Line

```bash
python formatConverter.py input.stl output.obj
```

You can also specify the backend:

```bash
python formatConverter.py input.glb output.obj --local      # Force local file system
python formatConverter.py input.glb output.obj --django     # Force Django storage
```

## Output

After conversion, a dictionary is returned containing:

```json
{
  "success": true,
  "error": null,
  "unit": "mm",
  "assumption": false
}
```

- `unit`: the detected unit (e.g., mm, cm, in)
- `assumption`: whether the unit was assumed (`true`) or extracted (`false`)

## Django Integration

The tool attempts to auto-detect Django support. If `django` is installed and properly configured, it will use Django's default storage by default unless overridden with `use_local=True`.

## Logging

Logs are printed to the console when using the CLI and can be configured in your app when used as a module.

## License

MIT License
