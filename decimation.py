import open3d as o3d
import argparse
import os

def decimate_mesh(input_path, reduction_ratio, output_path):
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
    parser = argparse.ArgumentParser(description="Decimate a 3D OBJ mesh.")
    parser.add_argument("--input", type=str, required=True, help="Path to input OBJ file")
    parser.add_argument("--reduction", type=float, required=True, help="Reduction ratio (0 < ratio < 1)")
    parser.add_argument("--output", type=str, required=True, help="Path to output file (e.g., reduced.obj)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        exit(1)

    if not (0 < args.reduction < 1):
        print("Error: Reduction ratio must be between 0 and 1 (e.g., 0.5 for 50%)")
        exit(1)

    decimate_mesh(args.input, args.reduction, args.output)
