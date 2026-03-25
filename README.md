To build a whl for Trellis do:

- open the `x64 Native Tools Command Prompt`
- navigate to the folder of Trellis and activate its `.venv`.
- confirm that venv already has pytorch 2.8 installed inside the venv.
- install the wheel: `pip install wheel`
- confirm your python version is 3.11 (`python --version`)
- jump back to this CuMesh repo folder.
- pull additional dependencies `git submodule update --init --recursive`
- point towards your cuda toolkit (watchout for the number at the end) `set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8`
- and also do (watchout for the number at the end) `set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;%PATH%`
- make it work with older cards (GTX 980) and newer ones, including RTX 5090: `set TORCH_CUDA_ARCH_LIST=5.2 6.1 7.5 8.0 8.6 8.9 9.0+PTX`
- also toggle this: `set DISTUTILS_USE_SDK=1`
- build the wheel `pip wheel . --no-build-isolation -w dist`
- you now will have your `.whl` file inside `dist` folder. Users can install it via `pip install` and it will work on the gpu cards specified above.

Original repo description:

# CuMesh: High-Performance Geometry Processing for PyTorch

**CuMesh** is a GPU-accelerated library designed for high-performance 3D geometry processing directly within the PyTorch ecosystem. It provides efficient primitives for mesh cleaning, decimation, remeshing, and UV unwrapping.

Key features include:
*   **CUDA-Accelerated Mesh Operations**: Fast topology queries, simplification, hole filling, and cleaning.
*   **Remeshing**: Remesh arbitrary meshes using narrow-band UDF and Dual Contouring.
*   **UV Unwrapping**: Efficient UV parameterization with `xatlas` enhenced by a fast mesh clustering on the GPU.

## Installation

### Prerequisites
*   Python >= 3.8
*   PyTorch >= 2.4 (with CUDA support)
*   CUDA Toolkit >= 12.4 (matching your PyTorch version)

### Build from Source

```bash
git clone https://github.com/JeffreyXiang/CuMesh.git --recursive
pip install CuMesh --no-build-isolation
```


## Quick Start & Modules

See the [examples](examples) directory for more detailed usage.


## API Reference

### `cumesh.CuMesh`

*   **`init(vertices, faces)`**: Initialize mesh with `[V,3]` and `[F,3]` CUDA tensors.
*   **`read()`**: Return current `(vertices, faces)` tensors.
*   **`simplify(target_num_faces, verbose=False, options={})`**: Fast GPU-accelerated mesh decimation.
*   **`uv_unwrap(verbose=False, ...)`**: Generate UVs using accelerated clustering and Xatlas.
*   **`fill_holes(max_hole_perimeter)`**: Triangulate and close boundary loops.
*   **`repair_non_manifold_edges()`**: Split edges to resolve non-manifold geometry.
*   **`remove_degenerate_faces()`**: Remove zero-area faces or those with NaN normals.
*   **`remove_duplicate_faces()`**: Remove faces with identical vertex indices.
*   **`remove_small_connected_components(min_area)`**: Delete isolated components below area threshold.
*   **`unify_face_orientations()`**: Reorient faces to have consistent winding order.
*   **`compute_face_normals()`**, **`compute_vertex_normals()`**: Trigger normal calculation (access via `read_*_normals`).
*   **`get_connected_components()`**: Compute connectivity (access via `read_connected_components`).
*   **`get_boundary_loops()`**: Detect boundaries (access via `read_boundary_loops`).
*   **Properties**: `num_vertices`, `num_faces`, `num_edges`, `num_boundaries`.

### `cumesh.remeshing`
*   `remesh_narrow_band_dc(...)`: Performs Dual Contouring reconstruction based on the UDF of the input mesh.

### `cumesh.cuBVH`

*NOTE: This is a wrapper around the [`cubvh`](https://github.com/ashawkey/cubvh) library.*

### `cumesh.Atlas`

*NOTE: This is a wrapper around the [`xatlas`](https://github.com/jpcy/xatlas) library.*

*   **`add_mesh(vertices, faces, normals=None, uvs=None)`**: Register mesh geometry (Must be **CPU** tensors).
*   **`compute_charts(max_chart_area, ...)`**: Segment mesh into UV charts (parameterization).
*   **`pack_charts(resolution, padding, ...)`**: Pack generated charts into a texture atlas.
*   **`get_mesh(index)`**: Retrieve processed data as `(vertex_map, faces, uvs)`.
    *   `vertex_map`: Maps new vertex indices to original input indices.


## Acknowledgements

This package builds upon and integrates code from several excellent open-source libraries. We would like to express our gratitude to the authors of:

*   **[cubvh](https://github.com/ashawkey/cubvh)**: For the high-performance CUDA BVH acceleration toolkit.
*   **[xatlas](https://github.com/jpcy/xatlas)**: For the robust UV parameterization and atlas packing library.
*   **[pamo](https://github.com/SarahWeiii/pamo)**: For the reference implementation of the GPU parallel edge collapse algorithm used in our mesh simplification module.

## License

[MIT License](LICENSE)
