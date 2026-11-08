#pragma once

#include <Eigen/Dense>
#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>
#include <cuda_runtime.h>
#include <memory>

using namespace Eigen;

using Verts = Matrix<float, Dynamic, 3, RowMajor>;
using Trigs = Matrix<uint32_t, Dynamic, 3, RowMajor>;

namespace cumesh {
namespace cubvh {

// abstract class of raytracer
class cuBVH {
public:
    cuBVH() {}
    virtual ~cuBVH() {}

    virtual void ray_trace(at::Tensor rays_o, at::Tensor rays_d, at::Tensor positions, at::Tensor face_id, at::Tensor depth) = 0;
    virtual void unsigned_distance(at::Tensor positions, at::Tensor distances, at::Tensor face_id, at::optional<at::Tensor> uvw) = 0;
    virtual void signed_distance(at::Tensor positions, at::Tensor distances, at::Tensor face_id, at::optional<at::Tensor> uvw, uint32_t mode) = 0;
    // Spellcast3D: signed_distance(mode raystab)'s inside test alone, per point (hits: bool), over the stab
    // directions [first_ray, last_ray), each point's directions seeded from rng_index (int32; none: its index
    // here); with closest, a point whose rays all hit is also held off the surface (bvh.cu, raystab_hits).
    // Spellcast3D: whether each ray hits the mesh at all (hits: bool) - ray_trace's face_id >= 0 for the same
    // ray, without the search for the nearest hit after the first.
    virtual void ray_hits(at::Tensor rays_o, at::Tensor rays_d, at::Tensor hits) = 0;
    virtual void raystab_hits(at::Tensor positions, at::optional<at::Tensor> rng_index, at::Tensor hits, uint32_t first_ray, uint32_t last_ray, bool closest) = 0;

    // Spellcast3D: the card copy of the tree (nodes and triangles) on demand. upload() makes it when it is
    // not there; release_device() frees it and keeps the host copy, so a later upload needs no rebuild; the
    // three queries above upload first when the copy is not there.
    virtual void upload() = 0;
    virtual void release_device() = 0;
    virtual bool is_on_device() = 0;
};

// function to create an implementation of cuBVH. Spellcast3D: with upload false the tree is built on the
// CPU and no CUDA call is made at all, so a worker thread can build it while the GPU thread works.
cuBVH* create_cuBVH(Ref<const Verts> vertices, Ref<const Trigs> triangles, bool upload = true);
// floodfill
at::Tensor floodfill(at::Tensor grid);

// sparse marching cubes
std::tuple<at::Tensor, at::Tensor> sparse_marching_cubes(at::Tensor coords, at::Tensor corners, double iso_d, bool ensure_consistency = false);

// GPU ND integer hash table (default D=3) - abstract interface
class cuHashTable {
public:
    cuHashTable() {}
    virtual ~cuHashTable() {}
    virtual void set_num_dims(int d) = 0;
    virtual int get_num_dims() const = 0;
    virtual void resize(int capacity) = 0;
    virtual void prepare() = 0;
    virtual void insert(at::Tensor coords) = 0;
    virtual void build(at::Tensor coords) = 0;
    virtual at::Tensor search(at::Tensor queries) const = 0;
};

// Factory to create an implementation of cuHashTable
cuHashTable* create_cuHashTable();

} // namespace cubvh
} // namespace cumesh