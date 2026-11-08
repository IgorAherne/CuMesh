#pragma once

#include <gpu/common.h>
#include <gpu/triangle.cuh>
#include <gpu/bounding_box.cuh>
#include <gpu/gpu_memory.h>

#include <memory>

namespace cumesh {
namespace cubvh {

struct TriangleBvhNode {
    BoundingBox bb;
    int left_idx; // negative values indicate leaves
    int right_idx;
    // Threaded BVH escape index for stackless traversal: index of the next node
    // to visit after finishing this node's subtree. -1 indicates termination.
    int escape_idx;
};


template <typename T, int MAX_SIZE=32>
class FixedStack {
public:
    __host__ __device__ void push(T val) {
        // If overflowing, flag and drop the push; a stackless fallback will be used.
        if (m_count >= MAX_SIZE) {
            if (!m_overflowed) {
                printf("WARNING TOO BIG (stack overflow)\n");
            }
            m_overflowed = true;
            return;
        }
        m_elems[m_count++] = val;
    }

    __host__ __device__ T pop() {
        return m_elems[--m_count];
    }

    __host__ __device__ bool empty() const {
        return m_count <= 0;
    }

    __host__ __device__ bool overflowed() const {
        return m_overflowed;
    }

private:
    T m_elems[MAX_SIZE];
    int m_count = 0;
    bool m_overflowed = false;
};

using FixedIntStack = FixedStack<int>;
using FixedIntStackLarge = FixedStack<int, 64>;


class TriangleBvh {

protected:
    std::vector<TriangleBvhNode> m_nodes;
    GPUMemory<TriangleBvhNode> m_nodes_gpu;
    TriangleBvh() {};

public:
    virtual void build(std::vector<Triangle>& triangles, uint32_t n_primitives_per_leaf) = 0;

    virtual void signed_distance_gpu(uint32_t n_elements, uint32_t mode, const float* positions, float* distances, int64_t* face_id, float* uvw, const Triangle* gpu_triangles, cudaStream_t stream) = 0;
    virtual void unsigned_distance_gpu(uint32_t n_elements, const float* positions, float* distances, int64_t* face_id, float* uvw, const Triangle* gpu_triangles, cudaStream_t stream) = 0;
    virtual void ray_trace_gpu(uint32_t n_elements, const float* rays_o, const float* rays_d, float* positions, int64_t* face_id, float* depth, const Triangle* gpu_triangles, cudaStream_t stream) = 0;
    // Spellcast3D: the raystab signed distance's inside test for a subset of a call's points (bvh.cu, raystab_hits)
    virtual void raystab_hits_gpu(uint32_t n_elements, const float* positions, const int32_t* rng_index, uint32_t first_ray, uint32_t last_ray, bool closest, bool* hits, const Triangle* gpu_triangles, cudaStream_t stream) = 0;

    // KIUI: not supported now.
    // virtual bool touches_triangle(const BoundingBox& bb, const Triangle* __restrict__ triangles) const = 0;
    // virtual void build_optix(const GPUMemory<Triangle>& triangles, cudaStream_t stream) = 0;

    static std::unique_ptr<TriangleBvh> make();

    TriangleBvhNode* nodes_gpu() const {
        return m_nodes_gpu.data();
    }

    // Spellcast3D: build() fills the host vector m_nodes only. upload() copies it to the card and
    // release_device() frees the card copy but keeps m_nodes, so the tree can go back to the card later
    // without being built again.
    virtual void upload() {
        m_nodes_gpu.resize_and_copy_from_host(m_nodes);
    }

    // resize(0) rather than free_memory(): free_memory() keeps the element count, and a later resize to the
    // same count would then skip the allocation and copy into a null pointer.
    virtual void release_device() {
        m_nodes_gpu.resize(0);
    }

};

} // namespace cubvh
} // namespace cumesh