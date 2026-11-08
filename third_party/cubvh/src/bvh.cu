// #include <Eigen/Dense>
#include <gpu/common.h>
#include <gpu/triangle.cuh>
#include <gpu/bvh.cuh>
#include <gpu/pcg32.h>

#include <stack>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <unordered_map>

using namespace Eigen;


namespace cumesh {
namespace cubvh {

constexpr float MAX_DIST = 1000.0f;
constexpr float MAX_DIST_SQ = MAX_DIST*MAX_DIST;

__global__ void signed_distance_watertight_kernel(uint32_t n_elements, const Vector3f* __restrict__ positions, float* __restrict__ distances, int64_t* __restrict__ face_id, Vector3f* __restrict__ uvw, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles, bool use_existing_distances_as_upper_bounds);
__global__ void signed_distance_raystab_kernel(uint32_t n_elements, const Vector3f* __restrict__ positions, float* __restrict__ distances, int64_t* __restrict__ face_id, Vector3f* __restrict__ uvw, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles, bool use_existing_distances_as_upper_bounds);
__global__ void unsigned_distance_kernel(uint32_t n_elements, const Vector3f* __restrict__ positions, float* __restrict__ distances, int64_t* __restrict__ face_id, Vector3f* __restrict__ uvw, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles, bool use_existing_distances_as_upper_bounds);
__global__ void raytrace_kernel(uint32_t n_elements, const Vector3f* __restrict__ rays_o, const Vector3f* __restrict__ rays_d, Vector3f* __restrict__ positions, int64_t* __restrict__ face_id, float* __restrict__ depth, const TriangleBvhNode* __restrict__ nodes, const Triangle* __restrict__ triangles);
__global__ void raystab_hits_kernel(uint32_t n_elements, const Vector3f* __restrict__ positions, const int32_t* __restrict__ rng_index, uint32_t first_ray, uint32_t last_ray, bool closest, bool* __restrict__ hits, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles);
__global__ void ray_hits_kernel(uint32_t n_elements, const Vector3f* __restrict__ rays_o, const Vector3f* __restrict__ rays_d, bool* __restrict__ hits, const TriangleBvhNode* __restrict__ nodes, const Triangle* __restrict__ triangles);

struct DistAndIdx {
    float dist;
    uint32_t idx;

    // Sort in descending order!
    __host__ __device__ bool operator<(const DistAndIdx& other) {
        return dist < other.dist;
    }
};

template <typename T>
__host__ __device__ void inline compare_and_swap(T& t1, T& t2) {
    if (t1 < t2) {
        T tmp{t1}; t1 = t2; t2 = tmp;
    }
}

// Sorting networks from http://users.telenet.be/bertdobbelaere/SorterHunter/sorting_networks.html#N4L5D3
template <uint32_t N, typename T>
__host__ __device__ void sorting_network(T values[N]) {
    static_assert(N <= 8, "Sorting networks are only implemented up to N==8");
    if (N <= 1) {
        return;
    } else if (N == 2) {
        compare_and_swap(values[0], values[1]);
    } else if (N == 3) {
        compare_and_swap(values[0], values[2]);
        compare_and_swap(values[0], values[1]);
        compare_and_swap(values[1], values[2]);
    } else if (N == 4) {
        compare_and_swap(values[0], values[2]);
        compare_and_swap(values[1], values[3]);
        compare_and_swap(values[0], values[1]);
        compare_and_swap(values[2], values[3]);
        compare_and_swap(values[1], values[2]);
    } else if (N == 5) {
        compare_and_swap(values[0], values[3]);
        compare_and_swap(values[1], values[4]);

        compare_and_swap(values[0], values[2]);
        compare_and_swap(values[1], values[3]);

        compare_and_swap(values[0], values[1]);
        compare_and_swap(values[2], values[4]);

        compare_and_swap(values[1], values[2]);
        compare_and_swap(values[3], values[4]);

        compare_and_swap(values[2], values[3]);
    } else if (N == 6) {
        compare_and_swap(values[0], values[5]);
        compare_and_swap(values[1], values[3]);
        compare_and_swap(values[2], values[4]);

        compare_and_swap(values[1], values[2]);
        compare_and_swap(values[3], values[4]);

        compare_and_swap(values[0], values[3]);
        compare_and_swap(values[2], values[5]);

        compare_and_swap(values[0], values[1]);
        compare_and_swap(values[2], values[3]);
        compare_and_swap(values[4], values[5]);

        compare_and_swap(values[1], values[2]);
        compare_and_swap(values[3], values[4]);
    } else if (N == 7) {
        compare_and_swap(values[0], values[6]);
        compare_and_swap(values[2], values[3]);
        compare_and_swap(values[4], values[5]);

        compare_and_swap(values[0], values[2]);
        compare_and_swap(values[1], values[4]);
        compare_and_swap(values[3], values[6]);

        compare_and_swap(values[0], values[1]);
        compare_and_swap(values[2], values[5]);
        compare_and_swap(values[3], values[4]);

        compare_and_swap(values[1], values[2]);
        compare_and_swap(values[4], values[6]);

        compare_and_swap(values[2], values[3]);
        compare_and_swap(values[4], values[5]);

        compare_and_swap(values[1], values[2]);
        compare_and_swap(values[3], values[4]);
        compare_and_swap(values[5], values[6]);
    } else if (N == 8) {
        compare_and_swap(values[0], values[2]);
        compare_and_swap(values[1], values[3]);
        compare_and_swap(values[4], values[6]);
        compare_and_swap(values[5], values[7]);

        compare_and_swap(values[0], values[4]);
        compare_and_swap(values[1], values[5]);
        compare_and_swap(values[2], values[6]);
        compare_and_swap(values[3], values[7]);

        compare_and_swap(values[0], values[1]);
        compare_and_swap(values[2], values[3]);
        compare_and_swap(values[4], values[5]);
        compare_and_swap(values[6], values[7]);

        compare_and_swap(values[2], values[4]);
        compare_and_swap(values[3], values[5]);

        compare_and_swap(values[1], values[4]);
        compare_and_swap(values[3], values[6]);

        compare_and_swap(values[1], values[2]);
        compare_and_swap(values[3], values[4]);
        compare_and_swap(values[5], values[6]);
    }
}

template <uint32_t BRANCHING_FACTOR>
class TriangleBvhWithBranchingFactor : public TriangleBvh {
public:
    // Stackless traversal using threaded BVH (escape links). Assumes escape_idx is populated.
    __host__ __device__ static std::pair<int, float> ray_intersect_stackless(Ref<const Vector3f> ro, Ref<const Vector3f> rd, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles) {
        float mint = MAX_DIST;
        int shortest_idx = -1;

        int idx = 0;
        while (idx != -1) {
            const TriangleBvhNode& node = bvhnodes[idx];

            float tbb = node.bb.ray_intersect(ro, rd).x();
            if (tbb >= mint) {
                idx = node.escape_idx;
                continue;
            }

            if (node.left_idx < 0) {
                int end = -node.right_idx-1;
                for (int i = -node.left_idx-1; i < end; ++i) {
                    float t = triangles[i].ray_intersect(ro, rd);
                    if (t < mint) {
                        mint = t;
                        shortest_idx = i;
                    }
                }
                idx = node.escape_idx;
            } else {
                // descend first child; siblings are reached via escape links
                idx = node.left_idx;
            }
        }

        return {shortest_idx, mint};
    }

    __host__ __device__ static std::pair<int, float> closest_triangle_stackless(const Vector3f& point, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles, float max_distance_sq = MAX_DIST_SQ) {
        float shortest_distance_sq = max_distance_sq;
        int shortest_idx = -1;

        int idx = 0;
        while (idx != -1) {
            const TriangleBvhNode& node = bvhnodes[idx];

            float dbb = node.bb.distance_sq(point);
            if (dbb > shortest_distance_sq) {
                idx = node.escape_idx;
                continue;
            }

            if (node.left_idx < 0) {
                int end = -node.right_idx-1;
                for (int i = -node.left_idx-1; i < end; ++i) {
                    float dist_sq = triangles[i].distance_sq(point);
                    if (dist_sq <= shortest_distance_sq) {
                        shortest_distance_sq = dist_sq;
                        shortest_idx = i;
                    }
                }
                idx = node.escape_idx;
            } else {
                idx = node.left_idx;
            }
        }

        if (shortest_idx == -1) {
            shortest_idx = 0;
            shortest_distance_sq = 0.0f;
        }

        return {shortest_idx, std::sqrt(shortest_distance_sq)};
    }

    // For normal averaging around a point on the surface. Returns unnormalized normal.
    __host__ __device__ static Vector3f avg_normal_around_point_stackless(const Vector3f& point, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles) {
        static constexpr float EPSILON = 1e-6f;

        float total_weight = 0;
        Vector3f result = Vector3f::Zero();

        int idx = 0;
        while (idx != -1) {
            const TriangleBvhNode& node = bvhnodes[idx];

            float dbb = node.bb.distance_sq(point);
            if (dbb >= EPSILON) {
                idx = node.escape_idx;
                continue;
            }

            if (node.left_idx < 0) {
                int end = -node.right_idx-1;
                for (int i = -node.left_idx-1; i < end; ++i) {
                    if (triangles[i].distance_sq(point) < EPSILON) {
                        float weight = 1; // TODO: cot weight
                        result += triangles[i].normal();
                        total_weight += weight;
                    }
                }
                idx = node.escape_idx;
            } else {
                idx = node.left_idx;
            }
        }

        return result / total_weight;
    }

    __host__ __device__ static std::pair<int, float> ray_intersect(Ref<const Vector3f> ro, Ref<const Vector3f> rd, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles) {
        FixedIntStack query_stack;
        query_stack.push(0);

        float mint = MAX_DIST;
        int shortest_idx = -1;

        while (!query_stack.empty()) {
            int idx = query_stack.pop();

            const TriangleBvhNode& node = bvhnodes[idx];

            if (node.left_idx < 0) {
                int end = -node.right_idx-1;
                for (int i = -node.left_idx-1; i < end; ++i) {
                    float t = triangles[i].ray_intersect(ro, rd);
                    if (t < mint) {
                        mint = t;
                        shortest_idx = i;
                    }
                }
            } else {
                DistAndIdx children[BRANCHING_FACTOR];

                uint32_t first_child = node.left_idx;

                #pragma unroll
                for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
                    children[i] = {bvhnodes[i+first_child].bb.ray_intersect(ro, rd).x(), i+first_child};
                }

                sorting_network<BRANCHING_FACTOR>(children);

                #pragma unroll
                for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
                    if (children[i].dist < mint) {
                        query_stack.push(children[i].idx);
                    }
                }
            }

            if (query_stack.overflowed()) {
                // Fallback to stackless traversal to guarantee correctness
                return ray_intersect_stackless(ro, rd, bvhnodes, triangles);
            }
        }

        return {shortest_idx, mint};
    }

    __host__ __device__ static std::pair<int, float> closest_triangle(const Vector3f& point, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles, float max_distance_sq = MAX_DIST_SQ) {
        FixedIntStackLarge query_stack;
        query_stack.push(0);

        float shortest_distance_sq = max_distance_sq;
        int shortest_idx = -1;

        while (!query_stack.empty()) {
            int idx = query_stack.pop();

            const TriangleBvhNode& node = bvhnodes[idx];

            if (node.left_idx < 0) {
                int end = -node.right_idx-1;
                for (int i = -node.left_idx-1; i < end; ++i) {
                    float dist_sq = triangles[i].distance_sq(point);
                    if (dist_sq <= shortest_distance_sq) {
                        shortest_distance_sq = dist_sq;
                        shortest_idx = i;
                    }
                }
            } else {
                DistAndIdx children[BRANCHING_FACTOR];

                uint32_t first_child = node.left_idx;

                #pragma unroll
                for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
                    children[i] = {bvhnodes[i+first_child].bb.distance_sq(point), i+first_child};
                }

                sorting_network<BRANCHING_FACTOR>(children);

                #pragma unroll
                for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
                    if (children[i].dist <= shortest_distance_sq) {
                        query_stack.push(children[i].idx);
                    }
                }
            }

            if (query_stack.overflowed()) {
                return closest_triangle_stackless(point, bvhnodes, triangles, shortest_distance_sq);
            }
        }

        if (shortest_idx == -1) {
            // printf("No closest triangle found. This must be a bug! %d\n", BRANCHING_FACTOR);
            shortest_idx = 0;
            shortest_distance_sq = 0.0f;
        }

        return {shortest_idx, std::sqrt(shortest_distance_sq)};
    }

    // Assumes that "point" is a location on a triangle
    __host__ __device__ static Vector3f avg_normal_around_point(const Vector3f& point, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles) {
    FixedIntStack query_stack;
    query_stack.push(0);

        static constexpr float EPSILON = 1e-6f;

        float total_weight = 0;
        Vector3f result = Vector3f::Zero();

        while (!query_stack.empty()) {
            int idx = query_stack.pop();

            const TriangleBvhNode& node = bvhnodes[idx];

            if (node.left_idx < 0) {
                int end = -node.right_idx-1;
                for (int i = -node.left_idx-1; i < end; ++i) {
                    if (triangles[i].distance_sq(point) < EPSILON) {
                        float weight = 1; // TODO: cot weight
                        result += triangles[i].normal();
                        total_weight += weight;
                    }
                }
            } else {
                uint32_t first_child = node.left_idx;

                #pragma unroll
                for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
                    if (bvhnodes[i+first_child].bb.distance_sq(point) < EPSILON) {
                        query_stack.push(i+first_child);
                    }
                }
            }
        }

        if (query_stack.overflowed()) {
            return avg_normal_around_point_stackless(point, bvhnodes, triangles);
        }

        return result / total_weight;
    }

    __host__ __device__ static std::pair<int, float> signed_distance_watertight(const Vector3f& point, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles, float max_distance_sq = MAX_DIST_SQ) {
        auto res = closest_triangle(point, bvhnodes, triangles, max_distance_sq);

        const Triangle& tri = triangles[res.first];
        Vector3f closest_point = tri.closest_point(point);
        Vector3f avg_normal = avg_normal_around_point(closest_point, bvhnodes, triangles);

        return {res.first, std::copysignf(res.second, avg_normal.dot(point - closest_point))};
    }

    // Spellcast3D: whether a ray hits the mesh at all - all that signed_distance_raystab asks of a stab ray,
    // where ray_intersect (the closest-hit traversal above) goes on searching for the nearest hit after its
    // first one. The answer is ray_intersect(ro, rd, ...).first >= 0 for every ray, by construction:
    //  - a triangle counts when Triangle::ray_intersect gives t < MAX_DIST, which is how ray_intersect takes
    //    its first hit (t < mint, and mint is MAX_DIST until then): the same function of the same arguments
    //    and the same comparison;
    //  - a child is left out when its box's entry distance (BoundingBox::ray_intersect(...).x()) is not below
    //    MAX_DIST, which is what ray_intersect leaves out until its first hit (it pushes a child when that
    //    distance is below mint); only after a first hit, when its answer is settled, does it leave out more;
    //  - so both answer "some leaf whose box has an entry distance below MAX_DIST holds a triangle with t below
    //    MAX_DIST", whatever the order of the visit: a box holds its descendants' boxes (min and max over the
    //    same vertex floats) and the slab test is monotone in the box (IEEE subtraction and division, no
    //    contraction possible), so every ancestor of such a leaf, the root included, has an entry distance no
    //    larger than the leaf's. The same holds for the escape-link traversals that both fall back to after a
    //    stack overflow (they test the root too, the stack traversals do not).
    // Only the order of the visit changes, for speed: children whose box lies wholly behind the ray's origin
    // (exit distance below 0) come after their siblings, which come by entry distance as in ray_intersect, so
    // that a ray from inside the mesh meets the surface ahead of it before it walks the boxes behind it.
    __host__ __device__ static bool ray_intersect_any_stackless(Ref<const Vector3f> ro, Ref<const Vector3f> rd, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles) {
        int idx = 0;
        while (idx != -1) {
            const TriangleBvhNode& node = bvhnodes[idx];

            float tbb = node.bb.ray_intersect(ro, rd).x();
            if (tbb >= MAX_DIST) {
                idx = node.escape_idx;
                continue;
            }

            if (node.left_idx < 0) {
                int end = -node.right_idx-1;
                for (int i = -node.left_idx-1; i < end; ++i) {
                    float t = triangles[i].ray_intersect(ro, rd);
                    if (t < MAX_DIST) {
                        return true;
                    }
                }
                idx = node.escape_idx;
            } else {
                idx = node.left_idx;
            }
        }

        return false;
    }

    __host__ __device__ static bool ray_intersect_any(Ref<const Vector3f> ro, Ref<const Vector3f> rd, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles) {
        FixedIntStack query_stack;
        query_stack.push(0);

        while (!query_stack.empty()) {
            int idx = query_stack.pop();

            const TriangleBvhNode& node = bvhnodes[idx];

            if (node.left_idx < 0) {
                int end = -node.right_idx-1;
                for (int i = -node.left_idx-1; i < end; ++i) {
                    float t = triangles[i].ray_intersect(ro, rd);
                    if (t < MAX_DIST) {
                        return true;
                    }
                }
            } else {
                DistAndIdx children[BRANCHING_FACTOR];

                uint32_t first_child = node.left_idx;

                #pragma unroll
                for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
                    Vector2f span = bvhnodes[i+first_child].bb.ray_intersect(ro, rd);
                    // The sort key: the entry distance; MAX_DIST (after every other child) for a box wholly
                    // behind the origin; infinity (never pushed) for a box ray_intersect leaves out before
                    // its first hit. So a child is pushed exactly when its entry distance is below MAX_DIST.
                    float key = span.x();
                    if (!(span.x() < MAX_DIST)) {
                        key = std::numeric_limits<float>::infinity();
                    } else if (span.y() < 0.0f) {
                        key = MAX_DIST;
                    }
                    children[i] = {key, i+first_child};
                }

                sorting_network<BRANCHING_FACTOR>(children);

                #pragma unroll
                for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
                    if (children[i].dist <= MAX_DIST) {
                        query_stack.push(children[i].idx);
                    }
                }
            }

            if (query_stack.overflowed()) {
                // Fallback to stackless traversal to guarantee correctness
                return ray_intersect_any_stackless(ro, rd, bvhnodes, triangles);
            }
        }

        return false;
    }

    __host__ __device__ static std::pair<int, float> signed_distance_raystab(const Vector3f& point, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles, float max_distance_sq = MAX_DIST_SQ, pcg32 rng={}) {
        auto res = closest_triangle(point, bvhnodes, triangles, max_distance_sq);

        Vector2f offset = {rng.next_float(), rng.next_float()};

        static constexpr uint32_t N_STAB_RAYS = 32;
        for (uint32_t i = 0; i < N_STAB_RAYS; ++i) {
            // Use a Fibonacci lattice (with random offset) to regularly
            // distribute the stab rays over the sphere.
            // ref: http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/
            Vector3f d = fibonacci_dir<N_STAB_RAYS>(i, offset);

            // If any of the stab rays goes outside the mesh, the SDF is positive.
            // Spellcast3D: only hit or no hit counts here, so ray_intersect_any, which stops at the first hit
            // (the same answer as ray_intersect(...).first >= 0, see above).
            if (!ray_intersect_any(point, -d, bvhnodes, triangles) || !ray_intersect_any(point, d, bvhnodes, triangles)) {
                return {res.first, res.second};
            }
        }

        return {res.first, -res.second};
    }

    // Spellcast3D: signed_distance_raystab's inside test alone, for any subset of the points of a call and any
    // range [first_ray, last_ray) of its stab directions. `rng` is seeded as the raystab kernel seeds it (from the
    // point's index in the call it stands for), so direction i is the same direction, and each ray's hit or miss
    // is a function of the point, the direction and the tree alone (ray_intersect_any, see above): whether some
    // ray of the range escapes is the answer signed_distance_raystab's loop gives over those directions, in
    // whatever range or order they are asked. With `closest`, a point whose rays all hit is then tested as
    // signed_distance_raystab tests it, by its distance to the nearest triangle (-distance < 0: a point on the
    // surface is not inside), so the whole range answers signed_distance_raystab(...).second < 0 bit for bit and
    // the nearest-triangle search runs only for the points whose rays all hit. True for a point whose rays of
    // the range all hit (and, with closest, that lies off the surface).
    __host__ __device__ static bool raystab_hits(const Vector3f& point, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles, uint32_t first_ray, uint32_t last_ray, bool closest, float max_distance_sq = MAX_DIST_SQ, pcg32 rng={}) {
        Vector2f offset = {rng.next_float(), rng.next_float()};

        static constexpr uint32_t N_STAB_RAYS = 32;     // signed_distance_raystab's
        if (last_ray > N_STAB_RAYS) {
            last_ray = N_STAB_RAYS;
        }
        for (uint32_t i = first_ray; i < last_ray; ++i) {
            Vector3f d = fibonacci_dir<N_STAB_RAYS>(i, offset);
            if (!ray_intersect_any(point, -d, bvhnodes, triangles) || !ray_intersect_any(point, d, bvhnodes, triangles)) {
                return false;
            }
        }

        if (closest) {
            auto res = closest_triangle(point, bvhnodes, triangles, max_distance_sq);
            return -res.second < 0.0f;
        }
        return true;
    }

    void raystab_hits_gpu(uint32_t n_elements, const float* positions, const int32_t* rng_index, uint32_t first_ray, uint32_t last_ray, bool closest, bool* hits, const Triangle* gpu_triangles, cudaStream_t stream) override {
        linear_kernel(raystab_hits_kernel, 0u, stream,
            n_elements,
            (const Vector3f*)positions,
            rng_index,
            first_ray,
            last_ray,
            closest,
            hits,
            m_nodes_gpu.data(),
            gpu_triangles
        );
    }


    void ray_hits_gpu(uint32_t n_elements, const float* rays_o, const float* rays_d, bool* hits, const Triangle* gpu_triangles, cudaStream_t stream) override {
        linear_kernel(ray_hits_kernel, 0u, stream,
            n_elements,
            (const Vector3f*)rays_o,
            (const Vector3f*)rays_d,
            hits,
            m_nodes_gpu.data(),
            gpu_triangles
        );
    }

    void signed_distance_gpu(uint32_t n_elements, uint32_t mode, const float* positions, float* distances, int64_t* face_id, float* uvw, const Triangle* gpu_triangles, cudaStream_t stream) override {

        const Vector3f* positions_vec = (const Vector3f*)positions;
        Vector3f* uvw_vec = (Vector3f*)uvw;

        if (mode == 0) {
            // watertight
            linear_kernel(signed_distance_watertight_kernel, 0u, stream,
                n_elements,
                positions_vec,
                distances,
                face_id,
                uvw_vec,
                m_nodes_gpu.data(),
                gpu_triangles,
                false
            );

        } else {
            // raystab
            linear_kernel(signed_distance_raystab_kernel, 0u, stream,
                n_elements,
                positions_vec,
                distances,
                face_id,
                uvw_vec,
                m_nodes_gpu.data(),
                gpu_triangles,
                false
            );
        }
    }

    void unsigned_distance_gpu(uint32_t n_elements, const float* positions, float* distances, int64_t* face_id, float* uvw, const Triangle* gpu_triangles, cudaStream_t stream) override {

        const Vector3f* positions_vec = (const Vector3f*)positions;
        Vector3f* uvw_vec = (Vector3f*)uvw;

        linear_kernel(unsigned_distance_kernel, 0u, stream,
            n_elements,
            positions_vec,
            distances,
            face_id,
            uvw_vec,
            m_nodes_gpu.data(),
            gpu_triangles,
            false
        );
    }

    void ray_trace_gpu(uint32_t n_elements, const float* rays_o, const float* rays_d, float* positions, int64_t* face_id, float* depth, const Triangle* gpu_triangles, cudaStream_t stream) override {

        // cast float* to Vector3f*
        const Vector3f* rays_o_vec = (const Vector3f*)rays_o;
        const Vector3f* rays_d_vec = (const Vector3f*)rays_d;
        Vector3f* positions_vec = (Vector3f*)positions;
        
        linear_kernel(raytrace_kernel, 0u, stream,
            n_elements,
            rays_o_vec,
            rays_d_vec,
            positions_vec,
            face_id,
            depth,
            m_nodes_gpu.data(),
            gpu_triangles
        );

    }

    // Spellcast3D: the tree built on several threads, the same tree the one-thread build makes, byte for byte.
    //
    // The one-thread build (build_range below, the original algorithm) takes a node off a stack, partitions its
    // triangles into BRANCHING_FACTOR children with nth_element (in place, within the node's own range), gives the
    // children the next node indices, and pushes the internal ones, so the last child's whole subtree is built
    // before the one before it. Two facts make the parallel build exact: a subtree only ever touches its own
    // triangle range (nth_element calls on disjoint ranges are the same calls whatever the thread), and the
    // number of nodes a subtree gets is a function of its triangle count alone (every split is at n/2), so every
    // node index is known before the subtree is built. build_subtree partitions a node as the one-thread build
    // would, computes each child's node count, hands each child its index range in the one-thread order (the
    // last child's descendants first) and builds the children's subtrees on threads of their own, each writing
    // its nodes into its own slice of m_nodes and reordering its own slice of the triangles. A 32 M-triangle
    // mesh (a Pixal3D prop at 1536) took 20 s on one core of a Ryzen 9, the card idle meanwhile.
    //
    // CUBVH_BUILD_THREADS=N picks the thread count (1 = the one-thread build; unset: the machine's threads, at
    // most 64). CUBVH_BUILD_CHECK=1 (developers) builds the tree both ways and compares nodes and triangles.

    using TriIt = std::vector<Triangle>::iterator;

    struct BuildRange {
        int node_idx;
        TriIt begin;
        TriIt end;
    };

    // The partition of a node's triangles into its children: the original code, unchanged.
    static void split_range(const BuildRange& curr, std::array<BuildRange, BRANCHING_FACTOR>& children) {
        children[0].begin = curr.begin;
        children[0].end = curr.end;
        int n_children = 1;
        while (n_children < BRANCHING_FACTOR) {
            for (int i = n_children - 1; i >= 0; --i) {
                auto& child = children[i];

                // Choose axis with maximum standard deviation
                Vector3f mean = Vector3f::Zero();
                for (auto it = child.begin; it != child.end; ++it) {
                    mean += it->centroid();
                }
                mean /= (float)std::distance(child.begin, child.end);

                Vector3f var = Vector3f::Zero();
                for (auto it = child.begin; it != child.end; ++it) {
                    Vector3f diff = it->centroid() - mean;
                    var += diff.cwiseProduct(diff);
                }
                var /= (float)std::distance(child.begin, child.end);

                Vector3f::Index axis;
                var.maxCoeff(&axis);

                auto m = child.begin + std::distance(child.begin, child.end)/2;
                std::nth_element(child.begin, m, child.end, [&](const Triangle& tri1, const Triangle& tri2) { return tri1.centroid(axis) < tri2.centroid(axis); });

                children[i*2].begin = children[i].begin;
                children[i*2+1].end = children[i].end;
                children[i*2].end = children[i*2+1].begin = m;
            }

            n_children *= 2;
        }
    }

    // The nodes of the subtree under an internal node with n triangles, that node included: the children's
    // sizes are what split_range makes them (n/2 and n - n/2, each halved again), a leaf has at most
    // n_primitives_per_leaf triangles.
    static size_t subtree_nodes(size_t n, uint32_t n_primitives_per_leaf, std::unordered_map<size_t, size_t>& memo) {
        auto found = memo.find(n);
        if (found != memo.end()) return found->second;
        const size_t h = n / 2, r = n - h;
        const size_t sizes[4] = {h / 2, h - h / 2, r / 2, r - r / 2};
        size_t total = 1;
        for (size_t s : sizes) {
            total += (s <= n_primitives_per_leaf) ? 1 : subtree_nodes(s, n_primitives_per_leaf, memo);
        }
        memo[n] = total;
        return total;
    }

    // The children of `curr` get the node indices [next, next + BRANCHING_FACTOR); each child's bounding box,
    // and the leaves' triangle ranges, as the one-thread build sets them. Returns the internal children.
    static int make_children(std::vector<TriangleBvhNode>& nodes, std::vector<Triangle>& triangles,
                             uint32_t n_primitives_per_leaf, const BuildRange& curr, size_t& next,
                             std::array<BuildRange, BRANCHING_FACTOR>& children, bool* internal) {
        nodes[curr.node_idx].left_idx = (int)next;
        int n_internal = 0;
        for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
            auto& child = children[i];
            assert(child.begin != child.end);
            child.node_idx = (int)next;
            TriangleBvhNode& node = nodes[next++];
            node.bb = BoundingBox(child.begin, child.end);
            if (std::distance(child.begin, child.end) <= n_primitives_per_leaf) {
                node.left_idx = -(int)std::distance(triangles.begin(), child.begin)-1;
                node.right_idx = -(int)std::distance(triangles.begin(), child.end)-1;
                internal[i] = false;
            } else {
                internal[i] = true;
                ++n_internal;
            }
        }
        nodes[curr.node_idx].right_idx = (int)next;
        return n_internal;
    }

    // The one-thread build of everything under the ranges on the stack, its nodes written from index `next` on.
    static void build_range(std::vector<TriangleBvhNode>& nodes, std::vector<Triangle>& triangles,
                            uint32_t n_primitives_per_leaf, std::stack<BuildRange>& build_stack, size_t next) {
        while (!build_stack.empty()) {
            const BuildRange curr = build_stack.top();
            build_stack.pop();
            std::array<BuildRange, BRANCHING_FACTOR> children;
            split_range(curr, children);
            bool internal[BRANCHING_FACTOR];
            make_children(nodes, triangles, n_primitives_per_leaf, curr, next, children, internal);
            for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
                if (internal[i]) build_stack.push(children[i]);
            }
        }
    }

    // The subtree under `curr` (an internal node whose slot is set already), its nodes from `next` on, with up
    // to `threads` threads: the node's children are built on threads of their own, each child's descendants in
    // the index range the one-thread build would give them.
    static void build_subtree(std::vector<TriangleBvhNode>& nodes, std::vector<Triangle>& triangles,
                              uint32_t n_primitives_per_leaf, const BuildRange& curr, size_t next, int threads) {
        const size_t n = (size_t)std::distance(curr.begin, curr.end);
        if (threads <= 1 || n < 4 * MIN_TRIANGLES_PER_THREAD) {
            std::stack<BuildRange> build_stack;
            build_stack.push(curr);
            build_range(nodes, triangles, n_primitives_per_leaf, build_stack, next);
            return;
        }
        std::array<BuildRange, BRANCHING_FACTOR> children;
        split_range(curr, children);
        bool internal[BRANCHING_FACTOR];
        make_children(nodes, triangles, n_primitives_per_leaf, curr, next, children, internal);

        // the one-thread build takes the last child off its stack first, so its descendants come first
        std::unordered_map<size_t, size_t> memo;
        size_t descendants[BRANCHING_FACTOR];
        size_t total = 0;
        for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
            const size_t count = (size_t)std::distance(children[i].begin, children[i].end);
            descendants[i] = internal[i] ? subtree_nodes(count, n_primitives_per_leaf, memo) - 1 : 0;
            total += descendants[i];
        }
        size_t offset[BRANCHING_FACTOR];
        size_t at = next;
        for (int i = (int)BRANCHING_FACTOR - 1; i >= 0; --i) {
            offset[i] = at;
            at += descendants[i];
        }

        std::vector<std::thread> workers;
        for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
            if (!internal[i]) continue;
            // the threads shared by the children in proportion to their work, one at least
            int share = (int)((threads * descendants[i] + total / 2) / (total ? total : 1));
            if (share < 1) share = 1;
            const BuildRange child = children[i];
            workers.emplace_back([&nodes, &triangles, n_primitives_per_leaf, child, off = offset[i], share]() {
                build_subtree(nodes, triangles, n_primitives_per_leaf, child, off, share);
            });
        }
        for (auto& worker : workers) worker.join();
    }

    static constexpr size_t MIN_TRIANGLES_PER_THREAD = 65536;

    static int build_threads() {
        const char* env = std::getenv("CUBVH_BUILD_THREADS");
        if (env && *env) {
            const int asked = std::atoi(env);
            if (asked >= 1) return asked > 256 ? 256 : asked;
        }
        const unsigned hardware = std::thread::hardware_concurrency();
        return hardware == 0 ? 1 : (hardware > 64 ? 64 : (int)hardware);
    }

    // The nodes of the tree over `triangles` (reordered in place), with `threads` threads.
    static void build_tree(std::vector<TriangleBvhNode>& nodes, std::vector<Triangle>& triangles,
                           uint32_t n_primitives_per_leaf, int threads) {
        std::unordered_map<size_t, size_t> memo;
        nodes.clear();
        nodes.resize(subtree_nodes(triangles.size(), n_primitives_per_leaf, memo));
        nodes.front().bb = BoundingBox(std::begin(triangles), std::end(triangles));
        BuildRange root{0, std::begin(triangles), std::end(triangles)};
        build_subtree(nodes, triangles, n_primitives_per_leaf, root, 1, threads);
    }

    void build(std::vector<Triangle>& triangles, uint32_t n_primitives_per_leaf) override {
        const int threads = build_threads();
        const char* check = std::getenv("CUBVH_BUILD_CHECK");
        if (check && *check && *check != '0' && threads > 1) {
            // developers: the one-thread tree, then the parallel one, compared node by node and triangle by triangle
            std::vector<Triangle> copy = triangles;
            std::vector<TriangleBvhNode> single;
            build_tree(single, copy, n_primitives_per_leaf, 1);
            build_tree(m_nodes, triangles, n_primitives_per_leaf, threads);
            const bool same_nodes = single.size() == m_nodes.size() &&
                std::memcmp(single.data(), m_nodes.data(), single.size() * sizeof(TriangleBvhNode)) == 0;
            // field by field: a Triangle has padding bytes, which a copy leaves undefined
            bool same_triangles = copy.size() == triangles.size();
            for (size_t i = 0; same_triangles && i < copy.size(); ++i) {
                same_triangles = copy[i].id == triangles[i].id && copy[i].a == triangles[i].a &&
                                 copy[i].b == triangles[i].b && copy[i].c == triangles[i].c;
            }
            std::cout << "[cubvh] build check: " << threads << " threads against 1: nodes "
                      << (same_nodes ? "identical" : "DIFFERENT") << ", triangles "
                      << (same_triangles ? "identical" : "DIFFERENT") << " (" << m_nodes.size() << " nodes, "
                      << triangles.size() << " triangles)" << std::endl;
        } else {
            build_tree(m_nodes, triangles, n_primitives_per_leaf, threads);
        }

        // Thread the BVH with escape links for stackless traversal.
        // Initialize all escape indices to -1.
        for (auto& n : m_nodes) n.escape_idx = -1;

        // Recursive lambda to assign escape links in pre-order (child order 0..BRANCHING_FACTOR-1)
        std::function<void(int,int)> thread_bvh = [&](int node_idx, int escape_idx) {
            TriangleBvhNode& node = m_nodes[node_idx];
            node.escape_idx = escape_idx;
            if (node.left_idx < 0) return; // leaf
            int first_child = node.left_idx;
            int end_child = node.right_idx; // exclusive
            for (int c = first_child; c < end_child; ++c) {
                int next_escape = (c+1 < end_child) ? (c+1) : escape_idx;
                thread_bvh(c, next_escape);
            }
        };

        if (!m_nodes.empty()) {
            thread_bvh(0, -1);
        }

        // Spellcast3D: no upload here, so that the build makes no CUDA call and a worker thread can run it
        // while the GPU is busy; the owner calls upload() when it wants the nodes on the card.

        // std::cout << "[INFO] Built TriangleBvh: nodes=" << m_nodes.size() << std::endl;
    }

    TriangleBvhWithBranchingFactor() {}
};

using TriangleBvh4 = TriangleBvhWithBranchingFactor<4>;

std::unique_ptr<TriangleBvh> TriangleBvh::make() {
    return std::unique_ptr<TriangleBvh>(new TriangleBvh4());
}

__global__ void signed_distance_watertight_kernel(
    uint32_t n_elements, const Vector3f* __restrict__ positions,
    float* __restrict__ distances, int64_t* __restrict__ face_id, Vector3f* __restrict__ uvw,
    const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles, bool use_existing_distances_as_upper_bounds
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;

    float max_distance = use_existing_distances_as_upper_bounds ? distances[i] : MAX_DIST;

    Vector3f point = positions[i];

    auto res = TriangleBvh4::signed_distance_watertight(point, bvhnodes, triangles, max_distance*max_distance);

    // write 
    distances[i] = res.second;
    face_id[i] = triangles[res.first].id;

    // optional output
    if (uvw) {
        // get closest point
        Vector3f cpoint = triangles[res.first].closest_point(point);
        // query uvw
        uvw[i] = triangles[res.first].barycentric(cpoint);
    }
}

__global__ void signed_distance_raystab_kernel(
    uint32_t n_elements, const Vector3f* __restrict__ positions,
    float* __restrict__ distances, int64_t* __restrict__ face_id, Vector3f* __restrict__ uvw,
    const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles, bool use_existing_distances_as_upper_bounds
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;

    float max_distance = use_existing_distances_as_upper_bounds ? distances[i] : MAX_DIST;
    pcg32 rng;
    rng.advance(i * 2);

    Vector3f point = positions[i];

    auto res = TriangleBvh4::signed_distance_raystab(point, bvhnodes, triangles, max_distance*max_distance, rng);

    // write 
    distances[i] = res.second;
    face_id[i] = triangles[res.first].id;

    // optional output
    if (uvw) {
        // get closest point
        Vector3f cpoint = triangles[res.first].closest_point(point);
        // query uvw
        uvw[i] = triangles[res.first].barycentric(cpoint);
    }
}

// Spellcast3D: TriangleBvh4::raystab_hits for each point. rng_index (may be null: the point's index here) is the
// point's index in the signed_distance call it stands for, which seeds its stab directions' random offset there
// as signed_distance_raystab_kernel seeds it (rng.advance(i * 2) on a uint32 index), and the nearest-triangle
// search has that kernel's bound (max_distance*max_distance, max_distance = MAX_DIST).
__global__ void raystab_hits_kernel(
    uint32_t n_elements, const Vector3f* __restrict__ positions, const int32_t* __restrict__ rng_index,
    uint32_t first_ray, uint32_t last_ray, bool closest, bool* __restrict__ hits,
    const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;

    uint32_t seed_index = rng_index ? (uint32_t)rng_index[i] : i;
    float max_distance = MAX_DIST;
    pcg32 rng;
    rng.advance(seed_index * 2);

    Vector3f point = positions[i];

    hits[i] = TriangleBvh4::raystab_hits(point, bvhnodes, triangles, first_ray, last_ray, closest, max_distance*max_distance, rng);
}

__global__ void unsigned_distance_kernel(
    uint32_t n_elements, const Vector3f* __restrict__ positions,
    float* __restrict__ distances, int64_t* __restrict__ face_id, Vector3f* __restrict__ uvw,
    const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles, bool use_existing_distances_as_upper_bounds
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;

    float max_distance = use_existing_distances_as_upper_bounds ? distances[i] : MAX_DIST;

    Vector3f point = positions[i];

    auto res = TriangleBvh4::closest_triangle(point, bvhnodes, triangles, max_distance*max_distance);

    // write 
    distances[i] = res.second;
    face_id[i] = triangles[res.first].id;

    // optional output
    if (uvw) {
        // get closest point
        Vector3f cpoint = triangles[res.first].closest_point(point);
        // query uvw
        uvw[i] = triangles[res.first].barycentric(cpoint);
    }
}

// Spellcast3D: whether each ray hits the mesh at all - ray_trace's face_id >= 0 for the same ray, bit for bit
// (ray_intersect_any: the same triangle and box tests as ray_intersect, see above), without the search for the
// nearest hit after the first.
__global__ void ray_hits_kernel(
    uint32_t n_elements, const Vector3f* __restrict__ rays_o, const Vector3f* __restrict__ rays_d,
    bool* __restrict__ hits, const TriangleBvhNode* __restrict__ nodes, const Triangle* __restrict__ triangles
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;

    Vector3f ro = rays_o[i];
    Vector3f rd = rays_d[i];

    hits[i] = TriangleBvh4::ray_intersect_any(ro, rd, nodes, triangles);
}

__global__ void raytrace_kernel(
    uint32_t n_elements, const Vector3f* __restrict__ rays_o, const Vector3f* __restrict__ rays_d, 
    Vector3f* __restrict__ positions, int64_t* __restrict__ face_id, float* __restrict__ depth, 
    const TriangleBvhNode* __restrict__ nodes, const Triangle* __restrict__ triangles
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;

    Vector3f ro = rays_o[i];
    Vector3f rd = rays_d[i];

    auto res = TriangleBvh4::ray_intersect(ro, rd, nodes, triangles);

    // write depth
    depth[i] = res.second;
 
    // intersection point is written back to positions.
    // non-intersect point reaches at most 10 depth
    positions[i] = ro + res.second * rd;

    // write face_id
    if (res.first >= 0) {
        face_id[i] = triangles[res.first].id;
    } else {
        face_id[i] = -1;
    }
}
    
} // namespace cubvh
} // namespace cumesh