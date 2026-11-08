import torch

# CUDA extension
from . import _cubvh as _backend

_sdf_mode_to_id = {
    'watertight': 0,
    'raystab': 1,
}

# Spellcast3D: the stab directions of signed_distance(mode='raystab'); each is stabbed both ways (bvh.cu N_STAB_RAYS)
RAYSTAB_DIRECTIONS = 32

class cuBVH():
    def __init__(self, vertices, triangles, upload=True):
        # vertices: np.ndarray, [N, 3]
        # triangles: np.ndarray, [M, 3]
        # upload: copy the tree to the card now, as upstream does. Spellcast3D passes False to build it on
        # the CPU only, on a worker thread, without taking card memory; upload() or the first query copies it.

        if torch.is_tensor(vertices): vertices = vertices.detach().cpu().numpy()
        if torch.is_tensor(triangles): triangles = triangles.detach().cpu().numpy()

        # check inputs
        assert triangles.shape[0] > 8, "BVH needs at least 8 triangles."
        
        # implementation
        self.impl = _backend.create_cuBVH(vertices, triangles, bool(upload))

    def upload(self):
        # Copies the tree (nodes and triangles) to the card when it is not there, without Python's lock so
        # other threads run meanwhile. The queries do it themselves when needed (holding the lock).
        self.impl.upload()

    def release_device(self):
        # Frees the card copy and keeps the host copy, so a later upload() or query needs no rebuild.
        self.impl.release_device()

    @property
    def on_device(self):
        return self.impl.on_device

    def ray_trace(self, rays_o, rays_d):
        # rays_o: torch.Tensor, float, [N, 3]
        # rays_d: torch.Tensor, float, [N, 3]

        rays_o = rays_o.float().contiguous()
        rays_d = rays_d.float().contiguous()

        if not rays_o.is_cuda: rays_o = rays_o.cuda()
        if not rays_d.is_cuda: rays_d = rays_d.cuda()

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.view(-1, 3)
        rays_d = rays_d.view(-1, 3)

        N = rays_o.shape[0]

        # init output buffers
        positions = torch.empty(N, 3, dtype=torch.float32, device=rays_o.device)
        face_id = torch.empty(N, dtype=torch.int64, device=rays_o.device)
        depth = torch.empty(N, dtype=torch.float32, device=rays_o.device)
        
        self.impl.ray_trace(rays_o, rays_d, positions, face_id, depth) # [N, 3]

        positions = positions.view(*prefix, 3)
        face_id = face_id.view(*prefix)
        depth = depth.view(*prefix)

        return positions, face_id, depth

    def unsigned_distance(self, positions, return_uvw=False):
        # positions: torch.Tensor, float, [N, 3]

        positions = positions.float().contiguous()

        if not positions.is_cuda: positions = positions.cuda()

        prefix = positions.shape[:-1]
        positions = positions.view(-1, 3)

        N = positions.shape[0]

        # init output buffers
        distances = torch.empty(N, dtype=torch.float32, device=positions.device)
        face_id = torch.empty(N, dtype=torch.int64, device=positions.device)

        if return_uvw:
            uvw = torch.empty(N, 3, dtype=torch.float32, device=positions.device)
        else:
            uvw = None
        
        self.impl.unsigned_distance(positions, distances, face_id, uvw) # [N, 3]

        distances = distances.view(*prefix)
        face_id = face_id.view(*prefix)
        if uvw is not None:
            uvw = uvw.view(*prefix, 3)

        return distances, face_id, uvw


    def raystab_inside(self, positions, rng_index=None, first_ray=0, last_ray=RAYSTAB_DIRECTIONS, closest=True):
        # Spellcast3D: signed_distance(mode='raystab')'s inside test alone, as a bool per point: whether every
        # stab ray of the directions [first_ray, last_ray) hits the mesh (both ways), and with `closest` also that
        # the point lies off the surface. Over all the directions with closest it is exactly
        # signed_distance(positions, mode='raystab')[0] < 0, without the nearest-triangle search for the points
        # a ray leaves. rng_index (int32, per point): the point's index in the signed_distance call it stands for,
        # which seeds its directions there; default its index in this call. So a subset of a call's points, or
        # its directions asked in parts (the points whose rays all hit in one range asked again for the next),
        # gets every point the answer the whole call gives it.
        # positions: torch.Tensor, float, [N, 3]

        positions = positions.float().contiguous()

        if not positions.is_cuda: positions = positions.cuda()

        prefix = positions.shape[:-1]
        positions = positions.view(-1, 3)

        N = positions.shape[0]

        if rng_index is not None:
            rng_index = rng_index.to(device=positions.device, dtype=torch.int32).contiguous().view(-1)
            assert rng_index.shape[0] == N, "one rng_index per point"

        hits = torch.empty(N, dtype=torch.bool, device=positions.device)
        if N > 0:
            self.impl.raystab_hits(positions, rng_index, hits, int(first_ray), int(last_ray), bool(closest))

        return hits.view(*prefix)

    def signed_distance(self, positions, return_uvw=False, mode='watertight'):
        # positions: torch.Tensor, float, [N, 3]

        positions = positions.float().contiguous()

        if not positions.is_cuda: positions = positions.cuda()

        prefix = positions.shape[:-1]
        positions = positions.view(-1, 3)

        N = positions.shape[0]

        # init output buffers
        distances = torch.empty(N, dtype=torch.float32, device=positions.device)
        face_id = torch.empty(N, dtype=torch.int64, device=positions.device)

        if return_uvw:
            uvw = torch.empty(N, 3, dtype=torch.float32, device=positions.device)
        else:
            uvw = None
        
        self.impl.signed_distance(positions, distances, face_id, uvw, _sdf_mode_to_id[mode]) # [N, 3]

        distances = distances.view(*prefix)
        face_id = face_id.view(*prefix)
        if uvw is not None:
            uvw = uvw.view(*prefix, 3)

        return distances, face_id, uvw
