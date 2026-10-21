#!/usr/bin/env python3
"""Precompute the isotropic heat-kernel texture basis (HKTEx) for a mesh.

Builds the cotangent Laplacian + mass matrix, computes the first K eigenpairs of
the generalized problem W φ = λ M φ (scipy eigsh), places S heat sources on the
surface with random colors, and dumps everything to a binary file for the
mupsi2 C++ renderer to load.

Usage: precompute_heat.py <mesh.obj> <out.bin> [K=64] [S=32] [t=1e-2]
"""
import sys
import struct
import colorsys

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


def load_obj(path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                toks = line.split()[1:]
                idx = [int(t.split("/")[0]) - 1 for t in toks[:3]]
                faces.append(idx)
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int64)


def cotan_laplacian(verts, faces):
    n = verts.shape[0]
    rows, cols, vals = [], [], []
    area = np.zeros(n)

    def cot(a, b):
        return float(np.dot(a, b) / max(np.linalg.norm(np.cross(a, b)), 1e-12))

    for (i, j, k) in faces:
        vi, vj, vk = verts[i], verts[j], verts[k]
        e_ij, e_jk, e_ki = vj - vi, vk - vj, vi - vk
        e_ik, e_ji, e_kj = -e_ki, -e_ij, -e_jk
        c_i = cot(e_ij, e_ik)   # angle at i
        c_j = cot(e_ji, e_jk)   # angle at j
        c_k = cot(e_ki, e_kj)   # angle at k
        # edge (j,k) opposite i
        for (a, b, w) in [(j, k, c_i), (k, i, c_j), (i, j, c_k)]:
            rows += [a, b]
            cols += [b, a]
            vals += [0.5 * w, 0.5 * w]
        tri_area = 0.5 * np.linalg.norm(np.cross(e_ij, -e_ki))
        area[i] += tri_area / 3.0
        area[j] += tri_area / 3.0
        area[k] += tri_area / 3.0

    W = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()
    W = (W + W.T) * 0.5
    W.setdiag(0.0)
    W.setdiag(-np.asarray(W.sum(axis=1)).ravel())
    return W, area


def uniform_surface_samples(verts, faces, n):
    """Uniformly sample n points on the mesh, return (face_id, barycentric)."""
    areas = []
    for (i, j, k) in faces:
        a = 0.5 * np.linalg.norm(np.cross(verts[j] - verts[i], verts[k] - verts[i]))
        areas.append(a)
    areas = np.array(areas)
    probs = areas / areas.sum()
    fid = np.random.choice(len(faces), size=n, p=probs)
    bary = np.random.rand(n, 3)
    s = bary.sum(axis=1)
    bary = bary / s[:, None]
    return fid, bary


def main():
    mesh = sys.argv[1]
    out = sys.argv[2]
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    S = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    t = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-2

    verts, faces = load_obj(mesh)
    N = verts.shape[0]
    print(f"mesh: {N} verts, {faces.shape[0]} faces")

    # Normalize to a canonical size (diag = 1) so the LBO eigenvalues are O(1);
    # eigenvectors are scale-invariant, so they stay valid for the renderer's
    # auto-fit mesh (same topology).
    c = 0.5 * (verts.min(0) + verts.max(0))
    verts = verts - c
    diag = np.linalg.norm(verts.max(0) - verts.min(0))
    verts = verts / diag

    W, mass = cotan_laplacian(verts, faces)

    print(f"eigendecomposition (K={K})...")
    eps = 1e-8
    evals, evecs = eigsh(-W, k=K, M=sparse.diags(mass), sigma=eps, which="LM")
    lam = np.clip(evals, 0.0, None)          # LBO eigenvalues (ascending, λ0≈0)
    Phi = evecs.astype(np.float32)            # (N, K), columns = eigenvectors
    print(f"lambda[0..4] = {lam[:5]}")

    # Place S heat sources on the surface.
    rng = np.random.default_rng(0)
    fids, barys = uniform_surface_samples(verts, faces, S)
    colors = np.array([colorsys.hsv_to_rgb(rng.random(), 1.0, 1.0) for _ in range(S)], dtype=np.float32)
    # threshold in [0.3, 0.99], sharpness in [10, 50]
    tau = (0.3 + 0.69 * rng.random(S)).astype(np.float32)
    sharp = (10.0 + 40.0 * rng.random(S)).astype(np.float32)

    # Source eigenvector phi(s) via barycentric interpolation.
    phi_s = np.zeros((S, K), dtype=np.float32)
    for s in range(S):
        f = faces[fids[s]]
        b = barys[s]
        phi_s[s] = (Phi[f[0]] * b[0] + Phi[f[1]] * b[1] + Phi[f[2]] * b[2])

    c_mean = np.mean(colors, axis=0).astype(np.float32)
    residual = (colors - c_mean).astype(np.float32)   # store residuals (repo convention)
    bh_sigma = 0.1                                   # biharmonic-distance Gaussian sigma

    with open(out, "wb") as fp:
        fp.write(struct.pack("<IIII", K, N, S, faces.shape[0]))
        fp.write(struct.pack("<f", np.float32(t)))
        fp.write(lam.astype(np.float32).tobytes())       # K
        fp.write(Phi.tobytes())                           # N*K
        fp.write(residual.tobytes())                      # S*3 (residual colors)
        fp.write(tau.tobytes())                           # S
        fp.write(sharp.tobytes())                         # S
        fp.write(phi_s.tobytes())                         # S*K
        fp.write(fids.astype(np.uint32).tobytes())        # S (face ids)
        fp.write(barys.astype(np.float32).tobytes())      # S*3
        fp.write(c_mean.tobytes())                        # 3
        fp.write(struct.pack("<f", np.float32(bh_sigma))) # 1
    print(f"wrote {out}: K={K} N={N} S={S} t={t}")


if __name__ == "__main__":
    main()
