#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, glob, warnings
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
import vtk

# -------------------- I/O helpers --------------------

def load_nifti_canonical(nii_path: str):
    """
    Load NIfTI, reorient to RAS+.
    Returns (img_can, data_xyz, affine). data_xyz has shape (X, Y, Z).
    """
    img = nib.load(nii_path)
    img_can = nib.as_closest_canonical(img)
    data_xyz = img_can.get_fdata(dtype=np.float32)
    affine = img_can.affine.astype(np.float64)
    ax = nib.aff2axcodes(affine)
    if ax != ('R', 'A', 'S'):
        warnings.warn(f"[warn] not RAS+? got {ax}")
    return img_can, data_xyz, affine

def load_gifti_mesh(gii_path: str):
    gii = nib.load(gii_path)
    verts = faces = None
    for d in gii.darrays:
        if d.intent == nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET']:
            verts = np.asarray(d.data, dtype=np.float64)
        elif d.intent == nib.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE']:
            faces = np.asarray(d.data, dtype=np.int64)
    if verts is None or faces is None:
        raise ValueError(f"POINTSET/TRIANGLE missing in {gii_path}")
    return verts, faces

def world_to_voxel(affine: np.ndarray, xyz_world: np.ndarray) -> np.ndarray:
    """Map world (mm) -> voxel coordinates (i,j,k) aligned with array axes (X,Y,Z)."""
    inv = np.linalg.inv(affine)
    xyz_h = np.c_[xyz_world, np.ones((xyz_world.shape[0], 1))]
    ijk_h = xyz_h @ inv.T
    return ijk_h[:, :3]

def voxel_to_world(affine: np.ndarray, ijk: np.ndarray) -> np.ndarray:
    """Map voxel coords (i,j,k) -> world (mm)."""
    ijk_h = np.c_[ijk, np.ones((ijk.shape[0], 1))]
    xyz_h = ijk_h @ affine.T
    return xyz_h[:, :3]

# -------------------- auto-orient GIfTI vs NIfTI --------------------

def _fraction_inside_volume(verts_vox: np.ndarray, vol_shape_xyz, pad=1.0):
    X, Y, Z = vol_shape_xyz
    x_ok = (verts_vox[:,0] >= -pad) & (verts_vox[:,0] <= X-1+pad)
    y_ok = (verts_vox[:,1] >= -pad) & (verts_vox[:,1] <= Y-1+pad)
    z_ok = (verts_vox[:,2] >= -pad) & (verts_vox[:,2] <= Z-1+pad)
    return np.mean(x_ok & y_ok & z_ok)

def auto_orient_gifti_vertices(verts_world: np.ndarray, affine_img: np.ndarray, vol_shape_xyz):
    """
    Try identity vs LPS->RAS (flip X&Y in world). Pick the one with most vertices inside volume bbox.
    """
    candidates = [
        ("identity", verts_world),
        ("flipXY", np.column_stack((-verts_world[:,0], -verts_world[:,1], verts_world[:,2]))),
    ]
    best_tag, best_V, best_frac = "identity", verts_world, -1.0
    for tag, V in candidates:
        Vvox = world_to_voxel(affine_img, V)
        frac = _fraction_inside_volume(Vvox, vol_shape_xyz, pad=2.0)
        if frac > best_frac:
            best_tag, best_V, best_frac = tag, V, frac
    if best_tag != "identity":
        print(f"[orient] applied {best_tag} to GIfTI vertices (inside={best_frac:.2f})")
    elif best_frac < 0.25:
        warnings.warn(f"[orient] low inside-volume fraction={best_frac:.2f}. Check reference spaces.")
    return best_V

# -------------------- intersection mask (returns shape = (X,Y,Z)) --------------------

def _vtk_polydata(vertices: np.ndarray, faces: np.ndarray):
    pts = vtk.vtkPoints(); pts.SetNumberOfPoints(vertices.shape[0])
    for i, p in enumerate(vertices): pts.SetPoint(i, float(p[0]), float(p[1]), float(p[2]))
    polys = vtk.vtkCellArray()
    for tri in faces:
        cell = vtk.vtkTriangle()
        cell.GetPointIds().SetId(0, int(tri[0])); cell.GetPointIds().SetId(1, int(tri[1])); cell.GetPointIds().SetId(2, int(tri[2]))
        polys.InsertNextCell(cell)
    poly = vtk.vtkPolyData(); poly.SetPoints(pts); poly.SetPolys(polys); poly.BuildCells(); poly.BuildLinks()
    return poly

def intersect_mask_from_surface_voxel_XYZ(vertices_vox: np.ndarray,
                                          faces: np.ndarray,
                                          vol_shape_xyz: tuple,
                                          margin_vox: int = 1,
                                          dist_thresh_vox: float = 0.9) -> np.ndarray:
    """
    Boolean mask (X,Y,Z) of voxels whose centers are within dist_thresh_vox of the surface.
    """
    X, Y, Z = vol_shape_xyz
    poly = _vtk_polydata(vertices_vox, faces)
    imp = vtk.vtkImplicitPolyDataDistance(); imp.SetInput(poly)

    i0, j0, k0 = np.maximum(np.floor(vertices_vox.min(0) - margin_vox), 0).astype(int)
    i1, j1, k1 = np.minimum(np.ceil(vertices_vox.max(0) + margin_vox), np.array([X-1, Y-1, Z-1])).astype(int)

    mask_xyz = np.zeros((X, Y, Z), dtype=bool)
    ii = np.arange(i0, i1 + 1, dtype=float) + 0.5
    jj = np.arange(j0, j1 + 1, dtype=float) + 0.5
    kk = np.arange(k0, k1 + 1, dtype=float) + 0.5

    for k in kk:
        I, J = np.meshgrid(ii, jj, indexing='xy')  # I: X, J: Y
        P = np.c_[I.ravel(), J.ravel(), np.full(I.size, k)]
        dists = np.array([imp.EvaluateFunction(p.tolist()) for p in P], dtype=np.float32).reshape(J.shape)
        hits = (np.abs(dists) <= dist_thresh_vox)  # shape (Y,X)
        k_idx = int(round(k - 0.5))
        mask_xyz[i0:i1+1, j0:j1+1, k_idx] = hits.T  # transpose to (X,Y)

    return mask_xyz

# -------------------- best-fit plane (from HIPPO mask) --------------------

def best_fit_plane_from_mask_XYZ(mask_xyz: np.ndarray, affine: np.ndarray):
    """
    Fit plane in world coords from mask (X,Y,Z). Returns (origin_world, normal_world, u_world, v_world).
    """
    idx = np.argwhere(mask_xyz)  # columns are (x,y,z)
    if idx.size == 0:
        raise ValueError("Mask empty; no intersecting voxels.")
    ijk = idx.astype(float) + 0.5
    xyz = voxel_to_world(affine, ijk)
    ctr = xyz.mean(0)
    Xc = xyz - ctr
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    n = Vt[-1]; n /= (np.linalg.norm(n) + 1e-12)
    u = Vt[0];  u /= (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u); v /= (np.linalg.norm(v) + 1e-12)
    return ctr, n, u, v

# -------------------- reslice along plane (data (X,Y,Z)) --------------------

def reslice_along_plane_XYZ(data_xyz: np.ndarray,
                            affine: np.ndarray,
                            origin_world: np.ndarray,
                            u_world: np.ndarray,
                            v_world: np.ndarray,
                            out_shape=(512, 512),
                            field_of_view_mm=None,
                            order=1):
    """
    Sample data on a plane spanned by (u,v) at origin (all world coords).
    Returns slice_img (nv, nu) and plane_world_pts (nv, nu, 3).
    """
    X, Y, Z = data_xyz.shape
    # Auto field-of-view from volume corners if not provided
    if field_of_view_mm is None:
        corners_ijk = np.array([[0,   0,   0],
                                [X-1, 0,   0],
                                [0,   Y-1, 0],
                                [0,   0,   Z-1],
                                [X-1, Y-1, 0],
                                [X-1, 0,   Z-1],
                                [0,   Y-1, Z-1],
                                [X-1, Y-1, Z-1]], dtype=float)
        corners_world = voxel_to_world(affine, corners_ijk)
        rel = corners_world - origin_world[None, :]
        proj_u = rel @ u_world
        proj_v = rel @ v_world
        field_of_view_mm = (0.8 * (proj_u.max() - proj_u.min()),
                            0.8 * (proj_v.max() - proj_v.min()))

    nu, nv = out_shape
    hu, hv = field_of_view_mm[0] * 0.5, field_of_view_mm[1] * 0.5
    u_lin = np.linspace(-hu, hu, nu, dtype=np.float64)
    v_lin = np.linspace(-hv, hv, nv, dtype=np.float64)
    UU, VV = np.meshgrid(u_lin, v_lin, indexing='xy')
    plane_world = origin_world + UU[..., None]*u_world + VV[..., None]*v_world

    # Map world -> voxel (i,j,k) which align with array axes (X,Y,Z)
    ijk = world_to_voxel(affine, plane_world.reshape(-1, 3))
    # map_coordinates expects coords in axis order (X, Y, Z)
    coords_xyz = np.stack([ijk[:, 0], ijk[:, 1], ijk[:, 2]], axis=0)
    slice_img = map_coordinates(data_xyz, coords_xyz, order=order, mode='nearest').reshape(nv, nu)
    return slice_img, plane_world

def sample_mask_on_plane_XYZ(mask_xyz: np.ndarray, affine: np.ndarray, plane_world: np.ndarray) -> np.ndarray:
    ijk = world_to_voxel(affine, plane_world.reshape(-1, 3))
    ijk_round = np.rint(ijk).astype(int)
    X, Y, Z = mask_xyz.shape
    ijk_round[:, 0] = np.clip(ijk_round[:, 0], 0, X - 1)
    ijk_round[:, 1] = np.clip(ijk_round[:, 1], 0, Y - 1)
    ijk_round[:, 2] = np.clip(ijk_round[:, 2], 0, Z - 1)
    vals = mask_xyz[ijk_round[:, 0], ijk_round[:, 1], ijk_round[:, 2]]
    return vals.reshape(plane_world.shape[:2])

# -------------------- plotting (95% intensity range) --------------------

def plot_slice_two_overlays(slice_img: np.ndarray,
                            mask_hipp: np.ndarray,
                            mask_dent: np.ndarray,
                            out_path: str,
                            title: str,
                            alpha: float = 0.6,
                            underlay_from_volume: np.ndarray = None):
    """
    Plot with robust windowing: vmin/vmax = 2.5th/97.5th percentiles of the *volume* (95% range).
    """
    if underlay_from_volume is not None:
        vmin, vmax = np.percentile(underlay_from_volume[np.isfinite(underlay_from_volume)], [2.5, 97.5])
    else:
        vmin, vmax = np.percentile(slice_img[np.isfinite(slice_img)], [2.5, 97.5])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(slice_img, origin='lower', cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')

    # Yellow HIPPO
    hipp_rgba = np.zeros(mask_hipp.shape + (4,), dtype=float)
    hipp_rgba[..., 0:2] = 1.0
    hipp_rgba[..., 3] = mask_hipp.astype(float) * alpha
    ax.imshow(hipp_rgba, origin='lower', interpolation='nearest')

    # Purple DENTATE
    dent_rgba = np.zeros(mask_dent.shape + (4,), dtype=float)
    dent_rgba[..., 0] = 1.0
    dent_rgba[..., 2] = 1.0
    dent_rgba[..., 3] = mask_dent.astype(float) * alpha
    ax.imshow(dent_rgba, origin='lower', interpolation='nearest')

    ax.set_title(title); ax.set_axis_off()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

# -------------------- discovery (T1w/T2w; crop variants) --------------------

def find_nifti_for_sub(sub_dir: Path):
    anat = sub_dir / "anat"
    candidates = []
    for sp in ("cropT1w", "cropT2w"):
        for mod in ("T1w", "T2w"):
            candidates += glob.glob(str(anat / f"{sub_dir.name}_hemi-*_space-{sp}_desc-preproc_{mod}.nii.gz"))
    if not candidates:
        raise FileNotFoundError(f"No preproc NIfTI found under {anat}")
    nii = Path(sorted(candidates)[0])
    space_token = "T1w" if "space-cropT1w" in nii.name else "T2w"
    return nii, space_token

def expected_surface_paths(sub_dir: Path, hemi: str, den: str, space_token: str):
    surf = sub_dir / "surf"
    sub = sub_dir.name
    hipp = surf / f"{sub}_hemi-{hemi}_space-{space_token}_den-{den}_label-hipp_midthickness.surf.gii"
    dent = surf / f"{sub}_hemi-{hemi}_space-{space_token}_den-{den}_label-dentate_midthickness.surf.gii"
    return hipp, dent

def get_cond_dir(root: Path, cond: str) -> Path:
    """
    Return the condition directory, handling v1 roots that have an extra 'hippunfold/' layer.
    Prefers the v1-style path if the root looks like v1 OR if that path exists.
    Falls back gracefully if only one exists.
    """
    base = root / cond
    v1_candidate = base / "hippunfold"

    root_has_v1 = "v1" in str(root).lower()

    if root_has_v1 and v1_candidate.exists():
        return v1_candidate
    if root_has_v1 and not v1_candidate.exists():
        return base  # user said it's v1 but folder doesn't exist—fall back

    # Not explicitly v1: prefer the path that actually exists
    if v1_candidate.exists() and not base.exists():
        return v1_candidate
    return base

# -------------------- per-case: plane slice --------------------

def process_case_plane_slice(nii_path: Path, gii_hipp_path: Path, gii_dent_path: Path,
                             out_png: Path, dist_thresh_vox=0.9, margin_vox=1, slice_size=(512, 512)):
    # 1) canonical volume (X,Y,Z)
    _, data_xyz, affine = load_nifti_canonical(str(nii_path))

    # 2) load surfaces, auto-orient, map to voxel
    v_hipp_w, f_hipp = load_gifti_mesh(str(gii_hipp_path))
    v_dent_w, f_dent = load_gifti_mesh(str(gii_dent_path))
    v_hipp_w = auto_orient_gifti_vertices(v_hipp_w, affine, data_xyz.shape)
    v_dent_w = auto_orient_gifti_vertices(v_dent_w, affine, data_xyz.shape)
    v_hipp_vox = world_to_voxel(affine, v_hipp_w)
    v_dent_vox = world_to_voxel(affine, v_dent_w)

    # 3) masks on same grid (X,Y,Z)
    mask_hipp = intersect_mask_from_surface_voxel_XYZ(v_hipp_vox, f_hipp, data_xyz.shape,
                                                      margin_vox=margin_vox, dist_thresh_vox=dist_thresh_vox)
    mask_dent = intersect_mask_from_surface_voxel_XYZ(v_dent_vox, f_dent, data_xyz.shape,
                                                      margin_vox=margin_vox, dist_thresh_vox=dist_thresh_vox)

    # 4) best-fit plane from HIPPO mask (world)
    origin_w, _, u_w, v_w = best_fit_plane_from_mask_XYZ(mask_hipp, affine)

    # 5) reslice once
    slice_img, plane_world = reslice_along_plane_XYZ(data_xyz, affine, origin_w, u_w, v_w,
                                                     out_shape=slice_size, field_of_view_mm=None, order=1)

    # 6) sample both overlays on that plane
    ov_hipp = sample_mask_on_plane_XYZ(mask_hipp, affine, plane_world)
    ov_dent = sample_mask_on_plane_XYZ(mask_dent, affine, plane_world)

    # 7) plot with robust 95% windowing (percentiles from full volume)
    title = f"{gii_hipp_path.name} + dentate on {nii_path.name}"
    plot_slice_two_overlays(slice_img, ov_hipp, ov_dent, str(out_png), title,
                            underlay_from_volume=data_xyz)

# -------------------- CLI loop (usage unchanged; flat '{cond}_...' filenames) --------------------

def main():
    ap = argparse.ArgumentParser(description="Plane slice overlays in canonical RAS+ with 95% intensity window.")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hemi", default="L", choices=["L", "R"])
    ap.add_argument("--den", default="8k", choices=["8k", "0p5mm"])
    ap.add_argument("--dist-thresh-vox", type=float, default=0.5)
    ap.add_argument("--margin-vox", type=int, default=1)
    ap.add_argument("--slice-size", type=int, nargs=2, default=[512, 512])
    args = ap.parse_args()

    conditions = ["highresMRI", "lowresMRI", "thickSlice", "atrophy", "neonate"]
    root = Path(args.root)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    for cond in conditions:
        cond_dir = get_cond_dir(root, cond)
        subs = sorted(glob.glob(str(cond_dir / "sub-*")))
        if not subs:
            warnings.warn(f"[{cond}] no subjects under {cond_dir}")
            continue
        sub_dir = Path(subs[0])
        try:
            nii, space_token = find_nifti_for_sub(sub_dir)
            gii_hipp, gii_dent = expected_surface_paths(sub_dir, args.hemi, args.den, space_token)
            missing = [p for p in (nii, gii_hipp, gii_dent) if not Path(p).exists()]
            if missing:
                warnings.warn(f"[{cond}] missing:\n  " + "\n  ".join(map(str, missing)))
                continue

            # Flat filename with {cond}
            out_png = outdir / f"{cond}_{sub_dir.name}_hemi-{args.hemi}_space-{space_token}_den-{args.den}_hipp-dent_planeslice.png"
            print(f"[INFO] {cond}: {sub_dir.name} ({space_token}) -> {out_png}")
            process_case_plane_slice(Path(nii), Path(gii_hipp), Path(gii_dent), out_png,
                                     dist_thresh_vox=args.dist_thresh_vox,
                                     margin_vox=args.margin_vox,
                                     slice_size=tuple(args.slice_size))
        except Exception as e:
            warnings.warn(f"[{cond}] failed: {e}")

    print("[DONE]")

if __name__ == "__main__":
    main()
