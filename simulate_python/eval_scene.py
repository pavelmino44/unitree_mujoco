"""MuJoCo scenes of the evaluation scenarios of unitree_rl_lab (eval_scenarios.py).

The terrains are exported from IsaacLab by unitree_rl_lab/scripts/rsl_rl/export_mujoco_scenarios.py
(catalog.json + terrains/<hash>.npz: the raw triangles of the evaluation tile, in world coordinates).
This module turns such a tile into MuJoCo geoms that have the same surface:

  * terrains made of axis-aligned blocks (stairs, inverted stairs, random boxes, flat ground) -> box geoms.
    The tile is cut along every x and y coordinate of its vertices; in every cell the surface is flat, so
    one box per cell (merged into larger rectangles) reproduces it exactly, edges included. MuJoCo meshes
    cannot be used for this: MuJoCo collides with the convex hull of a mesh.
  * any other terrain (slopes, random rough ground) -> a height field sampled from the IsaacLab triangles
    on a fine grid (default 2 cm).
  * the flat border around the tile (TerrainGeneratorCfg.border_width, top at z = 0), as in IsaacLab.

Coordinates are the IsaacLab world coordinates of the evaluation (the tile is centred on the origin), so
positions, the spawn point and the goal line mean the same thing in both simulators.

Command line (inspection; the evaluation uses it as a module):

    python eval_scene.py ../eval_scenarios/catalog.json rough_pyramid_stairs_d100 --seed 42 --view
    python eval_scene.py ../eval_scenarios/catalog.json all --check        # build + verify every scene
    python eval_scene.py ../eval_scenarios/catalog.json rough_boxes_d50 --save ../unitree_robots/go2/eval_boxes.xml
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROBOTS_DIR = os.path.normpath(os.path.join(HERE, "..", "unitree_robots"))

HFIELD_RESOLUTION = 0.02  # m, grid of the height-field terrains
HFIELD_BASE = 0.5  # m, thickness of the height field below its lowest point
COLUMN_DEPTH = 1.0  # m, the box columns reach this far below the lowest vertex of the tile
AXIS_TOL = 1e-6  # tolerance of the "axis-aligned triangles" test
COORD_DECIMALS = 6  # vertex coordinates are rounded to 1 um before cutting the tile into cells

TERRAIN_RGBA = ((0.58, 0.66, 0.78), (0.30, 0.40, 0.58))  # low / high colour of the tile boxes
BORDER_RGBA = "0.42 0.50 0.62 1"


##
# Terrain data
##


def load_terrain(path: str) -> dict:
    """Read a terrain file written by export_mujoco_scenarios.py."""
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    return {
        "vertices": np.asarray(data["vertices"], dtype=np.float64),
        "faces": np.asarray(data["faces"], dtype=np.int64),
        "origin": np.asarray(data["origin"], dtype=np.float64),
        "tile_size": np.asarray(data["tile_size"], dtype=np.float64),
        "border_width": float(data["border_width"]),
        "border_height": float(data["border_height"]),
        "sub_terrain": str(data["sub_terrain"]),
        "difficulty": float(data["difficulty"]),
        "path": os.path.abspath(path),
    }


def load_catalog(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        catalog = json.load(f)
    catalog["_dir"] = os.path.dirname(os.path.abspath(path))
    return catalog


def terrain_path(catalog: dict, name: str, seed: int) -> str:
    sc = catalog["scenarios"].get(name)
    if sc is None:
        raise KeyError(f"scenario '{name}' is not in the catalog")
    rel = sc["terrains"].get(str(seed))
    if rel is None:
        raise KeyError(
            f"scenario '{name}' has no terrain for seed {seed} (exported seeds: {sorted(sc['terrains'], key=int)}). "
            "Export it: python scripts/rsl_rl/export_mujoco_scenarios.py --headless --append --eval_seed ..."
        )
    return os.path.join(catalog["_dir"], rel)


##
# Geometry
##


def _up_triangles(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """(n, 3, 3) triangles whose projection on the xy-plane has a non-zero area (any orientation)."""
    tri = vertices[faces]
    e1, e2 = tri[:, 1, :2] - tri[:, 0, :2], tri[:, 2, :2] - tri[:, 0, :2]
    area2 = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    return tri[np.abs(area2) > 1e-12]


def rasterize_top(vertices: np.ndarray, faces: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Highest surface point above every grid point (xs[i], ys[j]) -> (len(xs), len(ys)); nan if none.

    z-buffer over the triangles (vertical triangles are skipped: their top edge belongs to a neighbour).
    xs and ys must be sorted.
    """
    out = np.full((len(xs), len(ys)), -np.inf)
    tri = _up_triangles(vertices, faces)
    eps = 1e-9
    for t in tri:
        (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = t
        i0, i1 = np.searchsorted(xs, min(x0, x1, x2) - eps), np.searchsorted(xs, max(x0, x1, x2) + eps, "right")
        j0, j1 = np.searchsorted(ys, min(y0, y1, y2) - eps), np.searchsorted(ys, max(y0, y1, y2) + eps, "right")
        if i0 >= i1 or j0 >= j1:
            continue
        px, py = np.meshgrid(xs[i0:i1], ys[j0:j1], indexing="ij")
        det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        a = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / det
        b = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / det
        c = 1.0 - a - b
        inside = (a >= -1e-9) & (b >= -1e-9) & (c >= -1e-9)
        if not inside.any():
            continue
        z = a * z0 + b * z1 + c * z2
        block = out[i0:i1, j0:j1]
        np.maximum(block, np.where(inside, z, -np.inf), out=block)
    out[~np.isfinite(out)] = np.nan
    return out


def is_axis_aligned(vertices: np.ndarray, faces: np.ndarray) -> bool:
    """True if every triangle is parallel to one of the coordinate planes (a terrain of blocks)."""
    tri = vertices[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(n, axis=1)
    n = n[norm > 1e-12] / norm[norm > 1e-12, None]
    return bool(np.all(np.max(np.abs(n), axis=1) > 1.0 - AXIS_TOL))


def _rectangles(height: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Greedy split of a grid of cell heights (nan = empty) into rectangles of equal height: (i0, i1, j0, j1)."""
    nx, ny = height.shape
    used = ~np.isfinite(height)
    rects = []
    for i in range(nx):
        for j in range(ny):
            if used[i, j]:
                continue
            h = height[i, j]
            j1 = j + 1
            while j1 < ny and not used[i, j1] and height[i, j1] == h:
                j1 += 1
            i1 = i + 1
            while i1 < nx and not used[i1, j:j1].any() and np.all(height[i1, j:j1] == h):
                i1 += 1
            used[i:i1, j:j1] = True
            rects.append((i, i1, j, j1))
    return rects


def column_boxes(vertices: np.ndarray, faces: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Exact box decomposition of a terrain of axis-aligned blocks: [(centre (3,), half size (3,))]."""
    v = np.round(vertices, COORD_DECIMALS)
    xs, ys = np.unique(v[:, 0]), np.unique(v[:, 1])
    cx, cy = 0.5 * (xs[1:] + xs[:-1]), 0.5 * (ys[1:] + ys[:-1])
    top = np.round(rasterize_top(v, faces, cx, cy), COORD_DECIMALS)
    bottom = float(v[:, 2].min()) - COLUMN_DEPTH
    boxes = []
    for i0, i1, j0, j1 in _rectangles(top):
        h = float(top[i0, j0])
        if h - bottom < 1e-6:
            continue
        lo = np.array([xs[i0], ys[j0], bottom])
        hi = np.array([xs[i1], ys[j1], h])
        boxes.append((0.5 * (lo + hi), 0.5 * (hi - lo)))
    return boxes


def height_grid(vertices: np.ndarray, faces: np.ndarray, resolution: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Heights of the surface on a regular grid covering the tile: (xs, ys, heights (len(xs), len(ys)))."""
    lo, hi = vertices[:, :2].min(axis=0), vertices[:, :2].max(axis=0)
    nx = int(round((hi[0] - lo[0]) / resolution)) + 1
    ny = int(round((hi[1] - lo[1]) / resolution)) + 1
    xs, ys = np.linspace(lo[0], hi[0], nx), np.linspace(lo[1], hi[1], ny)
    h = rasterize_top(vertices, faces, xs, ys)
    if np.isnan(h).any():  # holes (should not happen): fill with the lowest point
        h[np.isnan(h)] = np.nanmin(h)
    return xs, ys, h


def surface_height(vertices: np.ndarray, faces: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Height of the IsaacLab surface at arbitrary points (n, 2) (nan where there is none)."""
    out = np.full(len(xy), -np.inf)
    px, py = xy[:, 0], xy[:, 1]
    for t in _up_triangles(vertices, faces):
        (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = t
        sel = (px >= min(x0, x1, x2) - 1e-9) & (px <= max(x0, x1, x2) + 1e-9)
        sel &= (py >= min(y0, y1, y2) - 1e-9) & (py <= max(y0, y1, y2) + 1e-9)
        if not sel.any():
            continue
        det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        a = ((y1 - y2) * (px[sel] - x2) + (x2 - x1) * (py[sel] - y2)) / det
        b = ((y2 - y0) * (px[sel] - x2) + (x0 - x2) * (py[sel] - y2)) / det
        c = 1.0 - a - b
        inside = (a >= -1e-9) & (b >= -1e-9) & (c >= -1e-9)
        z = np.where(inside, a * z0 + b * z1 + c * z2, -np.inf)
        out[sel] = np.maximum(out[sel], z)
    out[~np.isfinite(out)] = np.nan
    return out


##
# MJCF
##


def _f(values) -> str:
    return " ".join(f"{float(v):.6g}" for v in np.ravel(values))


def _height_rgba(h: float, lo: float, hi: float) -> str:
    t = 0.5 if hi - lo < 1e-6 else (h - lo) / (hi - lo)
    c = np.array(TERRAIN_RGBA[0]) * (1 - t) + np.array(TERRAIN_RGBA[1]) * t
    return f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} 1"


def terrain_mjcf(terrain: dict, hfield_resolution: float = HFIELD_RESOLUTION) -> tuple[str, str, dict]:
    """(asset xml, worldbody xml, info) of the tile and its border."""
    v, f = terrain["vertices"], terrain["faces"]
    sx, sy = terrain["tile_size"]
    asset, body = [], []
    info = {"sub_terrain": terrain["sub_terrain"], "difficulty": terrain["difficulty"]}
    if is_axis_aligned(v, f):
        boxes = column_boxes(v, f)
        tops = [c[2] + s[2] for c, s in boxes]
        lo, hi = (min(tops), max(tops)) if tops else (0.0, 0.0)
        for k, (c, s) in enumerate(boxes):
            body.append(
                f'<geom name="terrain_{k}" type="box" pos="{_f(c)}" size="{_f(s)}" '
                f'rgba="{_height_rgba(c[2] + s[2], lo, hi)}" group="0"/>'
            )
        info.update(kind="boxes", geoms=len(boxes), top_min=lo, top_max=hi)
    else:
        xs, ys, h = height_grid(v, f, hfield_resolution)
        zmin, zmax = float(h.min()), float(h.max())
        span = max(zmax - zmin, 1e-6)
        data = (h - zmin) / span  # MuJoCo uses elevation data in [0, 1], scaled by size[2]
        # MuJoCo elevation data: nrow rows along y, ncol columns along x, row-major, the FIRST row at +y
        elev = data.T[::-1]
        rx, ry = 0.5 * (xs[-1] - xs[0]), 0.5 * (ys[-1] - ys[0])
        cx, cy = 0.5 * (xs[-1] + xs[0]), 0.5 * (ys[-1] + ys[0])
        asset.append(
            f'<hfield name="terrain" nrow="{len(ys)}" ncol="{len(xs)}" size="{_f([rx, ry, span, HFIELD_BASE])}" '
            f'elevation="{_f(elev)}"/>'
        )
        body.append(
            f'<geom name="terrain" type="hfield" hfield="terrain" pos="{_f([cx, cy, zmin])}" material="terrain_hf" group="0"/>'
        )
        info.update(kind="hfield", grid=[len(xs), len(ys)], resolution=hfield_resolution, top_min=zmin, top_max=zmax)

    # flat border around the tile, top at z = 0 (TerrainGenerator._add_terrain_border)
    w, bh = terrain["border_width"], terrain["border_height"]
    if w > 0:
        hx, hy, hz = 0.5 * sx, 0.5 * sy, 0.5 * bh
        for k, (c, s) in enumerate(
            (
                ((hx + w / 2, 0.0), (w / 2, hy + w)),
                ((-hx - w / 2, 0.0), (w / 2, hy + w)),
                ((0.0, hy + w / 2), (hx, w / 2)),
                ((0.0, -hy - w / 2), (hx, w / 2)),
            )
        ):
            body.append(
                f'<geom name="border_{k}" type="box" pos="{_f([c[0], c[1], -hz])}" size="{_f([s[0], s[1], hz])}" '
                f'material="border" group="0"/>'
            )
    return "\n    ".join(asset), "\n    ".join(body), info


def scene_xml(
    terrain: dict,
    robot: str = "go2",
    markers: list[dict] | None = None,
    hfield_resolution: float = HFIELD_RESOLUTION,
) -> tuple[str, dict]:
    """Complete scene (robot + terrain + visual markers). markers: [{"from": [x, y], "to": [x, y], "rgba": "r g b a"}]
    (thin visual-only strips on the ground, e.g. the start and the goal line)."""
    asset, body, info = terrain_mjcf(terrain, hfield_resolution)
    marks = []
    for k, m in enumerate(markers or []):
        (x0, y0), (x1, y1) = m["from"], m["to"]
        length = float(np.hypot(x1 - x0, y1 - y0))
        yaw = float(np.arctan2(y1 - y0, x1 - x0))
        z = float(m.get("z", 0.002))
        marks.append(
            f'<geom name="marker_{k}" type="box" pos="{_f([(x0 + x1) / 2, (y0 + y1) / 2, z])}" '
            f'size="{_f([length / 2, m.get("width", 0.02) / 2, 0.001])}" euler="0 0 {yaw:.6g}" '
            f'rgba="{m.get("rgba", "1 0.6 0 0.9")}" contype="0" conaffinity="0" group="1"/>'
        )
    xml = f"""<mujoco model="{robot} evaluation scene">
  <include file="{robot}.xml"/>

  <statistic center="0 0 0.1" extent="2.0"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-130" elevation="-20" offwidth="1280" offheight="720"/>
    <map zfar="200"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="grid" builtin="checker" mark="edge" rgb1="0.42 0.50 0.62" rgb2="0.38 0.46 0.58"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="border" texture="grid" texuniform="true" texrepeat="1 1" reflectance="0.1"/>
    <material name="terrain_hf" texture="grid" texuniform="true" texrepeat="4 4" rgba="0.75 0.85 1 1"/>
    {asset}
  </asset>

  <worldbody>
    <light pos="0 0 6" dir="0 0 -1" directional="true"/>
    {body}
    {chr(10).join(marks)}
  </worldbody>
</mujoco>
"""
    return xml, info


def load_model(xml: str, robot: str = "go2", save_as: str | None = None):
    """Compile the scene. The file is written next to <robot>.xml (the mesh paths of the robot are relative to
    it) and removed again unless save_as is given."""
    import mujoco

    robot_dir = os.path.join(ROBOTS_DIR, robot)
    if save_as:
        path = os.path.abspath(save_as)
        if os.path.dirname(path) != os.path.abspath(robot_dir):
            print(f"[WARN] {path} is not in {robot_dir}: the robot meshes will not be found when it is opened.")
        with open(path, "w") as f:
            f.write(xml)
        return mujoco.MjModel.from_xml_path(path)
    fd, path = tempfile.mkstemp(prefix=".eval_scene_", suffix=".xml", dir=robot_dir)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(xml)
        return mujoco.MjModel.from_xml_path(path)
    finally:
        os.remove(path)


def check_model(model, terrain: dict, samples: int = 400, seed: int = 0) -> dict:
    """Compare the MuJoCo surface (vertical rays) with the IsaacLab triangles at random points of the tile.

    Points closer than 2 cm to a step edge of a block terrain are skipped (the surface jumps there).
    """
    import mujoco

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    rng = np.random.default_rng(seed)
    sx, sy = terrain["tile_size"]
    v, f = terrain["vertices"], terrain["faces"]
    xy = rng.uniform([-sx / 2 + 0.05, -sy / 2 + 0.05], [sx / 2 - 0.05, sy / 2 - 0.05], size=(samples, 2))
    if is_axis_aligned(v, f):
        edges_x, edges_y = np.unique(np.round(v[:, 0], 6)), np.unique(np.round(v[:, 1], 6))
        far = (np.min(np.abs(xy[:, :1] - edges_x[None]), axis=1) > 0.02) & (np.min(np.abs(xy[:, 1:] - edges_y[None]), axis=1) > 0.02)
        xy = xy[far]
    expected = surface_height(v, f, xy)
    geomgroup = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)  # terrain only
    got = np.empty(len(xy))
    geomid = np.zeros(1, dtype=np.int32)
    for k, (x, y) in enumerate(xy):
        dist = mujoco.mj_ray(model, data, np.array([x, y, 10.0]), np.array([0.0, 0.0, -1.0]), geomgroup, 1, -1, geomid)
        got[k] = 10.0 - dist if dist >= 0 else np.nan
    err = np.abs(got - expected)
    return {"points": int(len(xy)), "max_err_m": float(np.nanmax(err)) if len(err) else 0.0,
            "mean_err_m": float(np.nanmean(err)) if len(err) else 0.0, "missed": int(np.isnan(got).sum())}


##
# Command line
##


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build / inspect the MuJoCo scenes of the evaluation scenarios.")
    ap.add_argument("catalog", help="catalog.json written by export_mujoco_scenarios.py (or a terrain .npz file)")
    ap.add_argument("scenarios", nargs="?", default="all", help="names / glob patterns, comma-separated (default: all)")
    ap.add_argument("--seed", type=int, default=None, help="terrain seed (default: the first exported one)")
    ap.add_argument("--robot", default="go2")
    ap.add_argument("--resolution", type=float, default=HFIELD_RESOLUTION, help="height-field grid, m")
    ap.add_argument("--check", action="store_true", help="compare the MuJoCo surface with the IsaacLab triangles")
    ap.add_argument("--view", action="store_true", help="open the MuJoCo viewer (first selected scenario)")
    ap.add_argument("--save", default=None, help="write the scene XML to this path (first selected scenario)")
    args = ap.parse_args(argv)

    jobs = []
    if args.catalog.endswith(".npz"):
        jobs.append((os.path.basename(args.catalog), load_terrain(args.catalog)))
    else:
        catalog = load_catalog(args.catalog)
        names = []
        for tok in [t.strip() for t in args.scenarios.split(",") if t.strip()]:
            pat = "*" if tok == "all" else ("rough_*" if tok == "rough" else tok)
            names += [n for n in catalog["scenarios"] if fnmatch.fnmatchcase(n, pat) and n not in names]
        if not names:
            sys.exit(f"[ERROR] no scenario matches '{args.scenarios}'")
        for n in names:
            seeds = sorted(catalog["scenarios"][n]["terrains"], key=int)
            seed = args.seed if args.seed is not None else int(seeds[0])
            jobs.append((f"{n} (seed {seed})", load_terrain(terrain_path(catalog, n, seed))))

    worst = 0.0
    for k, (label, terrain) in enumerate(jobs):
        xml, info = scene_xml(terrain, args.robot, hfield_resolution=args.resolution)
        model = load_model(xml, args.robot, save_as=args.save if k == 0 else None)
        line = f"{label}: {info['sub_terrain']} d={info['difficulty']:.3f} -> {info['kind']}"
        line += f" ({info['geoms']} boxes)" if info["kind"] == "boxes" else f" (grid {info['grid'][0]}x{info['grid'][1]})"
        line += f", top {info['top_min']:+.3f}..{info['top_max']:+.3f} m"
        if args.check:
            res = check_model(model, terrain)
            worst = max(worst, res["max_err_m"])
            line += f" | check: {res['points']} pts, max err {res['max_err_m'] * 1000:.2f} mm, mean {res['mean_err_m'] * 1000:.2f} mm"
            if res["missed"]:
                line += f", {res['missed']} rays missed"
        print(line, flush=True)
        if args.view and k == 0:
            import mujoco
            import mujoco.viewer

            data = mujoco.MjData(model)
            mujoco.mj_resetData(model, data)
            mujoco.viewer.launch(model, data)
    if args.check:
        print(f"worst height error over all scenes: {worst * 1000:.2f} mm")


if __name__ == "__main__":
    main()
