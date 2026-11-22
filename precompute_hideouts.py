"""Offline: score every candidate spot on the map by how exposed it is.  Writes hideouts.json.

exposure(spot) = number of open-ground observer points within 200 m (16 m grid, eye 8 m above
terrain) that have line of sight to the top of a mech standing at the spot (vision rule: centre OR
top visible -> seen; top is the easier one).  Buildings and terrain block; trees ignored.
Candidates: walkable ground every 8 m, and roofs 10-45 m tall every 8 m.
Run: python3 precompute_hideouts.py /path/to/arena
"""
import json, math, os, sys, time, types
from multiprocessing import Pool

stub = types.ModuleType("mechfight"); stub.log = lambda *a: None
class _D:
    def __getattr__(self, n): return lambda *a, **k: None
stub.draw = _D(); stub.run = lambda c: None
sys.modules["mechfight"] = stub
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location("botmod", os.path.join(os.path.dirname(os.path.abspath(__file__)), "bot.py"))
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)

ARENA = sys.argv[1] if len(sys.argv) > 1 else "/arena"
W = None
OBS = []


def los(a, b, step=3.0):
    dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    n = max(1, int(math.sqrt(dx * dx + dy * dy + dz * dz) / step))
    for i in range(1, n):
        f = i / n
        px, py, z = a[0] + dx * f, a[1] + dy * f, a[2] + dz * f
        if z < W.ground(px, py) - 0.2 or W.hit(px, py, z, z):
            return False
    return True


def init():
    global W, OBS
    W = B.World(ARENA, 1, budget_s=60.0)
    OBS = []
    y = W.miny + 8
    while y < W.miny + W.ny * W.step - 8:
        x = W.minx + 8
        while x < W.minx + W.nx * W.step - 8:
            if W.free(W.cell(x, y)):
                OBS.append((x, y, W.ground(x, y) + 8.0))
            x += 20.0
        y += 20.0


def score(c):
    x, y, z, roof = c
    top = (x, y, z + 9.0)
    cen = (x, y, z + 4.5)
    seen = near = shot = 0
    for ox, oy, oz in OBS:
        d = math.hypot(ox - x, oy - y)
        if d > 200 or d < 4:
            continue
        o = (ox, oy, oz)
        if los(o, top):
            seen += 1
            if seen > 80:
                break
            near += d < 60
            shot += los(o, cen)
    return [round(x, 1), round(y, 1), round(z, 2), seen, near, shot, roof]


def candidates():
    cs = []
    y = W.miny + 6
    while y < W.miny + W.ny * W.step - 6:
        x = W.minx + 6
        while x < W.minx + W.nx * W.step - 6:
            if W.free(W.cell(x, y)):
                cs.append((x, y, W.ground(x, y), 0))
            x += 10.0
        y += 10.0
    for idx, (base, top, ring, holes, x0, y0, x1, y1, kind) in enumerate(W.prisms):
        if kind != "building":
            continue
        g = W.ground((x0 + x1) / 2, (y0 + y1) / 2)
        if not (10.0 <= top - g <= 45.0):
            continue
        y = y0 + 4
        while y < y1 - 3:
            x = x0 + 4
            while x < x1 - 3:
                if B._pip(x, y, ring) and not any(B._pip(x, y, h) for h in holes):
                    # must be the top surface (nothing solid above), and at least 3 m from the roof edge
                    if not W.hit(x, y, top + 0.5, top + 9.0) and all(
                            B._pip(x + 3 * math.cos(a), y + 3 * math.sin(a), ring) for a in (0, 1.57, 3.14, 4.71)):
                        cs.append((x, y, top, idx + 1))
                x += 8.0
            y += 8.0
    return cs


if __name__ == "__main__":
    t0 = time.time()
    init()
    cs = candidates()
    print(f"observers {len(OBS)}  candidates {len(cs)} (roof {sum(1 for c in cs if c[3])})", flush=True)
    with Pool(os.cpu_count() or 2, initializer=init) as pool:
        res = pool.map(score, cs, chunksize=64)
    res.sort(key=lambda r: (r[3], r[4]))
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hideouts.json"), "w"), separators=(",", ":"))
    print(f"done in {time.time()-t0:.0f}s; best 10:", res[:10])
