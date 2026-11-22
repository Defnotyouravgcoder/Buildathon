
"""Survivor bot v2 for mechfight/1  (standard library only).

Placement is the score, so: stay in the safe region, stay quiet, avoid groups, duel lone hunters,
rank targets with targeting.score(), predict shots with velocity + 0.5*accel, keep every decision
well under the 50 ms budget (planner and searches are time-sliced against a soft 30 ms deadline).
"""
import heapq
import json
import math
import os
import random
import sys
import time
from array import array

import mechfight as mf
import targeting as T

TAU = 2 * math.pi
NOISE_FREE = 0.40        # walk fraction that stays under sound.step_speed (11.7*0.4 = 4.7 m/s)
REGION_LEAD = 30.0       # head for the next circle this many seconds before the shrink starts
FIRE_DELAY = 0.08        # s between reading obs and the bullet actually leaving
ACC_GAIN = 0.5           # fraction of the measured acceleration used when leading (sim: best of 5 variants)
MAX_ACC = 58.5
DUEL_DIST = 70.0
RETREAT_DIST = 120.0
SOFT_BUDGET = 0.030      # seconds of compute per decision we allow ourselves
PLAN_BUDGET = 0.010
CUBE_CAP = 10
# ---- ONE switch per experiment branch ----
#   "hide": never take long walks early; hiding spot must be within NEST_RANGE; quiet laser grabs only
#   "hunt": play like the built-in bots: grab a laser, then go find fights; retreat only when nearly dead
STYLE = "hunt"
# ---- v8: hideouts ----
# hideouts.json (made offline by precompute_hideouts.py, the map never changes): every candidate
# spot with the number of open-ground positions within 200 m that can see a mech standing there.
# Typical spot: 80+.  Best spots: 0-10.  We go to the least-seen spot inside the next zone circle.
BUNKER = True
DIST_COST = 0.03           # exposure points per metre of travel when choosing a hideout
NEST_RANGE = 70.0 if STYLE == "hide" else 400.0
LASER_FULL = 0.95        # release the laser only at >= 95% charge (315 dmg vs 110 at no charge)
LASER_SAFE_HOLD = 1.6    # s after full charge before we must let go (server auto-fires at hold_s = 2.0)
RUSH_UNTIL = 90.0        # early game: go get a laser

# exact upgrade ids (order = priority); anything else falls back to the fuzzy list
UPGRADE_PRIORITY = ["armor", "plating", "nanites", "gun_rate", "gun_damage", "sensors"]
FUZZY = ["armor", "plat", "hull", "nano", "regen", "gun_rate", "gun_dam", "sensor"]
# evidence (recorded match): built-in bots grab a laser in the first seconds and snipe at ~150 m.
# so: laser always first, lightning second (it shorts lasers up close), never a mortar
# (mortar holders take +60% from lasers and the shell is easy to dodge).
WEAPON_PREF = {"laser": 3.0, "lightning": 1.0, "mortar": -5.0}
DIST_MID = {"near": 35.0, "mid": 100.0, "far": 220.0}
SOUND_WEAPON = {"gun": "gun/NPC-free", "laser": "laser holder", "laser_charge": "laser holder charging",
                "lightning": "lightning holder", "mortar": "mortar holder", "step": "mech walking",
                "boost": "mech boosting", "jet": "mech jetting", "reload": "gun reloading"}


def wrap(a):
    return (a + math.pi) % TAU - math.pi


def clampf(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def norm2(x, y):
    n = math.hypot(x, y)
    return (x / n, y / n) if n > 1e-9 else (0.0, 0.0)


def direct_aim(src, pos):
    dx, dy, dz = pos[0] - src[0], pos[1] - src[1], pos[2] - src[2]
    return math.atan2(dy, dx), math.atan2(dz, math.hypot(dx, dy))


def predict_xy(pos, vel, acc, T_):
    ta = min(T_, 0.8)
    return (pos[0] + vel[0] * T_ + 0.5 * acc[0] * ta * ta, pos[1] + vel[1] * T_ + 0.5 * acc[1] * ta * ta)


def lead_aim(src, pos, vel, acc, speed, delay=FIRE_DELAY):
    """Intercept for a constant-velocity + damped-acceleration target; returns (yaw, pitch) or None."""
    t = math.dist(src, pos) / speed
    for _ in range(4):
        px, py = predict_xy(pos, vel, acc, delay + t)
        t = math.dist(src, (px, py, pos[2])) / speed
    return direct_aim(src, (px, py, pos[2]))


def mortar_aim(src, pos, vel, acc, v=72.0, g=22.0, delay=FIRE_DELAY):
    """Low-arc ballistic solution with target lead. Returns (yaw, pitch, dist) or None."""
    tx, ty = pos[0], pos[1]
    for _ in range(3):
        dx, dy = tx - src[0], ty - src[1]
        r = math.hypot(dx, dy)
        dz = pos[2] - src[2]
        disc = v ** 4 - g * (g * r * r + 2 * dz * v * v)
        if disc < 0 or r < 1:
            return None
        th = math.atan((v * v - math.sqrt(disc)) / (g * r))
        flight = r / (v * math.cos(th))
        tx, ty = predict_xy(pos, vel, acc, delay + flight)
    return math.atan2(ty - src[1], tx - src[0]), th, r


def _pip(x, y, pts):
    inside = False
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _pairs(flat):
    return [(float(flat[i]), float(flat[i + 1])) for i in range(0, len(flat) - 1, 2)]


class World:
    CELL = 40.0
    NAV = 4.0

    def __init__(self, map_dir, seed, budget_s=5.0):
        t0 = time.perf_counter()
        self.deadline = t0 + budget_s
        self.prisms, self.grid = [], {}
        self.terr = None
        self.minx = self.miny = 0.0
        self.step, self.nx, self.ny = 1.0, 0, 0
        self.nav, self.navw, self.navh = None, 0, 0
        self.nests = []
        self.hideouts = []
        try:
            with open(os.path.join(map_dir, "v2", "world.json")) as f:
                w = json.load(f)
            self._load_prisms(w)
            self._load_terrain(map_dir, w)
            self._build_nav()
            self._build_nests(random.Random(seed))
            self._components()
            self._load_hideouts()
        except Exception as e:
            mf.log("world init problem:", repr(e))
        mf.log(f"world: prisms={len(self.prisms)} nav={'y' if self.nav else 'n'} nests={len(self.nests)} hideouts={len(self.hideouts)} {time.perf_counter()-t0:.2f}s")

    # ----- loading -----
    def _load_prisms(self, w):
        c = self.CELL
        for p in w.get("prisms", []):
            ring = _pairs(p["ring"])
            if len(ring) < 3:
                continue
            holes = p.get("holes") or []
            if holes and not isinstance(holes[0], (list, tuple)):
                holes = [holes]
            holes = [_pairs(h) for h in holes if len(h) >= 6]
            xs = [q[0] for q in ring]
            ys = [q[1] for q in ring]
            idx = len(self.prisms)
            self.prisms.append((float(p["base"]), float(p["top"]), ring, holes, min(xs), min(ys), max(xs), max(ys), p.get("kind", "")))
            for cx in range(int(min(xs) // c), int(max(xs) // c) + 1):
                for cy in range(int(min(ys) // c), int(max(ys) // c) + 1):
                    self.grid.setdefault((cx, cy), []).append(idx)

    def _load_terrain(self, map_dir, w):
        t = w["terrain"]
        self.minx, self.miny, self.step = float(t["minX"]), float(t["minY"]), float(t["step"])
        self.nx, self.ny = int(t["nx"]), int(t["ny"])
        arr = array("h")
        with open(os.path.join(map_dir, "v2", "terrain.i16"), "rb") as f:
            arr.frombytes(f.read())
        if sys.byteorder == "big":
            arr.byteswap()
        if len(arr) >= self.nx * self.ny:
            self.terr = arr

    def ground(self, x, y):
        if self.terr is None:
            return 0.0
        fx = clampf((x - self.minx) / self.step, 0.0, self.nx - 1.001)
        fy = clampf((y - self.miny) / self.step, 0.0, self.ny - 1.001)
        i, j = int(fx), int(fy)
        tx, ty = fx - i, fy - j
        a = self.terr[j * self.nx + i]
        b = self.terr[j * self.nx + i + 1]
        c = self.terr[(j + 1) * self.nx + i]
        d = self.terr[(j + 1) * self.nx + i + 1]
        return ((a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty) / 100.0

    # ----- queries -----
    def hit(self, x, y, zlo, zhi):
        for i in self.grid.get((int(x // self.CELL), int(y // self.CELL)), ()):
            base, top, ring, holes, x0, y0, x1, y1, _k = self.prisms[i]
            if base > zhi or top < zlo or x < x0 or x > x1 or y < y0 or y > y1:
                continue
            if _pip(x, y, ring) and not any(_pip(x, y, h) for h in holes):
                return True
        return False

    def los(self, a, b, step=4.0):
        dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        n = max(1, int(math.sqrt(dx * dx + dy * dy + dz * dz) / step))
        for i in range(1, n):
            f = i / n
            z = a[2] + dz * f
            if self.hit(a[0] + dx * f, a[1] + dy * f, z, z):
                return False
        return True

    def top_at(self, x, y, z):
        """Highest top of the solids standing in our way at (x, y) for a mech whose feet are at z."""
        best = None
        for i in self.grid.get((int(x // self.CELL), int(y // self.CELL)), ()):
            base, top, ring, holes, x0, y0, x1, y1, _k = self.prisms[i]
            if base > z + 9.0 or top < z + 1.2 or x < x0 or x > x1 or y < y0 or y > y1:
                continue
            if _pip(x, y, ring) and not any(_pip(x, y, h) for h in holes):
                best = top if best is None else max(best, top)
        return best

    def _components(self):
        self.comp = None
        if self.nav is None:
            return
        w, h = self.navw, self.navh
        comp = [0] * (w * h)
        sizes = {}
        label = 0
        nav = self.nav
        for start in range(w * h):
            if nav[start] or comp[start]:
                continue
            label += 1
            comp[start] = label
            stack = [start]
            n = 0
            while stack:
                k = stack.pop()
                n += 1
                cx, cy = k % w, k // w
                for nb in ((k - 1) if cx > 0 else -1, (k + 1) if cx < w - 1 else -1,
                           (k - w) if cy > 0 else -1, (k + w) if cy < h - 1 else -1):
                    if nb >= 0 and not nav[nb] and not comp[nb]:
                        comp[nb] = label
                        stack.append(nb)
            sizes[label] = n
        self.comp = comp
        self.main_comp = max(sizes, key=sizes.get) if sizes else 0

    def on_foot(self, x, y):
        if self.comp is None:
            return True
        cx, cy = self.cell(x, y)
        return self.comp[cy * self.navw + cx] == self.main_comp

    def _load_hideouts(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hideouts.json")
        try:
            with open(path) as f:
                rows = json.load(f)
            # keep the reasonably hidden ones: x, y, z, seen, near, shot, roof
            self.hideouts = [{"id": i, "spot": (r[0], r[1], r[2]), "seen": r[3], "roof": bool(r[6]),
                              "foot": (not r[6]) and self.on_foot(r[0], r[1])}
                             for i, r in enumerate(rows) if r[3] <= 60]
        except Exception as e:
            mf.log("hideouts not loaded:", repr(e))

    def walk_blocked(self, x, y, z):
        return self.hit(x, y, z + 1.2, z + 9.0)

    # ----- navigation grid -----
    def _build_nav(self):
        if self.terr is None:
            return
        cs = self.NAV
        self.navw = int(self.nx * self.step / cs)
        self.navh = int(self.ny * self.step / cs)
        blocked = bytearray(self.navw * self.navh)
        offs = ((0, 0), (1.5, 1.5), (-1.5, 1.5), (1.5, -1.5), (-1.5, -1.5))
        for cy in range(self.navh):
            if time.perf_counter() > self.deadline:
                break
            for cx in range(self.navw):
                x = self.minx + (cx + 0.5) * cs
                y = self.miny + (cy + 0.5) * cs
                g = self.ground(x, y)
                bad = False
                if (int(x // self.CELL), int(y // self.CELL)) in self.grid:
                    bad = any(self.hit(x + ox, y + oy, g + 1.2, g + 9.0) for ox, oy in offs)
                if not bad:
                    gx = (self.ground(x + 2, y) - self.ground(x - 2, y)) / 4
                    gy = (self.ground(x, y + 2) - self.ground(x, y - 2)) / 4
                    bad = math.hypot(gx, gy) > 1.19      # tan(50 deg)
                if bad:
                    blocked[cy * self.navw + cx] = 1
        self.nav = blocked

    def cell(self, x, y):
        cx = int(clampf((x - self.minx) / self.NAV, 0, self.navw - 1))
        cy = int(clampf((y - self.miny) / self.NAV, 0, self.navh - 1))
        return cx, cy

    def free(self, c):
        return 0 <= c[0] < self.navw and 0 <= c[1] < self.navh and not self.nav[c[1] * self.navw + c[0]]

    def snap_free(self, c):
        if self.free(c):
            return c
        for r in range(1, 8):
            for dx in range(-r, r + 1):
                for dy in (-r, r):
                    for cc in ((c[0] + dx, c[1] + dy), (c[0] + dy, c[1] + dx)):
                        if self.free(cc):
                            return cc
        return c

    def center(self, c):
        return (self.minx + (c[0] + 0.5) * self.NAV, self.miny + (c[1] + 0.5) * self.NAV)

    # ----- nests (low-exposure spots beside buildings) -----
    def _build_nests(self, rnd):
        edges = []
        for base, top, ring, holes, x0, y0, x1, y1, kind in self.prisms:
            if kind not in ("building", "stadium") or top - base < 8:
                continue
            n = len(ring)
            for i in range(n):
                ax, ay = ring[i]
                bx, by = ring[(i + 1) % n]
                if math.hypot(bx - ax, by - ay) >= 8:
                    edges.append((ring, ax, ay, bx, by))
        rnd.shuffle(edges)
        dirs = [(math.cos(TAU * k / 16), math.sin(TAU * k / 16)) for k in range(16)]
        for ring, ax, ay, bx, by in edges[:600]:
            if time.perf_counter() > self.deadline:
                break
            mx, my = (ax + bx) / 2, (ay + by) / 2
            nx_, ny_ = norm2(-(by - ay), bx - ax)
            if _pip(mx + nx_ * 0.5, my + ny_ * 0.5, ring):
                nx_, ny_ = -nx_, -ny_
            px, py = mx + nx_ * 4.5, my + ny_ * 4.5
            g = self.ground(px, py)
            if self.walk_blocked(px, py, g):
                continue
            if self.nav is not None and not self.free(self.cell(px, py)):
                continue
            clear = 0
            for dx, dy in dirs:
                ok = True
                for s in range(1, 9):
                    if self.hit(px + dx * 7.5 * s, py + dy * 7.5 * s, g + 4.5, g + 4.5):
                        ok = False
                        break
                clear += ok
            self.nests.append((px, py, clear / 16.0))


NEIGH = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
         (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]


class Planner:
    """Time-sliced A* over the 4 m nav grid; call step(deadline) every tick until `path` is ready."""

    def __init__(self, world):
        self.w = world
        self.active = False
        self.goal = None
        self.path = []
        self.i = 0
        self.failed = False

    def request(self, start, goal):
        w = self.w
        sc, gc = w.snap_free(w.cell(*start)), w.snap_free(w.cell(*goal))
        self.gc, self.goal = gc, goal
        self.g = {sc: 0.0}
        self.came = {}
        self.open = [(self._h(sc, gc), 0.0, sc)]
        self.active, self.failed, self.exp = True, False, 0
        self.path, self.i = [], 0

    @staticmethod
    def _h(a, b):
        dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
        return (dx + dy) + (1.414 - 2) * min(dx, dy)

    def step(self, deadline):
        if not self.active:
            return
        w, g, op = self.w, self.g, self.open
        n = 0
        while op:
            n += 1
            if n % 24 == 0 and time.perf_counter() > deadline:
                return
            f, gc_, cur = heapq.heappop(op)
            if gc_ > g.get(cur, 1e18):
                continue
            if cur == self.gc:
                cells = [cur]
                while cur in self.came:
                    cur = self.came[cur]
                    cells.append(cur)
                cells.reverse()
                pts = [w.center(c) for c in cells[::2]] + [w.center(cells[-1])]
                self.path, self.i, self.active = pts, 0, False
                return
            self.exp += 1
            if self.exp > 30000:
                break
            cx, cy = cur
            for dx, dy, c in NEIGH:
                nb = (cx + dx, cy + dy)
                if not w.free(nb):
                    continue
                if dx and dy and not (w.free((cx + dx, cy)) and w.free((cx, cy + dy))):
                    continue
                ng = gc_ + c
                if ng < g.get(nb, 1e18):
                    g[nb] = ng
                    self.came[nb] = cur
                    heapq.heappush(op, (ng + self._h(nb, self.gc), ng, nb))
        self.active, self.failed = False, True


class Bot:
    name = "survivor-v2"

    def __init__(self, hello):
        seed = int(hello.get("seed", 0)) ^ (int(hello.get("slot", 0)) * 7919)
        self.rng = random.Random(seed)
        self.world = World(hello.get("map", {}).get("dir", "/arena"), seed)
        self.plan = Planner(self.world) if self.world.nav is not None else None
        self.plan_tick = -999
        self.pref = 1
        self.bad = set()
        self.tracks = {}
        self.marks = []              # (t, x, y, kind, dist_label)
        self.tb = None
        self.last_dmg = -99.0
        self.cover = None
        self.stuck = 0
        self.healing = False
        self.nest = None
        self.side = 1
        self.side_until = 0.0
        self.laser_start = None
        self.last_pos = None
        self.full_since = None       # match time our laser reached full charge
        self.beam_t = {}             # enemy id -> time of its last laser beam
        self.laser_threat_t = -99.0  # last time a laser was charging at us / hit us
        self.last_mech_seen = -99.0
        self.bunker = None          # chosen courtyard dict
        self.bunker_fail = {}       # court id -> failed climbs
        self.climb_t0 = None
        self.climb_top = None
        self.exit_from = None
        self.direct = False
        self.fail_areas = {}
        cfg = hello.get("config", {})
        self.pick_r = cfg.get("loot", {}).get("pickup_radius", 4.0) - 0.6

    # ------------------------------------------------------------ bookkeeping
    def read_events(self, obs, t, x, y):
        for ev in obs.get("events", []):
            ty = ev.get("type")
            if ty == "take_failed":
                self.bad.add(ev["id"])
            elif ty == "damaged":
                self.last_dmg = t
                if ev.get("weapon") == "laser":
                    self.laser_threat_t = t
                if ev.get("bearing") is not None:
                    self.tb = (ev["bearing"], t)
        for b in obs.get("beams", []):
            if b.get("kind") == "laser":
                self.beam_t[b["by"]] = t - b.get("age", 0.0)
        self.marks = [m for m in self.marks if t - m[0] < 4.0]
        rank = {"near": 0, "mid": 1, "far": 2}
        best = None
        for s in obs.get("sounds", []):
            k = s["kind"]
            if k not in SOUND_WEAPON:
                continue
            d = DIST_MID.get(s["dist"], 220.0)
            d = min(d, 0.8 * 320.0)
            b = s["bearing"]
            self.marks.append((t - s["age"], x + d * math.cos(b), y + d * math.sin(b), k, s["dist"]))
            key = (rank.get(s["dist"], 2), s["age"])
            if best is None or key < best[0]:
                best = (key, b)
        if best and best[0][0] <= 1 and (self.tb is None or self.tb[1] < t - 1.0):
            self.tb = (best[1], t)

    def update_tracks(self, obs, t, x, y):
        for m in obs["mechs"]:
            i = m["id"]
            tr = self.tracks.get(i)
            acc = [0.0, 0.0]
            d = math.dist(m["pos"][:2], (x, y))
            rate = 0.0
            if tr and 0 < t - tr["t"] <= 0.35:
                dt = t - tr["t"]
                ax = (m["vel"][0] - tr["vel"][0]) / dt
                ay = (m["vel"][1] - tr["vel"][1]) / dt
                n = math.hypot(ax, ay)
                if n > MAX_ACC:
                    ax, ay = ax * MAX_ACC / n, ay * MAX_ACC / n
                acc = [ACC_GAIN * ax, ACC_GAIN * ay]
                rate = (d - tr["d"]) / dt
            self.tracks[i] = {"t": t, "vel": m["vel"], "acc": acc, "d": d, "rate": rate,
                              "weapons": m.get("weapons", [])}

    def weapon_pref(self, t):
        return dict(WEAPON_PREF)

    def pick_upgrade(self, offer):
        ids = [c["id"].lower() for c in offer["choices"]]
        for want in UPGRADE_PRIORITY:
            if want in ids:
                return ids.index(want)
        for want in FUZZY:
            for i, cid in enumerate(ids):
                if want in cid:
                    return i
        return 0

    # ------------------------------------------------------------ geometry / movement
    def region(self, obs):
        z = obs["zone"]
        if obs["t"] >= z["shrink_start"] - REGION_LEAD:
            c, r = z["next_center"], z["next_radius"]
        else:
            c, r = z["center"], z["radius"]
        return c, max(r - max(10.0, 0.12 * r), 5.0)

    def steer(self, x, y, z, tx, ty):
        base = math.atan2(ty - y, tx - x)
        for off in (0, 0.4, 0.9, 1.4, 2.0, 2.6, 3.14):
            for sgn in ((1,) if off in (0, 3.14) else (self.pref, -self.pref)):
                a = base + off * sgn
                ca, sa = math.cos(a), math.sin(a)
                if not (self.world.walk_blocked(x + 4 * ca, y + 4 * sa, z) or self.world.walk_blocked(x + 9 * ca, y + 9 * sa, z)):
                    if off and off != 3.14:
                        self.pref = sgn
                    return ca, sa
        return math.cos(base), math.sin(base)

    def go_to(self, x, y, z, goal, tick, deadline):
        gx, gy = goal[0], goal[1]
        if self.plan is None or math.hypot(gx - x, gy - y) < 14:
            return self.steer(x, y, z, gx, gy)
        pl = self.plan
        need = (pl.goal is None or math.hypot(pl.goal[0] - gx, pl.goal[1] - gy) > 10
                or (pl.failed and tick - self.plan_tick > 40) or self.stuck >= 20)
        if need and tick - self.plan_tick >= 8:
            pl.request((x, y), (gx, gy))
            self.plan_tick = tick
            self.stuck = 0 if self.stuck >= 20 else self.stuck
        if pl.active:
            pl.step(min(deadline, time.perf_counter() + PLAN_BUDGET))
        if pl.path:
            while pl.i < len(pl.path) - 1 and math.hypot(pl.path[pl.i][0] - x, pl.path[pl.i][1] - y) < 5.5:
                pl.i += 1
            wp = pl.path[min(pl.i + 1, len(pl.path) - 1)]
            return self.steer(x, y, z, wp[0], wp[1])
        return self.steer(x, y, z, gx, gy)

    def find_cover(self, x, y, z, threats, rc, rr, deadline):
        ts = [(p[0], p[1], p[2] + 8.0) for p in threats[:3]]
        if not ts:
            return None
        for d in (8, 16, 26):
            best, bd = None, 1e9
            for k in range(16):
                a = TAU * k / 16
                px, py = x + d * math.cos(a), y + d * math.sin(a)
                if math.hypot(px - rc[0], py - rc[1]) > rr or self.world.walk_blocked(px, py, z):
                    continue
                ok = True
                for tx, ty, tz in ts:
                    if self.world.los((tx, ty, tz), (px, py, z + 4.5)) or self.world.los((tx, ty, tz), (px, py, z + 9)):
                        ok = False
                        break
                if ok:
                    dd = min(math.dist((px, py), (q[0], q[1])) for q in ts)
                    score = d - 0.05 * dd
                    if score < bd:
                        best, bd = (px, py), score
                if time.perf_counter() > deadline:
                    return best
            if best:
                return best
        return None

    def pick_nest(self, x, y, rc, rr):
        best, bs = None, 1e18
        for px, py, ex in self.world.nests:
            if math.hypot(px - rc[0], py - rc[1]) > rr - 3:
                continue
            d = math.hypot(px - x, py - y)
            if d > NEST_RANGE:
                continue
            s = d + 250.0 * ex
            if s < bs:
                best, bs = (px, py), s
        return best

    # ------------------------------------------------------------ dodging
    def dodge(self, obs, x, y, z, mechs, energy):
        cx, cy = x, y
        for p in obs.get("projectiles", []):
            px, py, pz = p["pos"]
            vx, vy, vz = p["vel"]
            if p["kind"] == "mortar":
                disc = vz * vz + 44 * (pz - z)
                if disc < 0:
                    continue
                tl = (vz + math.sqrt(disc)) / 22
                if 0 < tl < 3:
                    lx, ly = px + vx * tl, py + vy * tl
                    if math.hypot(lx - x, ly - y) < 12:
                        return norm2(x - lx, y - ly), False
            else:
                v2 = vx * vx + vy * vy
                if v2 < 1:
                    continue
                rx, ry = px - cx, py - cy
                tc = -(rx * vx + ry * vy) / v2
                if 0 < tc < 1.0:
                    mx, my = rx + vx * tc, ry + vy * tc
                    if math.hypot(mx, my) < 4.5:
                        n = math.sqrt(v2)
                        pxn, pyn = -vy / n, vx / n
                        side = -1 if (mx * pxn + my * pyn) > 0 else 1
                        return (pxn * side, pyn * side), False
        for m in mechs:                       # charging laser pointed at me -> leave the beam line
            if m.get("charging"):
                ex, ey = m["pos"][0], m["pos"][1]
                ux, uy = math.cos(m["aim"][0]), math.sin(m["aim"][0])
                rx, ry = x - ex, y - ey
                along = rx * ux + ry * uy
                s = rx * -uy + ry * ux
                if along > 0 and abs(s) < 6.0 and math.hypot(rx, ry) < 290:
                    sg = 1 if s >= 0 else -1
                    self.laser_threat_t = obs["t"]
                    if abs(s) < 0.5:
                        sg = self.rng.choice((-1, 1))
                    return (-uy * sg, ux * sg), energy > 25
        return None, False

    # ------------------------------------------------------------ main
    def act(self, obs):
        t0 = time.perf_counter()
        deadline = t0 + SOFT_BUDGET
        me = obs["self"]
        x, y, z = me["pos"]
        t = obs["t"]
        st = me.get("stats", {})
        eye_h = st.get("eye_height", 8.0)
        muz_h = st.get("muzzle_height", 7.0)
        act = {"tick": obs["tick"]}
        tick = obs["tick"]
        self.read_events(obs, t, x, y)
        self.update_tracks(obs, t, x, y)
        if obs.get("offer"):
            act["pick"] = self.pick_upgrade(obs["offer"])

        weapons = me["weapons"]
        slot_of = {w["kind"]: i for i, w in enumerate(weapons) if w}
        gi, li, ni, mi = slot_of.get("gun"), slot_of.get("laser"), slot_of.get("lightning"), slot_of.get("mortar")
        hpr = me["hp"] / max(me["max_hp"], 1.0)
        if hpr < (0.35 if STYLE == "hide" else 0.25):
            self.healing = True
        elif hpr >= 0.75:
            self.healing = False
        cur = me["aim"]
        muzzle = (x, y, z + muz_h)

        # ---------- rank enemies ----------
        mech_list = [m for m in obs["mechs"] if m["kind"] == "mech"]
        if mech_list:
            self.last_mech_seen = t
        npcs = [m for m in obs["mechs"] if m["kind"] == "npc"]
        mech_list.sort(key=lambda m: math.dist(m["pos"][:2], (x, y)))
        ranked = []
        for n_, m in enumerate(mech_list[:6]):
            tp = m["pos"]
            d = math.dist(tp[:2], (x, y))
            ch = 4.5 * (1 + 0.05 * m.get("cubes", 0))
            clear = n_ < 4 and d < 220 and self.world.los((x, y, z + eye_h), (tp[0], tp[1], tp[2] + ch))
            to_me = math.atan2(y - tp[1], x - tp[0])
            f_err = abs(wrap(m["aim"][0] - to_me))
            turn = abs(wrap(math.atan2(tp[1] - y, tp[0] - x) - cur[0]))
            chg = bool(m.get("charging")) and f_err < math.atan2(6.0, max(d, 10.0))
            sc = T.score(d, m.get("hp"), m.get("weapons") or ["gun"], f_err, turn, clear, chg)
            ranked.append((sc, d, m, clear, ch))
        ranked.sort(key=lambda r: r[0])
        clear_close = [r for r in ranked if r[3] and r[1] < 150]
        target = next((r for r in ranked if r[3] and r[1] < 220), None)
        hidden_vis = [r for r in ranked if not r[3] and r[1] < 200]
        threat_pos = [r[2]["pos"] for r in ranked if r[1] < 200]

        rc, rr = self.region(obs)
        dist_c = math.hypot(x - rc[0], y - rc[1])
        inside = dist_c <= rr
        if inside:
            rp = (x, y)
        else:
            k = rr / dist_c
            rp = (rc[0] + (x - rc[0]) * k, rc[1] + (y - rc[1]) * k)
        recent_dmg = t - self.last_dmg < 3.0

        # ---------- decide movement ----------
        mode, goal, mv = "hold", None, (0.0, 0.0)
        urgent = not me["in_zone"]
        boost = False
        farm = None            # non-mech thing to shoot: ('npc'|'crate', pos, vel, ch, d)
        dg, dg_boost = self.dodge(obs, x, y, z, mech_list, me["energy"])

        if t >= self.side_until:
            self.side = self.rng.choice((-1, 1))
            self.side_until = t + self.rng.uniform(0.6, 1.6)

        # ---- laser rush: the built-in bots win with lasers, so get one first ----
        rush = None
        if li is None and (t < RUSH_UNTIL or any(w is None for w in weapons) or mi is not None):
            lasers = [p for p in obs.get("pickups", []) if p["kind"] == "laser" and p["id"] not in self.bad
                      and abs(p["pos"][2] - z) < 25]
            if lasers:
                p_ = min(lasers, key=lambda p: math.dist(p["pos"][:2], (x, y)))
                dp_ = math.dist(p_["pos"][:2], (x, y))
                max_rush = 60.0 if (STYLE == "hide" and t < 75) else 200.0
                if dp_ < max_rush and (not clear_close or dp_ < 30) and (me["in_zone"] or dp_ < 40):
                    rush = (p_, dp_)
        outgunned = li is None and (any("laser" in (r[2].get("weapons") or []) for r in clear_close)
                                    or t - self.laser_threat_t < 2.5)
        # ---- v8: courtyard bunker overrides travel and fights (we still shoot back while moving) ----
        bunker = self.choose_bunker(obs, x, y, rc, rr)
        bmode = None
        if bunker is not None and not (rush is not None and rush[1] < 30):
            bmode, bmv, bdirect = self.bunker_move(bunker, me, x, y, z, t, tick, deadline)
        if bmode is not None:
            mode, goal, mv = bmode, bunker["spot"], bmv
            urgent = bmode not in ("bunker", "bunker-in")
            if bdirect and not self.climb_assist(act, me, x, y, z, mv):
                mode, mv = "bunker-charge", (0.0, 0.0)
        elif rush is not None and not dg:
            p_, dp_ = rush
            if dp_ < self.pick_r and abs(p_["pos"][2] - z) < 4:
                act["take"] = p_["id"]
                if not any(w is None for w in weapons):
                    act["drop"] = mi if mi is not None else (ni if ni is not None else 1)
                mode, goal, mv = "take", p_["pos"], (0.0, 0.0)
            else:
                mode, goal = "rush", p_["pos"]
                mv = self.go_to(x, y, z, p_["pos"], tick, deadline)
                urgent = True
                boost = STYLE == "hunt" and dp_ > 25 and me["energy"] > 35 and not clear_close
                urgent = STYLE == "hunt" or bool(clear_close)
        elif dg:
            mode, mv, urgent, boost = "dodge", dg, True, dg_boost
        elif clear_close:
            sc_, d, m, clear, ch = ranked[0] if ranked[0][3] else clear_close[0]
            tp = m["pos"]
            bearing = math.atan2(tp[1] - y, tp[0] - x)
            tr = self.tracks.get(m["id"], {})
            rate = tr.get("rate", 0.0)
            weak = hpr < (0.6 if STYLE == "hide" else 0.3)
            group = len(clear_close) >= (2 if STYLE == "hide" else 3)
            urgent = True
            if group or self.healing or weak or outgunned:
                mode = "retreat"
                if self.cover is None or tick - self.cover[0] >= 6:
                    self.cover = (tick, self.find_cover(x, y, z, threat_pos, rc, rr, deadline))
                cv = self.cover[1]
                if cv and math.dist(cv, (x, y)) > 2:
                    goal = cv
                    mv = self.go_to(x, y, z, cv, tick, deadline)
                    sp = DUEL_DIST
                else:
                    sp = RETREAT_DIST
                    mv = self.radial_strafe(d, rate, bearing, sp)
            else:
                mode = "duel"
                # with a laser, stay far: gun bots miss at 140 m, our beam does not
                mv = self.radial_strafe(d, rate, bearing, 140.0 if li is not None else DUEL_DIST)
            if not me["in_zone"]:
                bx, by = self.steer(x, y, z, rp[0], rp[1])
                mv = norm2(mv[0] + bx, mv[1] + by)
        elif hidden_vis or recent_dmg:
            mode = "wary"
            self.cover = None
            if self.healing and self.tb and t - self.tb[1] < 3:
                mode, mv, urgent = "heal-away", (-math.cos(self.tb[0]), -math.sin(self.tb[0])), True
                mv = self.steer(x, y, z, x + mv[0] * 20, y + mv[1] * 20)
            if not inside:
                mode, goal = "zone", rp
                mv = self.go_to(x, y, z, rp, tick, deadline)
        else:
            self.cover = None
            mode, goal, mv, farm = self.peaceful(obs, me, x, y, z, rc, rr, inside, rp, npcs, hpr, weapons, slot_of,
                                                 t, tick, deadline, act, muzzle)
            if mode in ("zone", "drop"):
                urgent = urgent or mode == "zone"

        # ---------- apply movement ----------
        if mode in ("hold", "crate", "heal", "take", "bunker", "bunker-charge") or (mv[0] == 0 and mv[1] == 0):
            move = [0.0, 0.0]
        else:
            sc_move = 1.0 if urgent else (0.75 if mode == "drop" else NOISE_FREE)
            move = [mv[0] * sc_move, mv[1] * sc_move]
        act["move"] = move
        if (not me["in_zone"] and me["energy"] > 8 and mode in ("zone", "retreat", "duel")) or boost:
            act["boost"] = True

        spd = math.hypot(me["vel"][0], me["vel"][1])
        if mode in ("zone", "loot", "cover", "retreat", "nest", "cubes", "drop", "rush", "hunt", "bunker-go") and (move[0] or move[1]) and spd < 1.5:
            self.stuck += 1
        else:
            self.stuck = max(0, self.stuck - 2) if self.stuck < 20 else 0
        if 6 <= self.stuck < 20 and not mode.startswith("bunker"):
            act["jump"] = True
            act["boost"] = True
            if self.stuck % 12 == 0:
                self.pref = -self.pref

        # ---------- aim and fire ----------
        fire = [False, False]
        aim = None
        aimed_at = None
        if target is not None:
            sc_, d, m, clear, ch = target
            tr = self.tracks.get(m["id"], {"acc": [0, 0]})
            tc = (m["pos"][0], m["pos"][1], m["pos"][2] + ch)
            aim, fire = self.shoot(me, weapons, gi, li, ni, mi, muzzle, cur, tc, m["vel"], tr["acc"], d, True, tick, 170.0)
            aimed_at = tc
        elif farm is not None:
            kind, p, v, ch, d = farm
            tc = (p[0], p[1], p[2] + ch)
            aim, fire = self.shoot(me, weapons, gi, None, None, None, muzzle, cur, tc, v, [0.0, 0.0], d, True, tick, 60.0)
            aimed_at = tc
        else:
            lob = next((r for r in ranked if not r[3] and r[1] < 200 and t - self.tracks.get(r[2]["id"], {"t": -99})["t"] < 1.0), None)
            if lob is not None and mi is not None:
                sc_, d, m, clear, ch = lob
                tr = self.tracks.get(m["id"], {"acc": [0, 0]})
                tc = (m["pos"][0], m["pos"][1], m["pos"][2] + ch)
                aim, fire = self.shoot(me, weapons, None, None, None, mi, muzzle, cur, tc, m["vel"], tr["acc"], d, False, tick, 170.0)
                aimed_at = tc
            if aim is None and li is not None and (hidden_vis or weapons[li].get("charge", 0) > 0):
                lw = weapons[li]
                charge = lw.get("charge", 0.0)
                t_now = tick * 0.05
                if charge <= 0:
                    self.full_since = None
                elif charge >= lw["stats"].get("charge", 0.75) * LASER_FULL and self.full_since is None:
                    self.full_since = t_now
                if hidden_vis:
                    tp = hidden_vis[0][2]["pos"]
                    aim = direct_aim(muzzle, (tp[0], tp[1], tp[2] + hidden_vis[0][4]))
                else:
                    aim = (cur[0], cur[1])
                held = t_now - self.full_since if self.full_since is not None else 0.0
                if charge > 0:
                    fire[li] = held < LASER_SAFE_HOLD
                elif lw["cooldown"] <= 0.05 and hidden_vis and hidden_vis[0][1] < 220:
                    fire[li] = True
            if aim is None:
                if hidden_vis:
                    tp = hidden_vis[0][2]["pos"]
                    aim = (math.atan2(tp[1] - y, tp[0] - x), 0.0)
                elif self.tb and t - self.tb[1] < 4.0:
                    aim = (self.tb[0], 0.0)
                elif (move[0] or move[1]) and mode in ("zone", "loot", "nest", "cubes", "drop", "hunt", "rush"):
                    aim = (math.atan2(move[1], move[0]), 0.0)
                else:
                    aim = (wrap(tick * 0.12), 0.0)
        if aim is not None:
            act["aim"] = [aim[0], aim[1]]
        if fire[0] or fire[1]:
            act["fire"] = fire
        elif gi is not None and not threat_pos:
            gs = weapons[gi]
            if gs.get("ammo", 99) < gs["stats"].get("mag", 12) * 0.6 and gs.get("reload", 0) <= 0:
                act["reload"] = True

        # ---------- debug ----------
        el = (time.perf_counter() - t0) * 1000
        mf.draw.label(f"{mode} hp{me['hp']:.0f} e{len(mech_list)} {el:.0f}ms")
        mf.draw.point([rc[0], rc[1], z], "#40c0ff", 5)
        if goal:
            mf.draw.point([goal[0], goal[1], z], "#60ff60", 6)
        if aimed_at:
            mf.draw.line([x, y, z + muz_h], list(aimed_at), "#ff5040")
        for mk in self.marks[-6:]:
            mf.draw.point([mk[1], mk[2], z], "#ffd060", 3)
            mf.draw.text(f"{mk[3]} {mk[4]}: {SOUND_WEAPON.get(mk[3], '?')}", [mk[1], mk[2], z + 3], "#ffd060")
        return act

    # ------------------------------------------------------------ v8 hideouts
    def at_spot(self, c, x, y, z):
        sx, sy, sz = c["spot"]
        return math.hypot(sx - x, sy - y) < 2.5 and abs(z - sz) < 2.5

    def choose_bunker(self, obs, x, y, rc, rr):
        if not BUNKER or not self.world.hideouts:
            return None
        zn = obs["zone"]
        nc, nr = zn["next_center"], zn["next_radius"]
        def score(c):
            sx, sy, _ = c["spot"]
            if math.hypot(sx - rc[0], sy - rc[1]) > rr - 5 or math.hypot(sx - nc[0], sy - nc[1]) > nr - 12:
                return None
            pen = 0.0 if c["foot"] else 12.0
            for (fx, fy), n in self.fail_areas.items():
                if math.hypot(sx - fx, sy - fy) < 25:
                    pen += 30.0 * n
            return c["seen"] + pen + DIST_COST * math.hypot(sx - x, sy - y)
        cur = score(self.bunker) if self.bunker is not None else None
        best, bs = None, 1e18
        for c in self.world.hideouts:
            v = score(c)
            if v is not None and v < bs:
                best, bs = c, v
        if best is None:
            return None if cur is None else self.bunker
        if cur is not None and bs > cur - 15.0:      # commit: only switch for a clearly better spot
            return self.bunker
        if best is not self.bunker:
            self.bunker = best
            self.climb_t0 = None
            self.direct = False
        return best

    def bunker_move(self, c, me, x, y, z, t, tick, deadline):
        """Returns (mode, mv, direct).  direct=True: walk straight at it and climb whatever is in the way."""
        sx, sy, sz = c["spot"]
        d = math.hypot(sx - x, sy - y)
        if self.at_spot(c, x, y, z):
            self.climb_t0 = None
            return "bunker", (0.0, 0.0), False
        elevated = z - self.world.ground(x, y) > 4.0 and abs(z - sz) > 2.5     # on the wrong roof
        pl = self.plan
        path_done = (pl is not None and not pl.active and pl.path and pl.i >= len(pl.path) - 2
                     and pl.goal is not None and math.hypot(pl.goal[0] - sx, pl.goal[1] - sy) < 3)
        if not getattr(self, "direct", False):
            if (elevated or d < (6 if c["foot"] else 14) or (c["roof"] and d < 35) or self.stuck >= 15
                    or (pl is not None and pl.failed) or (path_done and d > 4)):
                self.direct = True
        # progress check: bouncing in place without getting closer -> go straight and climb
        pg = getattr(self, "prog", None)
        if pg is None or pg[2] is not c:
            self.prog = (t, d, c)
        elif t - pg[0] > 4.0:
            if d > pg[1] - 3.0:
                self.direct = True
            self.prog = (t, d, c)
        if not self.direct:
            self.climb_t0 = None
            return "bunker-go", self.go_to(x, y, z, (sx, sy), tick, deadline), False
        if self.climb_t0 is None:
            self.climb_t0 = t
        elif t - self.climb_t0 > 12.0:             # cannot get there: penalise it and pick again
            self.bunker_fail[c["id"]] = self.bunker_fail.get(c["id"], 0) + 1
            key = (round(sx, -1), round(sy, -1))
            self.fail_areas[key] = self.fail_areas.get(key, 0) + 1
            self.climb_t0 = None
            self.direct = False
            self.bunker = None
        return "bunker-climb", norm2(sx - x, sy - y), True

    def climb_assist(self, act, me, x, y, z, mv):
        """Jump + jet over the solid in front of us; returns False if we must wait for energy."""
        if mv[0] == 0 and mv[1] == 0:
            return True
        px, py = x + mv[0] * 3.5, y + mv[1] * 3.5
        top = self.world.top_at(px, py, z)
        if me["grounded"]:
            # on the ground: only the wall right in front of us counts (forget older, taller ones)
            self.climb_top = top if (top is not None and top - z < 50) else None
        elif top is not None and top - z < 50 and (self.climb_top is None or top > self.climb_top):
            self.climb_top = top
        if self.climb_top is not None and z < self.climb_top + 2.0:
            if me["grounded"]:
                need = min(95.0, 1.6 * (self.climb_top - z) + 10.0)
                if me["energy"] < need:
                    return False
                act["jump"] = True
            else:
                act["boost"] = True
        return True

    # ------------------------------------------------------------ helpers used by act()
    def radial_strafe(self, d, rate, bearing, setpoint):
        """PD on distance (approach when too far, back off when too close) + random-period strafe."""
        u = clampf(0.04 * (d - setpoint) + 0.10 * rate, -1.0, 1.0)
        rx, ry = math.cos(bearing), math.sin(bearing)
        px, py = -ry * self.side, rx * self.side
        return norm2(rx * u + px * 0.9, ry * u + py * 0.9)

    def peaceful(self, obs, me, x, y, z, rc, rr, inside, rp, npcs, hpr, weapons, slot_of, t, tick, deadline, act, muzzle):
        farm = None
        if not inside:
            return "zone", rp, self.go_to(x, y, z, rp, tick, deadline), None
        if self.healing:
            return "heal", None, (0.0, 0.0), None
        have_free = any(w is None for w in weapons)
        # 1) weapon loot by triangle preference
        pref = self.weapon_pref(t)
        cur_other = next((k for k in slot_of if k != "gun"), None)
        cands = []
        for p in obs.get("pickups", []):
            k = p["kind"]
            if p["id"] in self.bad or k not in pref or k in slot_of or pref[k] < 0:
                continue
            if have_free or (cur_other is not None and pref[k] > pref.get(cur_other, -9) + 0.5
                             and not (cur_other == "laser" and weapons[slot_of["laser"]].get("charge", 0) > 0)):
                cands.append((-pref[k], math.dist(p["pos"][:2], (x, y)), p))
        cands.sort(key=lambda c: (c[0], c[1]))
        if cands:
            p = cands[0][2]
            hd = math.dist(p["pos"][:2], (x, y))
            if hd < 3.5 and abs(p["pos"][2] - z) < 4:
                act["take"] = p["id"]
                if not have_free:
                    act["drop"] = slot_of[cur_other]
            elif hd < 40 and math.hypot(p["pos"][0] - rc[0], p["pos"][1] - rc[1]) < rr:
                return "loot", p["pos"], self.go_to(x, y, z, p["pos"], tick, deadline), None
        # 2) supply drop close to me and nobody near it
        if t < 200 and hpr > 0.5:
            for dp in obs.get("drops", []):
                px, py = dp["pos"][0], dp["pos"][1]
                dd = math.hypot(px - x, py - y)
                if dd > 130 or math.hypot(px - rc[0], py - rc[1]) > rr:
                    continue
                if any(math.hypot(m["pos"][0] - px, m["pos"][1] - py) < 40 for m in obs["mechs"] if m["kind"] == "mech"):
                    continue
                if dd < dp["radius"] * 0.5 and dp["lands_in"] <= 0:
                    return "hold", (px, py), (0.0, 0.0), None
                if dd / (11.7 * 0.75) > dp["lands_in"] + 8:
                    continue
                return "drop", (px, py), self.go_to(x, y, z, (px, py), tick, deadline), None
        # 3) cubes
        if me["cubes"] < CUBE_CAP:
            cs = [c for c in obs.get("cubes", []) if math.dist(c["pos"][:2], (x, y)) < 40
                  and math.hypot(c["pos"][0] - rc[0], c["pos"][1] - rc[1]) < rr]
            if cs:
                c = min(cs, key=lambda c: math.dist(c["pos"][:2], (x, y)))
                return "cubes", c["pos"], self.go_to(x, y, z, c["pos"], tick, deadline), None
        # 4) NPC drones (only healthy) and 5) crates, both with the gun
        quiet = not any(m[3] in ("gun", "laser", "lightning", "mortar") and m[4] == "near" for m in self.marks)
        if hpr > 0.8 and quiet and t - self.last_mech_seen > 8.0 and any(w and w["kind"] == "gun" for w in weapons):
            ns = [(math.dist(n["pos"][:2], (x, y)), n) for n in npcs]
            ns = [(d, n) for d, n in ns if d < 45 and self.world.los((x, y, z + 8), (n["pos"][0], n["pos"][1], n["pos"][2] + 1.7))]
            if ns:
                d, n = min(ns, key=lambda a: a[0])
                return "npc", None, (0.0, 0.0), ("npc", n["pos"], n["vel"], 1.7, d)
            if t < 300:
                cr = [(math.dist(c["pos"][:2], (x, y)), c) for c in obs.get("crates", [])]
                cr = [(d, c) for d, c in cr if d < 45 and self.world.los((x, y, z + 8), (c["pos"][0], c["pos"][1], c["pos"][2] + 1.5))]
                if cr:
                    d, c = min(cr, key=lambda a: a[0])
                    return "crate", None, (0.0, 0.0), ("crate", c["pos"], [0, 0, 0], 1.5, d)
        # 6h) hunt style: walk toward the most recent enemy sighting or loud sound, else toward the centre
        if STYLE == "hunt":
            tgt = None
            recent = [tr for i_, tr in self.tracks.items() if i_ < 100 and t - tr["t"] < 10]
            if self.marks:
                mk = max(self.marks, key=lambda m: m[0])
                if mk[3] in ("gun", "laser", "lightning", "step", "boost", "reload"):
                    tgt = (mk[1], mk[2])
            if tgt is None:
                tgt = (rc[0], rc[1])
            if math.hypot(tgt[0] - rc[0], tgt[1] - rc[1]) > rr:
                tgt = (rc[0], rc[1])
            if math.hypot(tgt[0] - x, tgt[1] - y) > 25:
                return "hunt", tgt, self.go_to(x, y, z, tgt, tick, deadline), None
        # 6) tactical nest (low exposure spot beside a building), then hold
        if self.nest is None or math.hypot(self.nest[0] - rc[0], self.nest[1] - rc[1]) > rr - 3:
            self.nest = self.pick_nest(x, y, rc, rr)
        if self.nest and math.hypot(self.nest[0] - x, self.nest[1] - y) > 3.0:
            return "nest", self.nest, self.go_to(x, y, z, self.nest, tick, deadline), None
        return "hold", self.nest, (0.0, 0.0), None

    def shoot(self, me, weapons, gi, li, ni, mi, muzzle, cur, tc, vel, acc, d, clear, tick, gun_range):
        """Choose weapon + aim for target centre tc. Returns (aim, fire[2])."""
        fire = [False, False]
        aim = None
        # laser: charge fully, let go only when lined up; never let the server auto-fire it
        if li is not None:
            lw = weapons[li]
            charge = lw.get("charge", 0.0)
            full = lw["stats"].get("charge", 0.75)
            t_now = tick * 0.05
            if charge <= 0:
                self.full_since = None
            elif charge >= full * LASER_FULL and self.full_since is None:
                self.full_since = t_now
            if charge > 0 or (lw["cooldown"] <= 0.05 and clear and d <= 250):
                # hitscan, fired at release (next decision): aim where it will be in ~0.1 s
                lt = (tc[0] + vel[0] * 0.1, tc[1] + vel[1] * 0.1, tc[2])
                aim = direct_aim(muzzle, lt)
                thr = clampf(math.atan2(1.6, d), 0.006, 0.06)
                err = max(abs(wrap(aim[0] - cur[0])), abs(aim[1] - cur[1]))
                if charge > 0:
                    held = t_now - self.full_since if self.full_since is not None else 0.0
                    ready = charge >= full * LASER_FULL
                    release = (clear and ready and err < thr) or held >= LASER_SAFE_HOLD
                    fire[li] = not release
                elif err < 0.35:
                    fire[li] = True
                if fire[li] or charge > 0:
                    return aim, fire
        # mortar: lob at 30-200 m, never under 30 m (7 m splash hurts me at 50%)
        if mi is not None and 30 <= d <= 200 and (not clear or d > 60):
            mw = weapons[mi]
            src = (muzzle[0], muzzle[1], muzzle[2] + 2.5)
            sol = mortar_aim(src, tc, vel, acc, v=mw["stats"].get("speed", 72.0), g=22.0)
            if sol:
                aim = (sol[0], sol[1])
                err = max(abs(wrap(aim[0] - cur[0])), abs(aim[1] - cur[1]))
                if err < 0.03 and mw["cooldown"] <= 0.05:
                    fire[mi] = True
                return aim, fire
        if gi is not None:
            gs = weapons[gi]
            aim = lead_aim(muzzle, tc, vel, acc, gs["stats"].get("speed", 252.0))
            thr = clampf(math.atan2(3.0, d), 0.02, 0.15)
            err = max(abs(wrap(aim[0] - cur[0])), abs(aim[1] - cur[1]))
            rng = min(gs["stats"].get("range", 230.0), gun_range)
            if clear and err < thr and d <= rng and gs.get("ammo", 1) > 0 and gs.get("reload", 0) <= 0:
                fire[gi] = True
        if ni is not None and d <= 42 and me["energy"] >= 5 and not fire[ni]:
            if abs(wrap(direct_aim(muzzle, tc)[0] - cur[0])) < 0.2:
                fire[ni] = True
        return aim, fire


mf.run(Bot)
