"""Survivor bot v2 for mechfight/1.

Goal: placement, not kills.  Changes from v1 (see review):
  * decides whether a fight is worth it (targeting.should_engage) before shooting
  * ranks targets (targeting.rank_targets) instead of "nearest with LOS"
  * cover = precomputed spots hugging building walls; once there it HOLDS (silent), no strafing out
  * zone: leaves for the next circle based on travel time, not a fixed 20 s
  * lead aim = constant velocity + half-weighted measured acceleration + 1-tick latency
  * laser releases only on target; lightning only when the enemy is the nearest thing in the cone
  * dodges laser telegraphs and mortar landings, blends bullet dodges instead of overriding
  * terrain-aware line of sight and steering; reads numbers from hello.config
  * picks up power cubes when safe; exact upgrade ids
  * hard time guard: optional work is skipped once ~30 ms have been used
Standard library only.
"""
import array
import json
import math
import os
import time

import mechfight as mf
import targeting as TG
import sound_locator as SL

TAU = 2 * math.pi
BUDGET_S = 0.030           # soft per-decision compute budget (server limit is 50 ms)

# exact ids from hello.config.upgrades; earlier = preferred.  Survival first.
UPGRADE_PRIORITY = ["plating", "nanites", "armor", "sensors", "gyros", "gun_rate", "gun_damage",
                    "laser_charge", "laser_damage", "lightning_efficiency", "capacitor", "reactor",
                    "jets", "servos", "gun_mag", "gun_velocity", "lightning_range", "lightning_chain",
                    "laser_pierce", "mortar_reload", "mortar_velocity", "mortar_splash"]


def wrap(a):
    return (a + math.pi) % TAU - math.pi


def clampf(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def lead_aim(src, pos, vel, acc, speed, latency, vmax):
    """(yaw, pitch) to hit a target moving with vel + 0.5*acc (damped), fired `latency` s from now."""
    rx = pos[0] + vel[0] * latency - src[0]
    ry = pos[1] + vel[1] * latency - src[1]
    rz = pos[2] + vel[2] * latency - src[2]
    t = math.sqrt(rx * rx + ry * ry + rz * rz) / speed
    px = py = pz = 0.0
    for _ in range(4):                       # fixed-point iteration converges in 2-3 steps
        dx = vel[0] * t + 0.25 * acc[0] * t * t
        dy = vel[1] * t + 0.25 * acc[1] * t * t
        dn = math.hypot(dx, dy)
        if dn > vmax * t > 0:
            dx, dy = dx * vmax * t / dn, dy * vmax * t / dn
        px, py, pz = rx + dx, ry + dy, rz + vel[2] * t
        t = math.sqrt(px * px + py * py + pz * pz) / speed
    return math.atan2(py, px), math.atan2(pz, math.hypot(px, py)), t


def direct_aim(src, pos):
    dx, dy, dz = pos[0] - src[0], pos[1] - src[1], pos[2] - src[2]
    return math.atan2(dy, dx), math.atan2(dz, math.hypot(dx, dy))


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

    def __init__(self, map_dir):
        self.prisms = []
        self.grid = {}
        self.cover_pts = {}          # cell -> [(x, y, ground_z)]
        self.h = None
        try:
            with open(os.path.join(map_dir, "v2", "world.json")) as f:
                w = json.load(f)
            self._load_terrain(map_dir, w.get("terrain", {}))
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
                self.prisms.append((float(p["base"]), float(p["top"]), ring, holes,
                                    min(xs), min(ys), max(xs), max(ys), p.get("kind")))
                c = self.CELL
                for cx in range(int(min(xs) // c), int(max(xs) // c) + 1):
                    for cy in range(int(min(ys) // c), int(max(ys) // c) + 1):
                        self.grid.setdefault((cx, cy), []).append(idx)
            self._build_cover()
        except Exception as e:  # no map -> open field
            mf.log("world load failed:", repr(e))

    # ----- terrain -----
    def _load_terrain(self, map_dir, T):
        try:
            h = array.array("h")
            with open(os.path.join(map_dir, "v2", T.get("file", "terrain.i16")), "rb") as f:
                h.frombytes(f.read())
            if len(h) != T["nx"] * T["ny"]:
                return
            self.h, self.nx, self.ny = h, T["nx"], T["ny"]
            self.tx0, self.ty0, self.tstep, self.tscale = T["minX"], T["minY"], T["step"], T.get("scale", 0.01)
        except Exception as e:
            mf.log("terrain load failed:", repr(e))

    def ground(self, x, y):
        if self.h is None:
            return 0.0
        fx = (x - self.tx0) / self.tstep
        fy = (y - self.ty0) / self.tstep
        i = int(fx); j = int(fy)
        if i < 0: i = 0
        elif i > self.nx - 2: i = self.nx - 2
        if j < 0: j = 0
        elif j > self.ny - 2: j = self.ny - 2
        tx, ty = fx - i, fy - j
        tx = 0.0 if tx < 0 else 1.0 if tx > 1 else tx
        ty = 0.0 if ty < 0 else 1.0 if ty > 1 else ty
        h, nx = self.h, self.nx
        k = j * nx + i
        a, b, c, d = h[k], h[k + 1], h[k + nx], h[k + nx + 1]
        return ((a + (b - a) * tx) * (1 - ty) + (c + (d - c) * tx) * ty) * self.tscale

    # ----- solids -----
    def hit(self, x, y, zlo, zhi):
        """Is any solid prism present at (x, y) overlapping the height band [zlo, zhi]?"""
        for i in self.grid.get((int(x // self.CELL), int(y // self.CELL)), ()):
            base, top, ring, holes, x0, y0, x1, y1, _ = self.prisms[i]
            if base > zhi or top < zlo or x < x0 or x > x1 or y < y0 or y > y1:
                continue
            if _pip(x, y, ring) and not any(_pip(x, y, h) for h in holes):
                return True
        return False

    def los(self, a, b, step=3.0):
        dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        n = max(1, int(math.sqrt(dx * dx + dy * dy + dz * dz) / step))
        for i in range(1, n):
            f = i / n
            px, py, z = a[0] + dx * f, a[1] + dy * f, a[2] + dz * f
            if z < self.ground(px, py) or self.hit(px, py, z, z):
                return False
        return True

    def blocked(self, x, y, z, px, py, dist):
        """Can a mech standing at height z not walk to (px, py) `dist` m away?"""
        g = self.ground(px, py)
        if g - z > max(1.2, dist * 1.19):        # > step-up and steeper than 50 deg
            return True
        zb = max(z, g)
        return self.hit(px, py, zb + 1.2, zb + 9.0)

    # ----- precomputed cover: points 4.5 m outside the walls of buildings taller than a mech -----
    def _build_cover(self):
        n = 0
        for base, top, ring, holes, *_rest in self.prisms:
            kind = _rest[-1]
            if kind == "bridge":
                continue
            area = sum(ring[i][0] * ring[i - 1][1] - ring[i - 1][0] * ring[i][1] for i in range(len(ring)))
            sgn = 1.0 if area < 0 else -1.0           # outward normal side
            for i in range(len(ring)):
                (x1, y1), (x2, y2) = ring[i - 1], ring[i]
                L = math.hypot(x2 - x1, y2 - y1)
                if L < 4:
                    continue
                nxv, nyv = sgn * (y2 - y1) / L, -sgn * (x2 - x1) / L
                for k in range(int(L // 8) + 1):
                    f = (k + 0.5) / (int(L // 8) + 1)
                    px, py = x1 + (x2 - x1) * f + nxv * 4.5, y1 + (y2 - y1) * f + nyv * 4.5
                    g = self.ground(px, py)
                    if top - g < 10 or base > g + 1:        # wall must hide a 9 m mech
                        continue
                    if self.hit(px, py, g + 0.5, g + 9):
                        continue
                    self.cover_pts.setdefault((int(px // self.CELL), int(py // self.CELL)), []).append((px, py, g))
                    n += 1
        mf.log("cover points:", n)

    def cover_near(self, x, y, r):
        c = self.CELL
        out = []
        for cx in range(int((x - r) // c), int((x + r) // c) + 1):
            for cy in range(int((y - r) // c), int((y + r) // c) + 1):
                for p in self.cover_pts.get((cx, cy), ()):
                    d = math.hypot(p[0] - x, p[1] - y)
                    if d <= r:
                        out.append((d, p))
        out.sort(key=lambda e: e[0])
        return out


class Bot:
    name = "survivor-v2"

    def __init__(self, hello):
        cfg = hello.get("config", {})
        self.cfg = cfg
        mech = cfg.get("mech", {})
        snd = cfg.get("sound", {})
        self.walk = mech.get("walk_speed", 11.7)
        self.quiet = 0.9 * snd.get("step_speed", 5.0) / self.walk     # move fraction below step noise
        self.pick_r = cfg.get("loot", {}).get("pickup_radius", 4.0) - 0.5
        self.cube_r = cfg.get("cubes", {}).get("pickup_radius", 3.5)
        self.lt = cfg.get("weapons", {}).get("lightning", {})
        self.laser = cfg.get("weapons", {}).get("laser", {})
        self.dt_dec = cfg.get("match", {}).get("decide_every", 2) / cfg.get("match", {}).get("tick_hz", 20)
        self.world = World(hello.get("map", {}).get("dir", "/arena"))
        self.sounds = SL.SoundTracker(SL.build_table(snd))
        self.noise = snd.get("bearing_noise_deg", 7.0)
        self.pref = 1
        self.bad = set()
        self.tb = None               # (bearing, time) of last threat
        self.last_hit_by_mech = -99.0
        self.cover = None            # (time, (x, y) or None)
        self.stuck = 0
        self.prev = {}               # enemy id -> (t, vel) for acceleration estimate
        self.memory = {}             # enemy id -> (t, pos, weapons)
        self.charge_t = 0.0
        mf.log("bot ready; prisms:", len(self.world.prisms))

    # ---------- helpers ----------
    def read_events(self, obs, t, me_xy):
        for ev in obs.get("events", []):
            ty = ev.get("type")
            if ty == "take_failed":
                self.bad.add(ev["id"])
            elif ty == "damaged" and ev.get("bearing") is not None:
                self.tb = (ev["bearing"], t)
                if ev.get("weapon") != "npc":
                    self.last_hit_by_mech = t
        self.sounds.update(obs.get("sounds", []), me_xy, t, self.noise)
        tr = self.sounds.most_dangerous(t)
        if tr and (self.tb is None or self.tb[1] < t - 1.0):
            self.tb = (tr["bearing"], t)

    def pick_upgrade(self, offer):
        best, bi = 99, 0
        for i, ch in enumerate(offer["choices"]):
            r = UPGRADE_PRIORITY.index(ch["id"]) if ch["id"] in UPGRADE_PRIORITY else 50
            if r < best:
                best, bi = r, i
        return bi

    def region(self, obs, x, y):
        """Circle to stay inside.  The next circle is always inside the current one, so the
        'intersection of both' IS the next circle; the only question is when to start moving."""
        z = obs["zone"]
        t = obs["t"]
        nc, nr = z["next_center"], z["next_radius"]
        d_next = max(0.0, math.hypot(x - nc[0], y - nc[1]) - nr * 0.85)
        travel = d_next / (self.walk * self.quiet) + 10.0          # walking quietly + slack
        if t >= z["shrink_start"] - travel:
            c, r = nc, nr
        else:
            c, r = z["center"], z["radius"]
        return c, max(r - max(10.0, 0.12 * r), 5.0)

    def steer(self, x, y, z, tx, ty):
        base = math.atan2(ty - y, tx - x)
        W = self.world
        for off in (0, 0.4, 0.9, 1.4, 2.0, 2.6, 3.14):
            for sgn in ((1,) if off in (0, 3.14) else (self.pref, -self.pref)):
                a = base + off * sgn
                ca, sa = math.cos(a), math.sin(a)
                if not (W.blocked(x, y, z, x + 4 * ca, y + 4 * sa, 4)
                        or W.blocked(x, y, z, x + 9 * ca, y + 9 * sa, 9)):
                    if off and off != 3.14:
                        self.pref = sgn
                    return ca, sa
        return math.cos(base), math.sin(base)

    def find_cover(self, x, y, threats, rc, rr, deadline):
        W = self.world
        ts = [(e["pos"][0], e["pos"][1], e["pos"][2] + 8.0) for e in threats[:3]]
        for d, (px, py, g) in W.cover_near(x, y, 45.0)[:40]:
            if math.hypot(px - rc[0], py - rc[1]) > rr:
                continue
            ok = True
            for tx, ty, tz in ts:
                if W.los((tx, ty, tz), (px, py, g + 4.5)) or W.los((tx, ty, tz), (px, py, g + 8.5)):
                    ok = False
                    break
            if ok:
                return (px, py)
            if time.perf_counter() > deadline:
                break
        return None

    def dodge(self, obs, x, y, z, enemies):
        """(dx, dy, urgent) away from the most urgent danger, or None."""
        best = None
        cx, cy = x, y
        # 1) laser telegraph: a visible enemy charging and aiming at me
        for m in enemies:
            if m.get("charging"):
                d = math.dist(m["pos"][:2], (x, y))
                if d < 300 and TG.facing_angle(m, (x, y)) < math.atan2(4.0, max(d, 1)) + 0.04:
                    b = math.atan2(y - m["pos"][1], x - m["pos"][0])
                    s = self.pref
                    return (-math.sin(b) * s, math.cos(b) * s, True)
        for p in obs.get("projectiles", []):
            px, py, pz = p["pos"]
            vx, vy, vz = p["vel"]
            if p["kind"] == "mortar":
                g = self.cfg.get("weapons", {}).get("mortar", {}).get("gravity", 22.0)
                disc = vz * vz + 2 * g * (pz - z)
                if disc < 0:
                    continue
                tl = (vz + math.sqrt(disc)) / g
                if not 0 < tl < 3:
                    continue
                lx, ly = px + vx * tl, py + vy * tl
                d = math.hypot(lx - x, ly - y)
                if d < 11:
                    ax, ay = x - lx, y - ly
                    n = math.hypot(ax, ay) or 1.0
                    return (ax / n, ay / n, True)
            else:
                v2 = vx * vx + vy * vy
                if v2 < 1:
                    continue
                rx, ry = px - cx, py - cy
                tc = -(rx * vx + ry * vy) / v2
                if not 0 < tc < 0.6:
                    continue
                mx, my = rx + vx * tc, ry + vy * tc
                if math.hypot(mx, my) < 4.0 and (best is None or tc < best[0]):
                    n = math.sqrt(v2)
                    pxn, pyn = -vy / n, vx / n
                    side = -1 if (mx * pxn + my * pyn) > 0 else 1
                    best = (tc, pxn * side, pyn * side)
        return (best[1], best[2], False) if best else None

    def lightning_clear(self, obs, me_xy, target, cur_yaw):
        """Lightning hits the NEAREST thing in its cone: make sure that is the enemy."""
        rng = self.lt.get("range", 45.0)
        cone = math.radians(self.lt.get("cone_deg", 30.0))
        td = math.dist(target["pos"][:2], me_xy)
        if td > rng - 3:
            return False
        for o in list(obs.get("mechs", [])) + list(obs.get("crates", [])):
            if o is target:
                continue
            od = math.dist(o["pos"][:2], me_xy)
            if od < td and abs(wrap(math.atan2(o["pos"][1] - me_xy[1], o["pos"][0] - me_xy[0]) - cur_yaw)) < cone:
                return False
        return True

    # ---------- main ----------
    def act(self, obs):
        t0 = time.perf_counter()
        deadline = t0 + BUDGET_S
        me = obs["self"]
        x, y, z = me["pos"]
        tick, t = obs["tick"], obs["t"]
        st = me.get("stats", {})
        eye_h = st.get("eye_height", 8.0)
        muz_h = st.get("muzzle_height", 7.0)
        act = {"tick": tick}
        self.read_events(obs, t, (x, y))
        if obs.get("offer"):
            act["pick"] = self.pick_upgrade(obs["offer"])

        weapons = me["weapons"]
        slot_of = {w["kind"]: i for i, w in enumerate(weapons) if w}
        gi, li, ni = slot_of.get("gun"), slot_of.get("laser"), slot_of.get("lightning")
        my_kinds = list(slot_of)

        # --- enemies: visibility, memory, acceleration ---
        mechs = [m for m in obs["mechs"] if m["kind"] == "mech"]
        npcs = [m for m in obs["mechs"] if m["kind"] == "npc"]
        acc = {}
        for m in mechs:
            pv = self.prev.get(m["id"])
            if pv and 0 < t - pv[0] < 0.25:
                acc[m["id"]] = ((m["vel"][0] - pv[1][0]) / (t - pv[0]), (m["vel"][1] - pv[1][1]) / (t - pv[0]))
            self.prev[m["id"]] = (t, m["vel"])
            self.memory[m["id"]] = (t, m["pos"], m.get("weapons", []))
        enemies = sorted(((math.dist(m["pos"][:2], (x, y)), m) for m in mechs), key=lambda e: e[0])
        info = []
        for d, m in enemies[:4]:
            tp = m["pos"]
            ch = 4.5 * (1 + 0.05 * m.get("cubes", 0))
            clear = self.world.los((x, y, z + eye_h), (tp[0], tp[1], tp[2] + ch))
            info.append((d, m, clear))
        threats = [m for d, m, c in info if d < 200]
        shooters = [m for d, m, c in info if c and d < 230]

        # --- fight or hide? ---
        target, engage, exp_dmg = None, False, 0.0
        if shooters:
            ranked = TG.rank_targets(shooters, me["pos"])
            ok, exp_dmg = TG.should_engage(ranked, me, my_kinds, margin=0.5)
            top = ranked[0][1]
            dtop = math.dist(top["pos"][:2], (x, y))
            finishing = top.get("hp", 1400) < 250 and dtop < 120
            cornered = t - self.last_hit_by_mech < 1.5
            engage = ok or finishing or cornered
            target = (dtop, top)

        rc, rr = self.region(obs, x, y)
        dist_c = math.hypot(x - rc[0], y - rc[1])
        inside = dist_c <= rr
        if inside:
            rp = (x, y)
        else:
            k = rr / dist_c
            rp = (rc[0] + (x - rc[0]) * k, rc[1] + (y - rc[1]) * k)

        mode = "hold"
        mv = (0.0, 0.0)
        goal = None

        if target and engage:
            self.cover = None
        if target and not engage:
            # break line of sight; stay in cover once there
            if self.cover is None or t - self.cover[0] >= 0.5:
                if time.perf_counter() < deadline:
                    self.cover = (t, self.find_cover(x, y, threats, rc, rr, deadline))
            cv = self.cover[1] if self.cover else None
            if cv and math.dist(cv, (x, y)) > 2:
                mode, goal = "cover", cv
                mv = self.steer(x, y, z, cv[0], cv[1])
            elif cv:
                mode = "incover"
            else:
                d, m = target            # no cover: back away from the threat, toward the zone
                ax, ay = x - m["pos"][0], y - m["pos"][1]
                n = math.hypot(ax, ay) or 1.0
                bx, by = (0.0, 0.0) if inside else self.steer(x, y, z, rp[0], rp[1])
                mode, mv = "retreat", self.steer(x, y, z, x + ax / n * 10 + bx * 10, y + ay / n * 10 + by * 10)
        if target and engage:
            d, m = target
            tp = m["pos"]
            bearing = math.atan2(tp[1] - y, tp[0] - x)
            side = 1 if ((tick // 10) + int(me["id"])) % 2 == 0 else -1
            px, py = -math.sin(bearing) * side, math.cos(bearing) * side
            if self.world.blocked(x, y, z, x + 4 * px, y + 4 * py, 4):
                px, py = -px, -py
            bx, by = (0.0, 0.0) if inside else self.steer(x, y, z, rp[0], rp[1])
            retreat = -0.4 if d < 40 and "lightning" in m.get("weapons", []) else 0.0
            vx = 0.8 * px + 0.5 * bx + retreat * math.cos(bearing)
            vy = 0.8 * py + 0.5 * by + retreat * math.sin(bearing)
            n = math.hypot(vx, vy) or 1.0
            mode, mv = "strafe", (vx / n, vy / n)
        if not target:
            self.cover = None
            if not inside:
                mode, goal = "zone", rp
                mv = self.steer(x, y, z, rp[0], rp[1])
            elif not threats:
                # cubes: free max HP, damage and upgrade offers; silent to collect
                cubes = [c for c in obs.get("cubes", [])
                         if math.hypot(c["pos"][0] - rc[0], c["pos"][1] - rc[1]) < rr
                         and abs(c["pos"][2] - z) < 4 and math.dist(c["pos"][:2], (x, y)) < 40]
                cubes.sort(key=lambda c: math.dist(c["pos"][:2], (x, y)))
                have_free = any(w is None for w in weapons)
                mortar_slot = slot_of.get("mortar")
                picks = [p for p in obs.get("pickups", [])
                         if p["id"] not in self.bad and p["kind"] in ("laser", "lightning")
                         and p["kind"] not in slot_of and (have_free or mortar_slot is not None)]
                picks.sort(key=lambda p: (p["kind"] != "laser", math.dist(p["pos"][:2], (x, y))))
                if picks and math.dist(picks[0]["pos"][:2], (x, y)) < self.pick_r and abs(picks[0]["pos"][2] - z) < 4:
                    p = picks[0]
                    act["take"] = p["id"]
                    if not have_free:
                        act["drop"] = mortar_slot
                elif cubes:
                    c = cubes[0]
                    mode, goal = "cube", c["pos"]
                    mv = self.steer(x, y, z, c["pos"][0], c["pos"][1])
                elif picks and math.dist(picks[0]["pos"][:2], (x, y)) < 40 and \
                        math.hypot(picks[0]["pos"][0] - rc[0], picks[0]["pos"][1] - rc[1]) < rr:
                    p = picks[0]
                    mode, goal = "loot", p["pos"]
                    mv = self.steer(x, y, z, p["pos"][0], p["pos"][1])

        # drones shoot mechs within 45 m: steer around them when not fighting
        if mode in ("zone", "cube", "loot", "retreat"):
            for n_ in npcs:
                dn = math.dist(n_["pos"][:2], (x, y))
                if dn < 50:
                    ax, ay = (x - n_["pos"][0]) / max(dn, 1), (y - n_["pos"][1]) / max(dn, 1)
                    w = (50 - dn) / 50
                    vx, vy = mv[0] + ax * w, mv[1] + ay * w
                    nn = math.hypot(vx, vy) or 1.0
                    mv = (vx / nn, vy / nn)

        # dodge: urgent ones override, bullets blend in
        dg = self.dodge(obs, x, y, z, mechs)
        if dg:
            dx, dy, urgent_dg = dg
            if self.world.blocked(x, y, z, x + 4 * dx, y + 4 * dy, 4):
                dx, dy = -dx, -dy
            if urgent_dg:
                mv, mode = (dx, dy), "dodge"
            else:
                vx, vy = mv[0] + 1.5 * dx, mv[1] + 1.5 * dy
                n = math.hypot(vx, vy) or 1.0
                mv = (vx / n, vy / n)
                if mode in ("hold", "incover"):
                    mode = "dodge"

        # speed: quiet unless urgent
        zone_urgent = (not me["in_zone"]) or (not inside and obs["t"] > obs["zone"]["shrink_start"] - 5)
        urgent = zone_urgent or mode in ("dodge", "cover", "strafe", "retreat")
        sc = 1.0 if urgent else self.quiet
        move = [mv[0] * sc, mv[1] * sc]
        if mode in ("hold", "incover"):
            move = [0.0, 0.0]
        act["move"] = move
        if mode in ("zone", "retreat", "dodge") and (not me["in_zone"] or mode != "zone") and me["energy"] > 25:
            act["boost"] = True

        # stuck detection
        spd = math.hypot(me["vel"][0], me["vel"][1])
        if mode in ("zone", "loot", "cover", "cube", "retreat") and spd < 1.5:
            self.stuck += 1
        else:
            self.stuck = 0
        if self.stuck >= 6:
            act["jump"] = True
            act["boost"] = True
            if self.stuck % 12 == 0:
                self.pref = -self.pref
            if self.stuck > 40:
                self.stuck = 0

        # --- aim & fire ---
        fire = [False, False]
        aim = None
        cur = me["aim"]
        muzzle = (x, y, z + muz_h)
        lw = weapons[li] if li is not None else None
        charging = lw is not None and lw.get("charge", 0) > 0
        if target and engage:
            d, m = target
            tp = m["pos"]
            ch = 4.5 * (1 + 0.05 * m.get("cubes", 0))
            tc = (tp[0], tp[1], tp[2] + ch)
            a = acc.get(m["id"], (0.0, 0.0))
            if gi is not None:
                gs = weapons[gi]
                spd_b = gs["stats"].get("speed", 252.0)
                yaw, pitch, ft = lead_aim(muzzle, tc, m["vel"], a, spd_b, self.dt_dec * 0.5, 19.5)
                aim = (yaw, pitch)
            else:
                aim = direct_aim(muzzle, tc)
            err = max(abs(wrap(aim[0] - cur[0])), abs(aim[1] - cur[1]))
            # laser: charge when roughly on target, release only when exactly on target
            if lw is not None and d <= self.laser.get("range", 280) - 20:
                laim = direct_aim(muzzle, (tc[0] + m["vel"][0] * self.dt_dec, tc[1] + m["vel"][1] * self.dt_dec, tc[2]))
                lerr = max(abs(wrap(laim[0] - cur[0])), abs(laim[1] - cur[1]))
                full = lw["stats"].get("charge", 0.75)
                thr = clampf(math.atan2(2.0, d), 0.008, 0.08)
                if charging:
                    aim = laim
                    fire[li] = not (lw["charge"] >= full * 0.95 and lerr < thr)   # let go = fire
                elif lw["cooldown"] <= 0.05 and lerr < 0.2:
                    aim = laim
                    fire[li] = True
            if gi is not None and not charging:
                gs = weapons[gi]
                thr = clampf(math.atan2(2.2, d), 0.012, 0.12)
                rng = min(gs["stats"].get("range", 230.0), 160.0)
                if err < thr and d <= rng and gs.get("ammo", 1) > 0 and gs.get("reload", 0) <= 0:
                    fire[gi] = True
            if ni is not None and me["energy"] >= 30 and self.lightning_clear(obs, (x, y), m, cur[0]):
                fire[ni] = True
        elif charging:
            # target lost mid-charge: keep holding (auto-fires only after hold_s) and keep aim
            fire[li] = True
            aim = (cur[0], cur[1])
        if aim is None:
            if threats:
                tp = threats[0]["pos"]
                aim = (math.atan2(tp[1] - y, tp[0] - x), 0.0)      # front arc toward the threat
                # lightning shorts a charging laser in range: use it even when not engaging
                if ni is not None and threats[0].get("charging") and me["energy"] >= 10 \
                        and self.lightning_clear(obs, (x, y), threats[0], cur[0]):
                    fire[ni] = True
            elif self.tb and t - self.tb[1] < 4.0:
                aim = (self.tb[0], 0.0)
            else:
                recent = [v for v in self.memory.values() if t - v[0] < 8.0]
                if recent:
                    v = min(recent, key=lambda v: math.dist(v[1][:2], (x, y)))
                    aim = (math.atan2(v[1][1] - y, v[1][0] - x), 0.0)
                elif mode in ("zone", "loot", "cube", "cover", "retreat") and (move[0] or move[1]):
                    sweep = 0.6 * math.sin(t * 1.5)             # look around while walking
                    aim = (math.atan2(move[1], move[0]) + sweep, 0.0)
                else:
                    aim = (wrap(cur[0] + 0.25), 0.0)            # slow scan ~2.5 rad/s

        act["aim"] = [aim[0], aim[1]]
        if fire[0] or fire[1]:
            act["fire"] = fire
        elif gi is not None:
            gs = weapons[gi]
            if not threats and gs.get("ammo", 99) < gs["stats"].get("mag", 12) * 0.6 and gs.get("reload", 0) <= 0:
                act["reload"] = True

        # --- debug ---
        if time.perf_counter() < deadline:
            mf.draw.label(f"{mode} hp{me['hp']:.0f} e{len(mechs)} {'FIGHT' if engage else ''} x{exp_dmg:.0f}")
            mf.draw.point([rc[0], rc[1], z], "#40c0ff", 5)
            if goal:
                mf.draw.point([goal[0], goal[1], z], "#60ff60", 6)
            if target:
                tp = target[1]["pos"]
                mf.draw.line([x, y, z + muz_h], [tp[0], tp[1], tp[2] + 4.5], "#ff5040" if engage else "#ffd060")
            for tr in self.sounds.tracks[:4]:
                e = tr["est"]
                mf.draw.text(SL.popup(e)[:80], [e["center"][0], e["center"][1], z + 5], "#c0a0ff")
        return act


if __name__ == "__main__":
    mf.run(Bot)
