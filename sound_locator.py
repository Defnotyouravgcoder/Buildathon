"""Turn a heard sound into an estimated area on the map.  Standard library only.

A sound gives: kind, bearing (world yaw, Gaussian error sigma 7 deg), dist bucket
(near < 60 m, mid < 160 m, far beyond) and age.  Each kind also has a maximum carry range,
so  [bucket low, bucket high]  INTERSECT  [0, carry range]  is the true distance band.
Example: `step` + "near" -> 0..40 m (steps never carry past 40 m).  `gun` + "far" -> 160..320 m.
"""
import math

BUCKETS = {"near": (0.0, 60.0), "mid": (60.0, 160.0), "far": (160.0, 1e9)}

WHAT = {
    "step": "mech walking", "boost": "mech sprinting", "jet": "mech jetting (climbing / jumping)",
    "reload": "mech reloading its gun (vulnerable ~2.4 s)", "gun": "gun fire",
    "mortar": "mortar launch (shell incoming somewhere)", "explosion": "mortar impact",
    "laser": "laser fired", "laser_charge": "LASER CHARGING (beam in <0.75 s)",
    "lightning": "lightning (holder within 45 m of its target)", "jet_": "",
    "death": "a mech died (loot + cubes there)", "crate": "crate being shot (looter there)",
    "npc_gun": "drone firing (someone fighting a drone)", "drop": "supply drop landed",
}


def build_table(sound_cfg):
    """Precompute (kind, bucket) -> (lo, hi) once from hello.config.sound."""
    near, mid = sound_cfg.get("near", 60.0), sound_cfg.get("mid", 160.0)
    buckets = {"near": (0.0, near), "mid": (near, mid), "far": (mid, 1e9)}
    table = {}
    for kind, carry in sound_cfg.get("range", {}).items():
        for b, (lo, hi) in buckets.items():
            lo2, hi2 = lo, min(hi, carry)
            if lo2 < hi2:
                table[(kind, b)] = (lo2, hi2)
    return table


def locate(sound, me_xy, table, noise_deg=7.0):
    """Estimated source: dict with centre point, distance band, angular half-width, label."""
    lo, hi = table.get((sound["kind"], sound["dist"]), BUCKETS[sound["dist"]])
    if hi > 1e8:
        hi = lo + 200.0
    d = 0.5 * (lo + hi)
    b = sound["bearing"]
    half = math.radians(2 * noise_deg)            # ~95% of bearings within +-2 sigma
    cx, cy = me_xy[0] + d * math.cos(b), me_xy[1] + d * math.sin(b)
    width = 2 * hi * math.tan(half)               # sideways uncertainty at the far edge
    return {"center": (cx, cy), "d_lo": lo, "d_hi": hi, "d_est": d, "bearing": b,
            "half_angle": half, "width_m": width, "kind": sound["kind"],
            "what": WHAT.get(sound["kind"], sound["kind"]), "age": sound.get("age", 0.0)}


def popup(est):
    return f"{est['what']}: {est['d_lo']:.0f}-{est['d_hi']:.0f} m @ {math.degrees(est['bearing']):.0f} deg"


class SoundTracker:
    """Merges repeated sounds of the same kind from about the same direction into one track,
    narrowing the bearing by averaging (error shrinks ~1/sqrt(n))."""

    def __init__(self, table, keep_s=6.0):
        self.table = table
        self.keep = keep_s
        self.tracks = []        # dicts: kind, bearing_sum_xy, n, t, est

    def update(self, sounds, me_xy, t, noise_deg=7.0):
        for s in sounds:
            est = locate(s, me_xy, self.table, noise_deg)
            heard = t - s.get("age", 0.0)
            for tr in self.tracks:
                if tr["kind"] == s["kind"] and abs(_wrap(tr["bearing"] - s["bearing"])) < math.radians(20):
                    tr["sx"] += math.cos(s["bearing"]); tr["sy"] += math.sin(s["bearing"])
                    tr["n"] += 1
                    tr["bearing"] = math.atan2(tr["sy"], tr["sx"])
                    tr["t"] = heard
                    s2 = dict(s, bearing=tr["bearing"])
                    tr["est"] = locate(s2, me_xy, self.table, noise_deg / math.sqrt(tr["n"]))
                    break
            else:
                self.tracks.append({"kind": s["kind"], "bearing": s["bearing"], "sx": math.cos(s["bearing"]),
                                    "sy": math.sin(s["bearing"]), "n": 1, "t": heard, "est": est})
        self.tracks = [tr for tr in self.tracks if t - tr["t"] < self.keep]
        return self.tracks

    def most_dangerous(self, t):
        order = {"laser_charge": 0, "lightning": 1, "gun": 2, "laser": 2, "mortar": 3, "step": 4,
                 "boost": 4, "jet": 5, "reload": 6}
        best = None
        for tr in self.tracks:
            if tr["kind"] not in order or tr["est"]["d_lo"] > 160:
                continue
            k = (order[tr["kind"]], tr["est"]["d_est"])
            if best is None or k < best[0]:
                best = (k, tr)
        return best[1] if best else None


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


if __name__ == "__main__":
    cfg = {"near": 60, "mid": 160, "range": {"step": 40, "boost": 90, "reload": 60, "gun": 320, "mortar": 420,
           "explosion": 420, "laser_charge": 140, "laser": 380, "lightning": 160, "jet": 90, "death": 600}}
    tab = build_table(cfg)
    for s in [{"kind": "step", "dist": "near", "bearing": 0.5}, {"kind": "gun", "dist": "far", "bearing": 2.0},
              {"kind": "boost", "dist": "mid", "bearing": -1.0}, {"kind": "laser_charge", "dist": "mid", "bearing": 1.0},
              {"kind": "lightning", "dist": "mid", "bearing": 3.0}]:
        e = locate(s, (0, 0), tab)
        print(f"{s['kind']:13s} {s['dist']:5s} -> {popup(e)}, sideways +-{e['width_m']/2:.0f} m")
