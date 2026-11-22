"""Target ranking for the mech bot.  LOWER score = kill first.

score = W_DIST*dist_n + W_HP*hp_n - W_THREAT*threat_n - W_OPP*opp_n  (+ NO_LOS_PENALTY if no clean shot)

All four features are scaled to 0..1 so the weights are directly comparable.
"""
import math

TAU = 2 * math.pi

# tuned offline with tune_weights.py (see its docstring for what the oracle assumes)
WEIGHTS = {"dist": 0.35, "hp": 0.40, "threat": 0.15, "opp": 0.10}
NO_LOS_PENALTY = 2.0

# rough per-weapon danger to *me* when it is in range (0..1 scale)
WEAPON_THREAT = {"gun": 0.30, "laser": 0.90, "lightning": 0.75, "mortar": 0.60}
WEAPON_RANGE = {"gun": 230.0, "laser": 280.0, "lightning": 45.0, "mortar": 235.0}
HP_REF = 1960.0   # 1400 base * 1.4 (10 cubes)


def wrap(a):
    return (a + math.pi) % TAU - math.pi


def threat_n(weapons, d, facing_err, charging_at_me=False):
    """How dangerous this enemy is to me right now. facing_err = angle between its aim and the line to me."""
    s = 0.0
    for w in weapons or ("gun",):
        if d <= WEAPON_RANGE.get(w, 100.0) and not (w == "mortar" and d < 10):
            s += WEAPON_THREAT.get(w, 0.3)
    f = 0.35 + 0.65 * max(0.0, 1.0 - facing_err / math.pi)      # pointing away -> less dangerous
    s *= f
    if charging_at_me:
        s += 0.5
    return min(1.0, s)


def opp_n(d, facing_err, turn_needed, in_range):
    """How good a moment this is to shoot it: its back to me (x1.5 rear damage), little turning, in range."""
    rear = 1.0 if facing_err > math.pi / 2 else facing_err / (math.pi / 2) * 0.5
    align = max(0.0, 1.0 - turn_needed / math.pi)
    rng = 1.0 if in_range else max(0.0, 1.0 - (d - 170.0) / 100.0)
    return 0.4 * rear + 0.3 * align + 0.3 * rng


def score(d, hp, weapons, facing_err, turn_needed, clear, charging_at_me=False, w=None, gun_range=170.0):
    w = w or WEIGHTS
    d_n = min(1.0, d / 200.0)
    h_n = min(1.0, hp / HP_REF) if hp is not None else 0.5
    t_n = threat_n(weapons, d, facing_err, charging_at_me)
    o_n = opp_n(d, facing_err, turn_needed, d <= gun_range)
    s = w["dist"] * d_n + w["hp"] * h_n - w["threat"] * t_n - w["opp"] * o_n
    return s + (0.0 if clear else NO_LOS_PENALTY)
