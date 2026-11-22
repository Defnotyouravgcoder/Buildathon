"""Target ranking for mechfight/1.  Standard library only, ~10 microseconds per enemy.

rank_targets(...) gives every visible enemy mech a number; LOWER number = kill it FIRST.

    score = W_DIST * dist_n + W_HP * hp_n - W_THREAT * threat_n - W_OPP * opp_n

All four features are normalised to 0..1:
    dist_n    distance / 200 m                        (far   -> bigger number -> later)
    hp_n      enemy hp / 1400                          (tanky -> bigger number -> later)
    threat_n  damage/s it can do to ME at this range   (dangerous -> smaller number -> sooner)
    opp_n     attack opportunity: 1 = it is not facing me (rear hits x1.5, slow to answer),
              0 = it is aiming right at me

should_engage(...) answers the more important survivor question: is this fight worth it at all?
"""
import math

# weights tuned by tune_targeting.py (see the review for the numbers)
W_DIST, W_HP, W_THREAT, W_OPP = 0.25, 0.35, 0.40, 0.0

TAU = 2 * math.pi


def _wrap(a):
    return (a + math.pi) % TAU - math.pi


def hit_prob(d):
    """Rough chance a led gun bullet hits a dodging mech at distance d (m)."""
    return max(0.1, min(0.95, 1.0 - d / 250.0))


def weapon_dps(kind, d):
    """Damage/s one weapon of this kind does at range d (defaults; read real numbers from config)."""
    if kind == "gun":
        return 187.5 * hit_prob(d) if d <= 230 else 0.0
    if kind == "laser":
        return 130.0 if d <= 280 else 0.0
    if kind == "lightning":
        return 240.0 if d <= 45 else 0.0
    if kind == "mortar":
        return 90.0 if 15 <= d <= 235 else 0.0      # slow shell, often dodged
    return 0.0


def facing_angle(enemy, me_pos):
    """Angle between where the enemy aims and the direction from it to me (0 = aiming at me)."""
    ex, ey = enemy["pos"][0], enemy["pos"][1]
    to_me = math.atan2(me_pos[1] - ey, me_pos[0] - ex)
    return abs(_wrap(enemy["aim"][0] - to_me))


def opportunity(enemy, me_pos):
    a = facing_angle(enemy, me_pos)
    if a > math.pi / 2:
        return 1.0          # its back is to me: rear armor x1.5 and it must turn first
    if a > 0.5:
        return 0.5          # looking roughly elsewhere
    return 0.0              # looking at me


def features(enemy, me_pos):
    d = math.dist(enemy["pos"][:2], me_pos[:2])
    weps = enemy.get("weapons") or ["gun"]
    threat = sum(weapon_dps(k, d) for k in weps)
    if enemy.get("charging"):
        threat += 150.0     # a laser is about to fire
    hp = enemy.get("hp", 1400.0)
    return d, hp, threat, opportunity(enemy, me_pos)


def score(d, hp, threat, opp, w=None):
    wd, wh, wt, wo = w or (W_DIST, W_HP, W_THREAT, W_OPP)
    return (wd * min(d / 200.0, 1.0) + wh * min(hp / 1400.0, 1.5)
            - wt * min(threat / 400.0, 1.0) - wo * opp)


def rank_targets(mechs, me_pos, w=None):
    """[(score, enemy), ...] sorted, lowest score (= kill first) first. NPCs are skipped."""
    out = []
    for m in mechs:
        if m.get("kind") != "mech":
            continue
        out.append((score(*features(m, me_pos), w=w), m))
    out.sort(key=lambda e: e[0])
    return out


# ---------------------------------------------------------------- should I fight at all?
def my_dps(my_kinds, d, rear):
    return sum(weapon_dps(k, d) for k in my_kinds) * (1.5 if rear else 1.0)


def expected_damage(order, my_kinds, me_aim_target_bearing=None):
    """Very small fight model.  `order` = list of dicts with d, hp, threat, opp, bearing.
    Kill them one by one in that order; return damage I take meanwhile.
    Enemies already facing me shoot from t=0, others start after a reaction delay; my target
    takes rear damage until it has turned; enemies outside my front arc hit my back x1.5."""
    t = 0.0
    taken = 0.0
    alive = list(order)
    react = {id(e): (0.0 if e["opp"] == 0 else 0.4 if e["opp"] == 0.5 else 2.5) for e in order}
    for tgt in order:
        rate_rear = my_dps(my_kinds, tgt["d"], True)
        rate = my_dps(my_kinds, tgt["d"], False)
        if rate <= 0:
            return float("inf")
        turn_at = t + (0.0 if tgt["opp"] == 0 else 0.8)
        react[id(tgt)] = min(react[id(tgt)], turn_at)
        hp = tgt["hp"]
        rear_time = max(0.0, turn_at - t) if tgt["opp"] > 0 else 0.0
        if hp <= rate_rear * rear_time:
            dt = hp / rate_rear
        else:
            dt = rear_time + (hp - rate_rear * rear_time) / rate
        for e in alive:
            start = max(t, react[id(e)])
            if t + dt > start:
                rear = abs(_wrap(e["bearing"] - tgt["bearing"])) > math.pi / 2
                taken += e["threat"] * (1.5 if rear else 1.0) * (t + dt - start)
        t += dt
        alive.remove(tgt)
    return taken


def should_engage(ranked, me, my_kinds, margin=0.5):
    """Engage only if killing every visible enemy (in ranked order) costs < margin * my hp."""
    x, y = me["pos"][0], me["pos"][1]
    order = []
    for s, m in ranked:
        d, hp, threat, opp = features(m, me["pos"])
        order.append({"d": d, "hp": hp, "threat": threat, "opp": opp,
                      "bearing": math.atan2(m["pos"][1] - y, m["pos"][0] - x)})
    if not order:
        return False, 0.0
    dmg = expected_damage(order, my_kinds)
    return dmg < margin * me["hp"], dmg
