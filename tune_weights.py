"""Offline weight tuning for targeting.score().   python3 tune_weights.py

ORACLE (my assumption, not the real game): the best target is the one that maximises
    (damage per second it would deal to me) / (time I need to kill it)
with time-to-kill = turn time + hp / (my gun dps * hit probability * rear bonus).
Random scenarios -> pick by oracle -> search weights so the cheap linear score agrees with it.
Metric: regret = 1 - oracle_value(chosen)/oracle_value(best), averaged (0 = perfect).
"""
import math, random
import targeting as T

def hit_p(d):  # gun hit probability falls with range
    return max(0.25, 0.95 - 0.0035 * d)

def enemy_dps(w, d, f_err):
    s = 0.0
    if "gun" in w: s += 45 / 0.24 * hit_p(d)
    if "laser" in w and d < 280: s += 315 / 2.4 * hit_p(d) * 0.6
    if "lightning" in w and d < 45: s += 60 / 0.25 * 0.9
    if "mortar" in w and 10 < d < 235: s += 200 / 2.0 * 0.4
    face = 0.35 + 0.65 * max(0, 1 - f_err / math.pi)
    return s * face

def oracle_value(e):
    my_dps = 45 / 0.24 * hit_p(e["d"]) * (1.5 if e["f_err"] > math.pi / 2 else 1.0)
    ttk = e["turn"] / 4.0 + e["hp"] / my_dps
    v = enemy_dps(e["w"], e["d"], e["f_err"]) / max(ttk, 0.2)
    return v if e["clear"] else v * 0.05

def scenario(r):
    es = []
    for _ in range(r.randint(2, 5)):
        w = ["gun"] + [k for k in ("laser", "lightning", "mortar") if r.random() < 0.3]
        es.append(dict(d=r.uniform(15, 200), hp=r.uniform(100, 1960), w=w, f_err=r.uniform(0, math.pi),
                       turn=r.uniform(0, math.pi), clear=r.random() < 0.8, chg=("laser" in w and r.random() < 0.2)))
    return es

def pick(es, wt):
    return min(range(len(es)), key=lambda i: T.score(es[i]["d"], es[i]["hp"], es[i]["w"], es[i]["f_err"],
                                                      es[i]["turn"], es[i]["clear"], es[i]["chg"], wt))

def evaluate(sc, wt):
    reg = agree = 0
    for es in sc:
        vals = [oracle_value(e) for e in es]; best = max(vals); i = pick(es, wt)
        reg += 1 - vals[i] / best if best > 0 else 0; agree += vals[i] == best
    return reg / len(sc), agree / len(sc)

def norm(w):
    s = sum(w.values()); return {k: v / s for k, v in w.items()}

if __name__ == "__main__":
    r = random.Random(7)
    train = [scenario(r) for _ in range(3000)]; test = [scenario(r) for _ in range(3000)]
    cands = {
        "closest only": dict(dist=1, hp=0, threat=0, opp=0),
        "lowest hp only": dict(dist=0, hp=1, threat=0, opp=0),
        "equal weights": dict(dist=.25, hp=.25, threat=.25, opp=.25),
        "your intuition (hp/opp heavy)": norm(dict(dist=.15, hp=.4, threat=.15, opp=.3)),
    }
    best = norm(dict(dist=.25, hp=.25, threat=.25, opp=.25)); bs = evaluate(train, best)[0]
    for it in range(1500):
        c = norm({k: max(0.0, v + r.gauss(0, 0.08)) + 1e-6 for k, v in best.items()})
        s = evaluate(train[:800], c)[0] if it % 1 == 0 else bs
        if s < evaluate(train[:800], best)[0]:
            best = c
    cands["TUNED"] = best
    for n, wt in cands.items():
        rg, ag = evaluate(test, wt)
        print(f"{n:32s} regret={rg:.3f} top1={ag:.3f}  " + " ".join(f"{k}={v:.2f}" for k, v in wt.items()))
