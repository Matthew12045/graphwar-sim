import numpy as np, sympy as sp

X = sp.Symbol('x')

def compile_fn(s):
    expr = sp.sympify(s, locals={'x': X})          # whitelist symbols in production
    return sp.lambdify(X, expr, 'numpy')

def simulate(state, expr, dt=0.05, max_len=60):
    f = compile_fn(expr)
    sx, sy = state.shooter_pos; d = state.facing   # +1 or -1
    path = []
    for i in range(int(max_len / dt)):
        t = i * dt
        try: y = float(f(t))
        except Exception: return {"outcome": "domain_error", "at": t, "path": path}
        if not np.isfinite(y) or abs(y) > 1e4:
            return {"outcome": "diverged", "at": t, "path": path}
        p = (sx + d * t, sy + y); path.append(p)
        hit = state.collide(p, ignore=state.shooter_id)
        if hit: return {"outcome": hit.kind, "target": hit.id, "point": p, "path": path}
    return {"outcome": "out_of_range", "path": path}