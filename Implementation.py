import numpy as np, cvxpy as cp

def one_shot(enemies, mates, terrain, L, R_blast, R_friend,
             lam=1.0, sigma=3.0, dx=0.25):
    for theta in scan_angles(enemies):                 # Stage 0
        U, V = rotate(enemies, theta)
        lo, hi = corridor(rotate_grid(terrain, -theta), L, dx)
        guide, sides = dijkstra_guide(lo, hi, U, V, mates, theta)   # Stage 1
        if guide is None:
            continue

        u = np.arange(0, L, dx)
        b = 9.0 / max(np.min(np.diff(np.sort(U)))**2, 1e-6)
        C = np.concatenate([U, np.linspace(0, L, 12)])              # centers
        P   = np.exp(-b * (u[:, None] - C)**2)                      # Phi
        Pdd = P * (4*b*b*(u[:, None]-C)**2 - 2*b)                   # Phi''

        w = cp.Variable(len(C))
        Q = dx * Pdd.T @ Pdd
        obj = cp.Minimize(cp.quad_form(w, cp.psd_wrap(Q))
                          + lam * cp.sum_squares(P @ w - guide))

        m  = 0.5                                                    # provisional margin
        cons = [np.exp(-b*C**2) @ w == 0,
                P @ w >= lo + m,
                P @ w <= hi - m]
        for ui, vi in zip(U, V):                                    # enemies
            phi = np.exp(-b*(ui - C)**2)
            cons += [cp.abs(phi @ w - vi) <= R_blast - m]
        for (uk, vk), s in zip(rotate(mates, theta), sides):        # friends
            win = u[np.abs(u - uk) <= R_friend + sigma]
            Pw  = np.exp(-b*(win[:, None] - C)**2)
            cons += [s * (Pw @ w - vk) >= R_friend + sigma]

        prob = cp.Problem(obj, cons)
        prob.solve(solver=cp.OSQP)
        if prob.status != cp.OPTIMAL:
            continue

        M = np.abs(w.value).sum() * 2*b                             # Stage 3
        if M * dx**2 / 8 > m:
            dx /= 2; continue
        return theta, to_graphwar(w.value, C, b)

    return degrade(enemies, mates, terrain, L)                      # Stage 4


def to_graphwar(w, C, b):
    return " + ".join(f"({wi:.4f})*e^(-{b:.4f}*(x-{ci:.3f})^2)"
                      for wi, ci in zip(w, C) if abs(wi) > 1e-3)