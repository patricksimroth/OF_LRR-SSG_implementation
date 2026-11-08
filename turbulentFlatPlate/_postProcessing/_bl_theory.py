"""Turbulent flat-plate boundary-layer thickness: correlations and a composite-profile
momentum-integral solution.

Composite profile (inner + outer):
    u+ = u+_Spalding(y+) + (1/kappa) [ Pi (1 - cos(pi eta)) + eta^2 - eta^3 ],  eta = y/delta
Spalding's law of the wall covers sublayer, buffer and log layer; the outer part is
Coles' wake with Granville's (1976) eta^2 - eta^3 term, which makes du/dy = 0 at
y = delta so the profile meets the freestream smoothly (plain Coles leaves a kink, so
delta99 would sit unrealistically close to delta).  Edge: u_e+ = (1/kappa) ln delta+
+ B + 2 Pi / kappa, Cf = 2 / u_e+^2.

Zero pressure gradient momentum integral dtheta/dx = Cf / 2, turbulent from x = 0:
    Re_x = integral (2 / Cf) dRe_theta
with the profile parametrised by delta+.  Pi(Re_theta) from the Cebeci & Smith (1974)
fit to Coles' data, or held fixed.
"""
import numpy as np

KAPPA = 0.41
B = 5.0


_SPALDING = {}


def spalding_uplus(yplus, kappa=KAPPA, b=B):
    """Invert Spalding's y+(u+) on a fine u+ grid (cached per kappa, B)."""
    if (kappa, b) not in _SPALDING:
        up = np.linspace(0.0, 60.0, 60001)
        ku = kappa * up
        yp = up + np.exp(-kappa * b) * (np.exp(ku) - 1 - ku - ku ** 2 / 2 - ku ** 3 / 6)
        _SPALDING[(kappa, b)] = (yp, up)
    yp, up = _SPALDING[(kappa, b)]
    return np.interp(yplus, yp, up)


def pi_cebeci_smith(re_theta):
    z = np.maximum(re_theta / 425.0 - 1.0, 0.0)
    return 0.55 * (1.0 - np.exp(-0.243 * np.sqrt(z) - 0.298 * z))


def profile(delta_plus, pi, kappa=KAPPA, b=B, n=4000):
    """eta grid, u/u_e, u_e+ for one delta+."""
    eta = np.concatenate([np.geomspace(1e-7, 1e-2, n // 2, endpoint=False),
                          np.linspace(1e-2, 1.0, n // 2)])
    up = spalding_uplus(eta * delta_plus, kappa, b) + (
        pi * (1 - np.cos(np.pi * eta)) + eta ** 2 - eta ** 3) / kappa
    ue_plus = up[-1]
    return eta, up / ue_plus, ue_plus


def thicknesses(delta_plus, pi, kappa=KAPPA, b=B):
    """(theta/delta, deltastar/delta, delta99/delta, u_e+)."""
    eta, u, ue_plus = profile(delta_plus, pi, kappa, b)
    eta = np.concatenate([[0.0], eta])
    u = np.concatenate([[0.0], u])
    theta = np.trapezoid(u * (1 - u), eta)
    dstar = np.trapezoid(1 - u, eta)
    j = int(np.argmax(u >= 0.99))
    d99 = eta[j - 1] + (0.99 - u[j - 1]) * (eta[j] - eta[j - 1]) / (u[j] - u[j - 1])
    return theta, dstar, d99, ue_plus


_SOLUTIONS = {}


def solve_plate(pi=None, kappa=KAPPA, b=B, delta_plus_range=(30.0, 2e4), n=300):
    key = (pi, kappa, b, delta_plus_range, n)
    if key not in _SOLUTIONS:
        _SOLUTIONS[key] = _solve_plate(pi, kappa, b, delta_plus_range, n)
    return _SOLUTIONS[key]


def _solve_plate(pi, kappa, b, delta_plus_range, n):
    """Arrays over delta+: Re_x, Re_theta, Re_delta99, Cf, H, Pi.  pi=None -> Cebeci-Smith."""
    dps = np.geomspace(*delta_plus_range, n)
    rows = []
    pi_k = 0.55 if pi is None else pi
    for dp in dps:
        for _ in range(30):
            th, ds, d99, uep = thicknesses(dp, pi_k, kappa, b)
            re_theta = th * dp * uep
            new = pi if pi is not None else float(pi_cebeci_smith(re_theta))
            if abs(new - pi_k) < 1e-6:
                break
            pi_k = new
        rows.append((re_theta, d99 * dp * uep, 2.0 / uep ** 2, ds / th, pi_k))
    re_theta, re_d99, cf, h, pis = map(np.array, zip(*rows))
    integrand = 2.0 / cf
    re_x = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1])
                                             * np.diff(re_theta))])
    re_x += re_theta[0] * integrand[0]
    return {"re_x": re_x, "re_theta": re_theta, "re_d99": re_d99, "cf": cf, "H": h, "Pi": pis}


def delta99_composite(x, re_per_m, pi=None):
    sol = solve_plate(pi)
    return np.interp(x * re_per_m, sol["re_x"], sol["re_d99"]) / re_per_m


def cf_composite(x, re_per_m, pi=None):
    sol = solve_plate(pi)
    return np.interp(x * re_per_m, sol["re_x"], sol["cf"])


def delta_power_law(x, re_per_m):
    """Prandtl 1/7 power law: the thickness where u = U_e EXACTLY."""
    return 0.37 * x / (re_per_m * x) ** 0.2


def delta99_power_law(x, re_per_m):
    """Same profile, 99 % point: (0.99)^7 = 0.932 of it."""
    return 0.99 ** 7 * delta_power_law(x, re_per_m)


def delta_white(x, re_per_m):
    """White, Fluid Mechanics: delta/x = 0.16 / Re_x^(1/7)."""
    return 0.16 * x / (re_per_m * x) ** (1.0 / 7.0)


if __name__ == "__main__":
    re_m = 69.4 / 1.38801e-5
    for x in (0.97008, 1.0, 1.90334):
        sol = solve_plate()
        rt = np.interp(x * re_m, sol["re_x"], sol["re_theta"])
        print(f"x={x}: 0.37 law {1e3*delta_power_law(x, re_m):.2f}  "
              f"1/7 delta99 {1e3*delta99_power_law(x, re_m):.2f}  "
              f"White {1e3*delta_white(x, re_m):.2f}  "
              f"composite(CS Pi) {1e3*delta99_composite(x, re_m):.2f} "
              f"(Re_theta {rt:.0f}, Cf {np.interp(x*re_m, sol['re_x'], sol['cf']):.5f}, "
              f"Pi {np.interp(x*re_m, sol['re_x'], sol['Pi']):.3f}, "
              f"theta {1e3*rt/re_m:.3f} mm)  "
              f"composite(Pi=0.55) {1e3*delta99_composite(x, re_m, 0.55):.2f}  "
              f"composite(Pi=0.45) {1e3*delta99_composite(x, re_m, 0.45):.2f}")
