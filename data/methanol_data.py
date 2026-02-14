import numpy as np
from scipy.optimize import least_squares

# Propiedades críticas del metanol
Tc = 512.6     # K
Pc = 8097.0    # kPa

# Datos experimentales (Pa → kPa)
T_exp = np.array([
    274.15, 278.15, 283.15, 293.15, 303.15, 313.15, 323.15,
    333.15, 343.15, 353.15, 363.15, 373.15, 383.15, 393.15,
    403.15, 413.15, 423.15, 433.15, 443.15, 453.15, 463.15, 468.67
])

P_exp = np.array([
    4301, 5508, 7419, 13019, 21899, 35470, 55598,
    84584, 125407, 180995, 255362, 353752, 480418, 641127,
    841997, 1090301, 1393392, 1758832, 2194795, 2709810,
    3312704, 3686440
]) / 1000.0  # kPa


def wagner_pressure(T, params, Tc, Pc):
    A, B, C, D = params
    tau = 1.0 - T / Tc

    exponent = (
        A * tau
        + B * tau**1.5
        + C * tau**3
        + D * tau**6
    ) / (1.0 - tau)

    return Pc * np.exp(exponent)

def residuals_wagner(params, T, P_exp, Tc, Pc):
    P_calc = wagner_pressure(T, params, Tc, Pc)
    return np.log(P_calc / Pc) - np.log(P_exp / Pc)

# Estimación inicial (razonable)
theta0 = np.array([-7.0, 1.5, -10.0, 20.0])

res = least_squares(
    residuals_wagner,
    theta0,
    args=(T_exp, P_exp, Tc, Pc),
    bounds=([-20, -10, -50, -50], [10, 10, 50, 50]),
    xtol=1e-12,
    ftol=1e-12,
    gtol=1e-12
)

if __name__ == "__main__":
    A, B, C, D = res.x

    print("Wagner coefficients (Methanol):")
    print(f"A = {A:.6f}")
    print(f"B = {B:.6f}")
    print(f"C = {C:.6f}")
    print(f"D = {D:.6f}")