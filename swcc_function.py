import numpy as np

def campbell(psi, theta_s, psi_e, b):
    theta = np.zeros(len(psi))
    for i in range(len(psi)):
        if psi[i] < psi_e:
            theta[i] = theta_s
        else:
            theta[i] = theta_s * (psi[i] / psi_e) ** (-1 / b)
    return theta


def vg(psi, theta_s, theta_r, alpha, n):
    m = 1 - 1/n
    theta = np.zeros(len(psi))
    for i in range(len(psi)):
        S_e = 1 / (1 + (alpha * psi[i])**n)**m
        theta[i] = S_e * (theta_s - theta_r) + theta_r
    return theta


def ippisch_vg(psi, theta_s, theta_r, psi_e, alpha, n):
    theta = np.zeros(len(psi))
    m = 1 - 1 / n
    S_c = (1 + (alpha * psi_e) ** n) ** m
    for i in range(len(psi)):
        if psi[i] <= psi_e:
            S_e = 1
        else:
            S_e = S_c * (1 + (alpha * psi[i]) ** n) ** (-m)
    return theta

def campbell_ippisch_vg(psi, theta_s, theta_r, psi_e, alpha, n):
    theta = np.zeros(len(psi))
    m = 1 - 1 / n
    S_c = (1 + (alpha * psi_e) ** n) ** m
    for i in range(len(psi)):
        if psi[i] <= psi_e:
            S_e = 1
        else:
            S_e = S_c * (1 + (alpha * psi[i]) ** n) ** (-m)
        residual = theta_r * (
            1 - np.log(alpha * psi[i] + 1) / np.log(alpha * 10 ** 6 + 1)
        )
        theta[i] = max(0, S_e * (theta_s - residual) + residual)
    return theta


def main():

    ivg = ippisch_vg([ 10000, 1000, 100 ], 0.5, 0.02, 1, 0.1, 3)
    civ = campbell_ippisch_vg([ 10000, 1000, 100 ], 0.5, 0.02, 1, 0.1, 3)
    print('ivg',ivg)
    print('civ', civ)

if __name__ == "__main__":
    main()

