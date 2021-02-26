from abc import ABC, abstractmethod
import swcc_function as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as op
from sklearn.metrics import r2_score


class SWCCModel(ABC):
    """
    base class for all SWCC model
    """

    @abstractmethod
    def curve_fit(self):
        """
        fit a swcc with given psi-theta data
        """

    @abstractmethod
    def data_generator(self):
        """
        generate data points of a given swcc
        """

    @abstractmethod
    def compute_k(self):
        """
        compute the hydraulic conductivity from a given swcc
        """

    @abstractmethod
    def compute_C(self):
        """
        compute the specific water storage from a given swcc
        """


class Compbell(SWCCModel):
    """
    create a Compbell SWCC model
    ## Parameters:
        `theta_s`: volumetric water content
        `psi_e`: air-entry suction value
        `b`: a shape parameter related to the pore size distribution of the soil
    ## Returns:
        a Compbell model
    """

    def __init__(self, theta_s=0.5, psi_e=-1, b=3):
        self._theta_s = theta_s
        self._psi_e = psi_e
        self._b = b

    def curve_fit(self, dataset, visual=True):
        """
        fit a SWCC with given dataset
        params:
            `dataset`: an ndarray with a shape of (n, 2), the first column of which is psi, the second one of which is theta_s
        returns:
            `self._theta_s`
            `self._psi_e`
            `self._b`
            `R^2`
        """
        psi = dataset[:, 0]
        theta = dataset[:, 1]
        bounds = (-np.inf, [1, np.inf, np.inf])  # make sure theta_s <= 1.0
        self._theta_s, self._psi_e, self._b = op.curve_fit(
            sf.campbell, psi, theta, bounds=bounds
        )[0]

        # compute R^2
        r2 = r2_score(
            theta, sf.campbell(psi, self._theta_s, self._psi_e, self._b)
        )
        print("campbell")
        print(self._theta_s, self._psi_e, self._b, r2)
        return self._theta_s, self._psi_e, self._b, r2

    def data_generator(self):
        pass

    def compute_k(self):
        pass

    def compute_C(self):
        pass


class VG(SWCCModel):
    """
    create a restricted VG SWCC model
    ## Parameters:
        theta_s: volumetric water content
        theta_r: residual volumetric water content
        alpha: related to air-entry suction value
        n: curve shape parameter
    ## Returns:
        a restricted VG model
    """

    def __init__(self, theta_s=0.5, theta_r=0.01, alpha=0.3, n=1):
        self._theta_s = theta_s
        self._theta_r = theta_r
        self._alpha = alpha
        self._n = n

    def curve_fit(self, dataset):
        psi = dataset[:, 0]
        theta = dataset[:, 1]
        bounds = ([0, 0, -np.inf, -np.inf], [1, 1, np.inf, np.inf])
        self._theta_s, self._theta_r, self._alpha, self._n = op.curve_fit(
            sf.vg, psi, theta, bounds=bounds
        )[0]

        # compute R^2
        r2 = r2_score(
            theta,
            sf.vg(psi, self._theta_s, self._theta_r, self._alpha, self._n),
        )
        print("vg")
        print(self._theta_s, self._theta_r, self._alpha, self._n, r2)
        return self._theta_s, self._theta_r, self._alpha, self._n, r2

    def data_generator(self):
        pass

    def compute_C(self):
        pass

    def compute_k(self):
        pass


class IppischVG(SWCCModel):
    """
    create a Ippisch-VG model
    """

    def __init__(self, theta_s=0.5, theta_r=0.01, psi_e=-1, alpha=0.3, n=1):
        self._theta_s = theta_s
        self._theta_r = theta_r
        self._psi_e = psi_e
        self._alpha = alpha
        self._n = n

    def curve_fit(self, dataset):
        psi = dataset[:, 0]
        theta = dataset[:, 1]
        bounds = (
            [0, 0, -np.inf, -np.inf, -np.inf],
            [1, 1, np.inf, np.inf, np.inf],
        )
        (
            self._theta_s,
            self._theta_r,
            self._psi_e,
            self._alpha,
            self._n,
        ) = op.curve_fit(sf.ippisch_vg, psi, theta, bounds=bounds)[0]

        # compute R^2
        r2 = r2_score(
            theta,
            sf.ippisch_vg(
                psi,
                self._theta_s,
                self._theta_r,
                self._psi_e,
                self._alpha,
                self._n,
            ),
        )
        print("ippisch_vg")
        print(
            self._theta_s, self._theta_r, self._psi_e, self._alpha, self._n, r2
        )
        return (
            self._theta_s,
            self._theta_r,
            self._psi_e,
            self._alpha,
            self._n,
            r2,
        )

    def data_generator(self):
        pass

    def compute_k(self):
        pass

    def compute_C(self):
        pass


class CompbellIppischVG(SWCCModel):
    def __init__(self, theta_s=0.5, theta_r=0.01, psi_e=-1, alpha=0.3, n=1):
        self._theta_s = theta_s
        self._theta_r = theta_r
        self._psi_e = psi_e
        self._alpha = alpha
        self._n = n

    def curve_fit(self, dataset):
        psi = dataset[:, 0]
        theta = dataset[:, 1]
        bounds = (
            [0, 0, -np.inf, -np.inf, -np.inf],
            [1, 1, np.inf, np.inf, np.inf],
        )
        (
            self._theta_s,
            self._theta_r,
            self._psi_e,
            self._alpha,
            self._n,
        ) = op.curve_fit(sf.campbell_ippisch_vg, psi, theta, bounds=bounds)[0]

        # compute R^2
        r2 = r2_score(
            theta,
            sf.campbell_ippisch_vg(
                psi,
                self._theta_s,
                self._theta_r,
                self._psi_e,
                self._alpha,
                self._n,
            ),
        )
        print("campbell_ippisch_vg")
        print(
            self._theta_s, self._theta_r, self._psi_e, self._alpha, self._n, r2
        )
        return (
            self._theta_s,
            self._theta_r,
            self._psi_e,
            self._alpha,
            self._n,
            r2,
        )

    def data_generator(self):
        pass

    def compute_k(self):
        pass

    def compute_C(self):
        pass


def main():
    psi = np.array([5030.1, 3974.9, 4209.3, 2131.0, 3673.2, 2088.8, 1048.6])
    theta_s = np.array([14.7, 15.2, 17.6, 19.7, 21.5, 23.8, 25.5]) / 31
    dataset = np.array((psi, theta_s)).T

    cpb = Compbell()
    cpb.curve_fit(dataset)

    vg = VG()
    vg.curve_fit(dataset)

    is_vg = IppischVG()
    is_vg.curve_fit(dataset)

    civ = CompbellIppischVG()
    civ.curve_fit(dataset)


if __name__ == "__main__":
    main()
