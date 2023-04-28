import numpy as np


class MuraMaskObject:
    def __init__(self, style_or_rank="rozhkov", size_mm=22.08, thickness=1.0, aperture_shape="circle", aperture_radius_mm=0.170):
        """
        Creates a Coded Aperture mask object. The main mask array is contained in self.mask_array but
        the simulation needs other info as well which is kept here. Default values are from the Rozhkov setup

        style_or_rank: Either "rozhkov", "russo2020" or an integer denoting the MURA rank.

        Relevant publications:
        Rozhkov et al. (2020): https://doi.org/10.1088/1748-0221/15/06/p06028
        Russo et al. (2020): https://doi.org/10.1016/j.ejmp.2019.12.024
        """
        self.style_or_rank = style_or_rank  # Rank of the mask

        if self.style_or_rank == "rozhkov":
            self.mask_array = self.create_rozhkov()  # Creates the mask array
        elif self.style_or_rank == "russo2020":
            self.mask_array = self.create_russo2020()
        else:
            self.mask_array = self.create(int(self.style_or_rank))

        # Mask Properties
        self.size_px = self.mask_array.shape[0]
        self.size_mm = size_mm  # height of the mask
        self.t = thickness  # thickness of the mask. Unused
        self.aperture_shape = aperture_shape  # aperture shape of the mask. Unused
        self.aperture_radius_mm = aperture_radius_mm  # aperture radius. assumes circular apertures

        # Calculated vals
        self.pixel_mm = self.size_mm / self.size_px

    def create(self, rank: int):
        # Creates the MURA NTHT mask based on quadratic residues with given rank.

        quadratic_res = np.array([])
        for x in range(int(np.ceil(rank / 2))):
            quadratic_res = np.append(quadratic_res, np.remainder(x ** 2, rank))
        quadratic_res = np.unique(quadratic_res)

        A = np.zeros((rank, rank))
        for x in range(A.shape[0]):
            for y in range(A.shape[1]):
                x_is_quadratic_residue = np.any(np.in1d(quadratic_res, x))
                y_is_quadratic_residue = np.any(np.in1d(quadratic_res, y))

                if x == 0:
                    A[x, y] = 0
                elif y == 0 and x != 0:
                    A[x, y] = 1
                elif x_is_quadratic_residue and y_is_quadratic_residue:
                    A[x, y] = 1
                elif not x_is_quadratic_residue and not y_is_quadratic_residue:
                    A[x, y] = 1
                else:
                    A[x, y] = 0

        ntht = np.zeros((A.shape[0] * 2, A.shape[1] * 2))
        ntht[::2, ::2] = A
        return ntht

    def create_russo2020(self):
        # Creates the mask pattern from Russo2020
        rank = 31

        quadratic_res = np.array([])
        for x in range(int(np.ceil(rank / 2))):
            quadratic_res = np.append(quadratic_res, np.remainder(x ** 2, rank))
        quadratic_res = np.unique(quadratic_res)

        A = np.zeros((rank, rank))
        for x in range(A.shape[0]):
            for y in range(A.shape[1]):
                x_is_quadratic_residue = np.any(np.in1d(quadratic_res, x))
                y_is_quadratic_residue = np.any(np.in1d(quadratic_res, y))

                if x == 0:
                    A[x, y] = 0
                elif y == 0 and x != 0:
                    A[x, y] = 1
                elif x_is_quadratic_residue and y_is_quadratic_residue:
                    A[x, y] = 1
                elif not x_is_quadratic_residue and not y_is_quadratic_residue:
                    A[x, y] = 1
                else:
                    A[x, y] = 0

        # Add the erroeous pixel and flip the single pattern:
        A[0, 0] = 1
        A = np.roll(A, -1, axis=0)
        A = np.flipud(A)

        # Make it NTHT and tile to the 2x2 arrangement:
        ntht = np.zeros((A.shape[0] * 2, A.shape[1] * 2))
        ntht[::2, ::2] = A
        ntht = np.tile(ntht, (2, 2))
        return ntht

    def create_rozhkov(self):
        # Creates the mask pattern from Rozhkov
        rank = 31

        quadratic_res = np.array([])
        for x in range(int(np.ceil(rank / 2))):
            quadratic_res = np.append(quadratic_res, np.remainder(x ** 2, rank))
        quadratic_res = np.unique(quadratic_res)

        A = np.zeros((rank, rank))
        for x in range(A.shape[0]):
            for y in range(A.shape[1]):
                x_is_quadratic_residue = np.any(np.in1d(quadratic_res, x))
                y_is_quadratic_residue = np.any(np.in1d(quadratic_res, y))

                if x == 0:
                    A[x, y] = 0
                elif y == 0 and x != 0:
                    A[x, y] = 1
                elif x_is_quadratic_residue and y_is_quadratic_residue:
                    A[x, y] = 1
                elif not x_is_quadratic_residue and not y_is_quadratic_residue:
                    A[x, y] = 1
                else:
                    A[x, y] = 0

        ntht = np.zeros((A.shape[0] * 2, A.shape[1] * 2))
        ntht[::2, ::2] = A
        ntht = np.rot90(ntht)
        ntht = np.tile(ntht, (2, 2))
        # Delete 3 rows and columns:
        ntht = ntht[1:-2, 2:-1]

        return ntht
