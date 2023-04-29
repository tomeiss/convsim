
import numpy as np
import matplotlib.pyplot as plt


class EllipsoidSourceGenerator():
    def __init__(self, size_px, size_xy_mm, size_z_mm, centers_mm, radii_mm, intens, origin_mm):
        """
            Object coordinate: system origin is in the box center.
            Camera coordinate: system is right in front of the mask, with z pointing towards the object and x and y
            pointing to the right and bottom, when seen from the camera.
            origin_mm is the object's center coordinates, seen from the camera coordinate system!
            radii_mm is 3d with a radius in x, y and z-direction!
        """
        # General stuff:
        self.size_px = np.squeeze(size_px)
        self.size_xy_mm = np.squeeze(size_xy_mm)
        self.size_z_mm = np.squeeze(size_z_mm)
        self.resolution_xy = self.size_xy_mm / self.size_px  # mm/pixel in x and y directions
        self.resolution_z = self.size_z_mm / self.size_px  # mm/pixel in z directions
        self.center_px = int(self.size_px / 2)  # the center_px pixel
        self.origin_mm = np.squeeze(origin_mm)
        # Vector from top left frontal corner pixel to the cube's center in mm:
        self.top_left_frontal_corner_mm = np.array([(self.size_px / 2.0 - 0.5) * self.resolution_xy,
                                                    (self.size_px / 2.0 - 0.5) * self.resolution_xy,
                                                    (self.size_px / 2.0 - 0.5) * self.resolution_z], np.float32)

        # Params for the spheres:
        self.centers_mm = np.array(centers_mm).reshape(-1, 3).astype(np.float32)
        self.radii_mm = np.array(radii_mm).reshape(-1, 3).astype(np.float32)
        self.intens = np.array(intens).reshape(-1, 1).astype(np.float32)

        # Initialize empty source array and fill it with spheres:
        self.source_array = np.zeros((self.size_px, self.size_px, self.size_px), dtype=np.float32)
        self.create_multi_ellipsoid3D(self.centers_mm, self.radii_mm, self.intens)

        # Flag which changes to "pyramid" after convert_cube_to_pyramid has been called
        self.cube_represents = "cube"

    def create_multi_ellipsoid3D(self, centers_mm, radii_mm, weights):
        for c_mm, r_mm, w in zip(centers_mm, radii_mm, weights):
            # Transform from camera coordinate system to object:
            c_o_mm = c_mm - self.origin_mm

            # Transform to matrix origin in the TOP-LEFT-CENTRAL corner:
            c_matrix_mm = c_o_mm + self.top_left_frontal_corner_mm

            # Convert center and radii to pixels (seperated by xy and z resolution):
            c_px = c_matrix_mm / [self.resolution_xy, self.resolution_xy, self.resolution_z]
            r_px = r_mm / [self.resolution_xy, self.resolution_xy, self.resolution_z]

            # If any radius is smaller than 1px it is increased to 1px:
            if np.less(r_px, 1.0).any():
                r_px = np.where(np.array(r_px) < 1.0, 1.0, r_px)
                print("Radius was ceiled to 1px")

            # Round indices to the nearest integers:
            c_px_rounded = np.round(c_px).astype(int)
            r_px_rounded = np.round(r_px).astype(int)

            print("Object COS: (%.2f, %.2f, %.2f)mm. World COS: (%.2f, %.2f, %.2f)mm" % (*c_o_mm, *c_mm))
            print("with radii: (%.2f, %.2f, %.2f)mm" % (*r_mm,))
            print("Creating an ellipsoid at (%i, %i, %i)px with radii (%i, %i, %i)" % (*c_px_rounded, *r_px_rounded))
            print("Retransforming rounded parameters back to world coordinate system yields: ")
            print("(%.2f, %.2f, %.2f) with radius of (%.2f, %.2f, %.2f)mm"
                  % (*(c_px_rounded * [self.resolution_xy, self.resolution_xy, self.resolution_z]
                       - self.top_left_frontal_corner_mm + self.origin_mm),
                     *(r_px_rounded * [self.resolution_xy, self.resolution_xy, self.resolution_z])))

            # Draw ellipsoid into data cube:
            self.draw_ellipsoid3D(c_px_rounded, r_px_rounded, w)
        # print("Done drawing circles.")

    def draw_ellipsoid3D(self, center_px, radius_px, intensity):
        """
        Draws a spherical source into the source array. ATTENTION: It is additive!
        center_px:      3D and already converted and rounded to pixels
        radius_px:      3D. Radii in pixels according to x, y and z direction.
        intensity:      Intensity of the circle. A
        """

        # Find radius in xy and z in pixels
        r_x_px = radius_px[0]
        r_y_px = radius_px[1]
        r_z_px = radius_px[2]

        # Creates a bounding box for the sphere
        x_square_min = center_px[0] - r_x_px
        x_square_max = center_px[0] + r_x_px + 1
        y_square_min = center_px[1] - r_y_px
        y_square_max = center_px[1] + r_y_px + 1
        z_square_min = center_px[2] - r_z_px
        z_square_max = center_px[2] + r_z_px + 1

        # Done: Should be alright. Check if boundaries are alright here...
        # Limit the *_square_max and min values:
        x_square_min = np.clip(x_square_min, 0, self.size_px - 1)
        y_square_min = np.clip(y_square_min, 0, self.size_px - 1)
        z_square_min = np.clip(z_square_min, 0, self.size_px - 1)
        x_square_max = np.clip(x_square_max, 0, self.size_px - 1)
        y_square_max = np.clip(y_square_max, 0, self.size_px - 1)
        z_square_max = np.clip(z_square_max, 0, self.size_px - 1)

        # Iterates over each pixel in the bounding box, assigning 1 if falls within radius of sphere
        for p in range(x_square_min, x_square_max + 1):
            for q in range(y_square_min, y_square_max + 1):
                for u in range(z_square_min, z_square_max + 1):
                    # This is used to scale the ellipsoid:
                    ellipsoid_eq = ((p - center_px[0]) / r_x_px) ** 2 + ((q - center_px[1]) / r_y_px) ** 2 + \
                                   ((u - center_px[2]) / r_z_px) ** 2
                    if ellipsoid_eq <= 1.0:
                        self.source_array[p, q, u] = self.source_array[p, q, u] + intensity

    def convert_cube_to_pyramide(self, b, hd):
        """ VERY IMPORTANT STEP RIGHT HERE: Because for CAI the FoV is a pyramid with the spike pointing towards the
        mask, before applying the ConvSim, the data cube must be morphed into a pyramide. Slices closer to the camera
        are cropped, slices behind the object center are padded with zeros, so that the xy-resolution
        fits. Afterwards the slice is resized to yield a 256x256 image again.

        => Bilinear interpolation with intensity conservation!

        b: mask-detector distance in mm.
        hd: Mask side length in mm.
        """

        if self.cube_represents == "pyramid":
            print(" Cube already in pyramid representation.")
            return None

        import tensorflow as tf
        data_cube = self.source_array
        nx, ny, nz = data_cube.shape
        # Only for non-zero z-slices:
        nnz_zslices = np.transpose(np.nonzero(np.any(data_cube, (0, 1)))[0])
        for i in nnz_zslices:
            slice = data_cube[:, :, i]
            this_z_mm = self.origin_mm[2] - (self.top_left_frontal_corner_mm[2] - (i * self.resolution_z))
            # print("this_z_mm: %.2f" % this_z_mm)
            this_fov = this_z_mm * hd / b
            actual_res = this_fov / data_cube.shape[0]
            # print("Simplified xy-resolution: %.3f. Actual xy-resolution: %.3f" % (self.resolution_xy, actual_res))
            f = actual_res / self.resolution_xy

            # The TF function below combines both possibilities into a single function:
            if f <= 1.0:
                # Cropping and upsample:
                this_pyramide_slice = tf.image.central_crop(slice[None, ..., None], f)
                intermediate_res = this_fov / this_pyramide_slice.shape[1]  # Should be ~resolution.xy
                crop_or_padded_sum = tf.reduce_sum(this_pyramide_slice)
                this_pyramide_slice = tf.image.resize(this_pyramide_slice, [nx, ny], "bilinear")
            else:
                # Zero-padding and downsample:
                this_pyramide_slice = tf.image.resize_with_crop_or_pad(slice[None, ..., None],
                                                                       np.round(f * nx).astype(int),
                                                                       np.round(f * ny).astype(int))
                crop_or_padded_sum = tf.reduce_sum(this_pyramide_slice)
                this_pyramide_slice = tf.image.resize(this_pyramide_slice, [nx, ny], "bilinear")

            # Reset the initial intensity if the source has not been cut out:
            if crop_or_padded_sum != 0:
                this_pyramide_slice = this_pyramide_slice / tf.reduce_sum(this_pyramide_slice) * crop_or_padded_sum
            # Convert to numpy and squeeze it:
            this_pyramide_slice = np.squeeze(this_pyramide_slice)

            data_cube[:, :, i] = this_pyramide_slice

        self.cube_represents = "pyramid"

    def draw_sketch(self, b, hd, hm):
        """ Draws a sketch with an optical axis, the detector, mask, central pattern,
        the pinhole FoV, the top-left-frontal corner, the center, and the draw ellipsoid"""
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 8))

        max_z = 1.1*(self.origin_mm[2]+self.size_z_mm/2)
        z = np.linspace(-22, max_z, 100)
        # Optical axis:
        plt.plot([-b - 5, max_z], [0, 0], linestyle="--", color="black")
        # Detector and mask:
        plt.vlines(-b, -hd / 2, hd / 2, colors="black", label="Detector")
        plt.vlines(0, -hm / 2, hm / 2, colors="black", label="Mask")
        # Central pattern:
        plt.hlines([0.25 * hm, -0.25 * hm], -2, 2, colors="gray", label="Central pattern")
        # Pinhole:
        plt.plot(z, [0.5 * (k * hd / b) for k in z], color="b", label="Pinhole")
        plt.plot(z, [-.5 * (k * hd / b) for k in z], color="b", label="")

        # Draw top left frontal corner:
        plt.scatter(self.origin_mm[2] - self.top_left_frontal_corner_mm[2],
                    self.origin_mm[1] + self.top_left_frontal_corner_mm[1], label="TLF")
        plt.scatter(self.origin_mm[2], self.origin_mm[1], label="origin")
        # Draw cube:
        zs = [self.origin_mm[2] - self.size_z_mm/2,
              self.origin_mm[2] + self.size_z_mm/2,
              self.origin_mm[2] + self.size_z_mm/2,
              self.origin_mm[2] - self.size_z_mm/2,
              self.origin_mm[2] - self.size_z_mm/2]
        xs = [self.origin_mm[1] + self.size_xy_mm/2,
              self.origin_mm[1] + self.size_xy_mm/2,
              self.origin_mm[1] - self.size_xy_mm/2,
              self.origin_mm[1] - self.size_xy_mm/2,
              self.origin_mm[1] + self.size_xy_mm/2]
        plt.plot(zs, xs, color="green")

        # Draw ellipsoid:
        """   """
        for c, r in zip(self.centers_mm, self.radii_mm):
            cz = c[2]
            cx = c[1]
            a = r[2]
            b = r[1]
            t = np.linspace(0, 2 * np.pi, 100)
            plt.plot(cz + a * np.cos(t), cx + b * np.sin(t), color="red", label="ellipsoid", zorder=10000)

        plt.axis("equal")
        plt.grid()
        plt.xlabel("x [mm]")
        plt.ylabel("y [mm]")
        plt.legend()
        return fig


if __name__ == "__main__":
    print("This is the EllipsoidSourceGenerator")
