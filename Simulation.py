import numpy as np
import cv2


class Simulation:
    """
    This simulation takes a Source object with a data cube in pyramid representation, generates PSFs according to the
    Mask, Detector and their geometrical setup and finally generates a single detector image including Poisson
    noise to the given Photon_count_total.
    a: Mask to object's CENTER distance
    b: Mask to detector distance

    Attention:
    * The (row|column) <-> (y|x) confusion has been solved with transpose operations at certain points.
    * The output detector image also contains the 180° rotation against the input cube orientation coming from
     mirroring.
    * The final size of the detector is determined by the x- & y size of the input cube. So far, no adaption to the
    given size of the DetectorObject is carried out.
    * All convolutions are replaced by multiplication in the Fourier domain with corresponding padding! This causes
     a little error due to pixel roundings, I guess. But it is roughly 100x faster.
    * small_value_clip: Due to performing the convolutions in the Fourier domain, small negative and positive
        numbers occur. All values smaller than small_value_clip (default: 1e-3) are suppressed and set to 0.

    Relevant publications:
    Accorsi et al.(2001): https://doi.org/10.1364/AO.40.004697
    Mu et al. (2006): https://doi.org/10.1109/TMI.2006.873298
    """

    def __init__(self, Mask, Detector,  # Basic objects for CAI
                 a, b,  # Setup geometry
                 photon_count,  # Simulation params
                 Source,  # input data
                 simulation_name="simulation",
                 transmission_rate=0.0,
                 non_central_cos3_effect=False,
                 non_central_coll_effect=False,
                 small_value_clip=1e-3,
                 ):

        # Organization
        self.simulation_name = simulation_name

        # CAI objects
        self.Source = Source
        self.Mask = Mask
        self.Detector = Detector

        # Setup geometry
        self.a = a  # Mask to object's CENTER distance
        self.b = b  # Mask to detector distance

        # Place to store slices of Source array
        self.num_slices = None  # how many slices in the source array?
        self.a_values = None  # distance of each slice from the mask
        self.slices = None  # the 2D array of the slices themselves

        # Place to store PSFs
        self.PSFs = None

        # Simulation params
        # The photon count for the ENTIRE 3D source image. It also contains the noise photons!
        self.Photon_count_total = np.array(photon_count, np.int32)
        self.photon_allotment = None  # Array of photons allotted to each slice
        self.transmission_rate = transmission_rate  # for Russo2020: 1% = 0.01
        self.cos3_maps = None  # Size depends on number of slices
        self.coll_maps = None  # Size depends on number of slices

        # non-central cos3 effect can be simulated:
        self.non_central_cos3_effect = non_central_cos3_effect

        # Currently not used anymore, because only a single non-central coll. map is calculate:
        self.non_central_coll_effect = non_central_coll_effect
        # To suppress small numbers emerging from the Fourier domain:
        self.small_value_clip = small_value_clip

        # Store the detector images
        # The stack of detector images for each slice: yet of unknown amount of slices:
        self.Detector_images_2D_raw = None
        self.Detector_images_2D_noisefree = None
        # The summed detector image of all object slices, noise-free and with Poisson noise applied:
        # Will be filled at the last stage of simulation:
        self.Detector_image_noisefree = np.zeros((self.Detector.size_px, self.Detector.size_px), np.float32)
        self.Detector_image_noisy = np.zeros((self.Detector.size_px, self.Detector.size_px), np.float32)

        # Run the Simulation
        self.run()

    def run(self):
        # Sets slice array, num_slices and a_values:
        self.slice_source_all_slices()
        # Sets PSFs:
        self.create_PSFs_for_source3D()
        # Get raw detector images per slice:
        self.get_detector_images_conv()

        # Everything below here modifies Detector_images_2D_raw
        # --------- COS3 EFFECT ---------
        if self.non_central_cos3_effect:
            self.apply_non_central_cos3_map_for_single_source()

        # --------- COLLIMATION EFFECT ---------
        if self.non_central_coll_effect:
            self.apply_non_central_coll_maps_for_single_source()

        # Do relative weighting of the raw detector images due to distance and intensities per slice:
        self.do_photon_allotment()

        # Sets Detector_image_noisy and Detector_image_noisefree:
        self.get_summed_detector_image_conv()

    # ===============================SECTION: Simulation methods===========================================
    def PSF_raycast_mask_resolution_no_limit(self, z):
        """
        In difference to PSF_raycast_from_mask_keep_resolution, the PSF is not limited to a certain size, but is chosen
        so that all pinholes fit on the image. Close slices might generate a gigantic PSF. Thus, raycasting and
        generating the PSF are now seperated.

        a: mask-object-distance. It is variable so that PSF can be generated for each slice of source array
        """

        # Find magnification ratio
        m = (z + self.b) / z

        # Calculate size of the shadow in mm, then convert to pixels using the detector resolution
        shadow_size_mm = self.Mask.size_mm * m
        shadow_size_px = int(np.round(shadow_size_mm / self.Detector.resolution_mm))

        # Find projected aperture radius in pixels
        r = int(np.round(self.Mask.aperture_radius_mm * m / self.Detector.resolution_mm))

        # Loop over all nonzero pixels of mask and cast ray through each to the shadow
        nonzero_entries = np.array(np.nonzero(self.Mask.mask_array)).astype(int)

        projected_positions = np.zeros((nonzero_entries.shape[1], 2), np.int32)
        n = 0
        for (i, j) in zip(*nonzero_entries):
            mask_pixel_pos = np.array([-self.Mask.size_mm / 2 + i * self.Mask.pixel_mm + self.Mask.pixel_mm / 2,
                                       -self.Mask.size_mm / 2 + j * self.Mask.pixel_mm + self.Mask.pixel_mm / 2,
                                       z])

            # Find corresponding distance in mm on shadow array by scaling it with magnification
            shadow_pixel_pos = mask_pixel_pos * m

            # Convert shadow pixel position into index values in shadow array.
            # Indexing starts at [0,0] and distance values in mm at the center_px
            i_shadow_new = shadow_size_px / 2 + (shadow_pixel_pos[0] / shadow_size_mm) * shadow_size_px
            j_shadow_new = shadow_size_px / 2 + (shadow_pixel_pos[1] / shadow_size_mm) * shadow_size_px
            i_shadow_new = int(np.round(i_shadow_new))
            j_shadow_new = int(np.round(j_shadow_new))

            projected_positions[n, :] = i_shadow_new, j_shadow_new
            n += 1

        # Generate a shadow array to fit all projected pinhole positions plus radius:
        mx_size = np.max(projected_positions) + r
        Shadow_array = np.zeros((mx_size, mx_size), np.float32)

        for i_shadow_new, j_shadow_new in projected_positions:
            # Use OpenCV to draw the circle of projected radius
            cv2.circle(Shadow_array, (i_shadow_new, j_shadow_new), r, color=1, thickness=-1)

        # IMPORTANT HERE: Due to the (row|column) <-> (y|x) confusion, we need to switch x and y axis here, to obtain
        # a Shadow array with a left-to-right orientation of the central main stripe.
        Shadow_array = Shadow_array.transpose()

        return Shadow_array

    def slice_source_all_slices(self):
        """
        Slices the source along z direction and stores the slices and their corresponding distances from mask
        """

        # Cull any sparse elements in source array
        nonzeros = np.nonzero(self.Source.source_array)

        # z values of each slice of array in pixels
        z_vals = np.unique(nonzeros[2])

        # Set num slices
        self.num_slices = len(z_vals)
        self.a_values = np.zeros(self.num_slices, np.float32)

        # Initialize slices array
        self.slices = np.zeros((self.Source.size_px, self.Source.size_px, self.num_slices), np.float32)

        # Fill slices array
        spanstart = self.a - self.Source.size_z_mm / 2
        assert spanstart > 0, "Negative distance here. Something does not add up."
        for i in range(self.num_slices):
            self.slices[:, :, i] = self.Source.source_array[:, :, z_vals[i]]
            self.a_values[i] = z_vals[i] * self.Source.resolution_z + spanstart

    def create_PSFs_for_source3D(self):
        """
        It generates a PSF for every slice of the object along z axis by calling the PSF method.
        """
        # Initialize empty python list to store different shaped PSFs as numpy arrays:
        self.PSFs = []

        for k in range(self.num_slices):
            # Append the PSF with different sizes:
            temp_PSF = self.PSF_raycast_mask_resolution_no_limit(self.a_values[k])
            self.PSFs.append(temp_PSF)

    def get_detector_images_conv(self):
        """
        Returns an array of detector images for the given array of PSFs and object slices by equation
            p = f * h where p is forward projection, f is object array, h is PSF array, and * is convolution
        """
        # Initialize empty arrays to store projections
        self.Detector_images_2D_raw = np.zeros((self.Detector.size_px, self.Detector.size_px, self.num_slices),
                                               np.float32)
        # Use the PSF for each slice and do the convolution
        for i in range(self.num_slices):
            this_PSF_a = self.PSFs[i]
            this_slice = self.slices[:, :, i]

            # NEW: Convolution in Fourier domain: 100x faster.
            temp_dect = self.conv_in_fourier(this_slice, this_PSF_a)

            # Small negative numbers from the FFT can occur, they are clipped:
            temp_dect[temp_dect < self.small_value_clip] = 0.0

            # Rotate image to mimic mirroring effect:
            temp_dect = np.rot90(temp_dect, 2)
            self.Detector_images_2D_raw[:, :, i] = temp_dect

    # ------- COS3 EFFECT SIMULATION -------
    def calc_cos3_map(self, z):
        """"
        z is the mask-object slice distance
        """
        detector_px_size = self.Detector.resolution_mm
        detector_pxs = self.Detector.size_px

        # Determine center positions of detector pixels.
        # All at once without double for-loop:
        x_px, y_px = np.mgrid[0:detector_pxs, 0:detector_pxs].astype(np.float32)
        ri_px = np.array([x_px, y_px]) - ((detector_pxs / 2.0) - 0.5)
        ri_px = ri_px.transpose((1, 2, 0))
        ri_mm = ri_px * detector_px_size
        ri_mm_center_dist = np.linalg.norm(ri_mm, axis=2, ord=2)

        # Calculate all thetas:
        # Attention: In Accorsi2001 Eq. (10) "z" means a+b, aka the detector-object distance:
        # That is why here, we must write self.b + z
        theta_map = np.arctan(ri_mm_center_dist / (self.b + z))
        # Calculate corresponding cos3:
        cos3_map = np.cos(theta_map) ** 3
        return cos3_map

    def create_cos3_maps(self):
        """
        It generates the cos3-maps, describing the first order near-field effect. For each slice, one of those
        maps is generated and stored. Afterwards all simulation slices will be multiplied by this cos3-maps.
        """
        # Initialize empty python list to store the cos3-maps:
        self.cos3_maps = np.zeros((self.Detector.size_px, self.Detector.size_px, self.num_slices), np.float32)

        for i, a in enumerate(self.a_values):
            this_cos3_map = self.calc_cos3_map(z=a)
            self.cos3_maps[..., i] = this_cos3_map

    def apply_cos3_maps(self):
        # Iterate through all raw detector images and multiply it with the according cos3-map:
        for i in range(np.shape(self.cos3_maps)[-1]):
            self.Detector_images_2D_raw[..., i] = self.Detector_images_2D_raw[..., i] * self.cos3_maps[..., i]

    # --------------------------------------

    # ------- NON-CENTRAL COS3 EFFECT SIMULATION -------
    def calc_non_central_cos3_map(self, x, y, z):
        """"
        Here, we do not assume the source to be central but non-central.
        Output is the cos3 map for a single point sources at position (x, y, z).
        At the end, the cos3 map is mirrored because detector mirroring takes place in the convolution part.
        """
        detector_px_size = self.Detector.resolution_mm
        detector_pxs = self.Detector.size_px

        # Determine center positions of detector pixels.
        # All at once without double for-loop:
        x_px, y_px = np.mgrid[0:detector_pxs, 0:detector_pxs].astype(np.float32)
        ri_px = np.array([x_px, y_px]) - ((detector_pxs / 2.0) - 0.5)
        ri_px = ri_px.transpose((1, 2, 0))
        ri_mm = ri_px * detector_px_size

        # Now subtract the vector to the point source position...
        r_diff = ri_mm - np.array([x, y], np.float32)
        # ... and determine the thetas from those vectors:
        r_diff_mm_center_dist = np.linalg.norm(r_diff, axis=2, ord=2)
        # Calculate all thetas:
        # Attention: In Mu2006 Eq. (15) "z" means a+b, aka the detector-object distance:
        # That is why here, we must write self.b + z
        theta_map = np.arctan(r_diff_mm_center_dist / (self.b + z))
        # The final cos3 map: C_planar
        cos3_map = np.cos(theta_map) ** 3

        # Rotate coll_map because, the mirroring effect is carried out later separately, and this coll_map
        # is applied on the un-mirrored convolution results.
        cos3_map = np.rot90(cos3_map, 2)
        return cos3_map

    def apply_non_central_cos3_map_for_single_source(self):
        """ Instead of assuming a central source, here a single cos3 map is generated and multiplied with the
         raw detector images based on the single source position from the Source object. """
        # Use the center given by the source object:
        center_mm = self.Source.centers_mm
        assert center_mm.shape == (1, 3), "More than one source center was found, " \
                                          "apply_nonc_cos3_map_for_single_source not applicable."
        # Calculate the non-central pinhole collimation map. Also, the coll-map is NOT CONVOLVED WITH A PSF!
        temp_noncos3 = self.calc_non_central_cos3_map(center_mm[0, 0], center_mm[0, 1], center_mm[0, 2])
        temp_noncos3 = np.expand_dims(temp_noncos3, -1)

        # Multiply all raw detector images by a single non-central cos3 map
        self.Detector_images_2D_raw = self.Detector_images_2D_raw * temp_noncos3

    # ------- NON-CENTRAL COLLIMATION EFFECT SIMULATION -------
    def calc_non_central_pinhole_coll_map(self, x, y, z):
        """"
        x, y, z is the point source position in mm.
        NEW: Rotation by 180° is important here!
        """
        detector_px_size = self.Detector.resolution_mm
        detector_pxs = self.Detector.size_px
        r = self.Mask.aperture_radius_mm
        t = self.Mask.t

        # Determine center positions of detector pixels.
        # All at once without double for-loop:
        x_px, y_px = np.mgrid[0:detector_pxs, 0:detector_pxs].astype(np.float32)
        ri_px = np.array([x_px, y_px]) - ((detector_pxs / 2.0) - 0.5)
        ri_px = ri_px.transpose((1, 2, 0))
        ri_mm = ri_px * detector_px_size

        # Now subtract the vector to the point source position...
        r_diff = ri_mm - np.array([x, y], np.float32)
        # ... and determine the thetas from those vectors:
        r_diff_mm_center_dist = np.linalg.norm(r_diff, axis=2, ord=2)
        # Calculate all thetas:
        # Attention: In Mu2006 Eq. (15) "z" means a+b, aka the detector-object distance:
        # That is why here, we must write self.b + z
        theta_map = np.arctan(r_diff_mm_center_dist / (self.b + z))

        # Displacement:
        d = np.minimum(t * np.tan(theta_map), 2 * r)

        # Angle alpha of sector in Mu2006 Fig. 2c
        alpha_map = np.arccos(d / (2 * r))

        # Collimation correction factor:
        coll_map = (2 * alpha_map * r - d * np.sin(alpha_map)) / (np.pi * r)

        # Rotate coll_map because, the mirroring effect is carried out later separately, and this coll_map
        # is applied on the un-mirrored convolution results.
        coll_map = np.rot90(coll_map, 2)
        return coll_map

    def apply_non_central_coll_maps_for_single_source(self):
        """ DIFFERENCE TO apply_non_central_coll_maps_for_single_source:
        The non-central collimation map is multiplied with the pre-calculated raw detector images.
        CHANGES self.Detector_images_2D_raw
        """
        # Do not take the center of mass, but the center given by the source object:
        center_mm = self.Source.centers_mm
        assert center_mm.shape == (1, 3), "More than one source center was found, " \
                                          "apply_non_central_coll_maps_for_single_source not applicable."
        # Calculate the non-central pinhole collimation map. Also, the coll-map is NOT CONVOLVED WITH A PSF!
        temp_nonc = self.calc_non_central_pinhole_coll_map(center_mm[0, 0], center_mm[0, 1], center_mm[0, 2])
        # Rotate by 180° again because the raw detector images are already mirrored:
        temp_nonc = np.rot90(temp_nonc, 2)
        temp_nonc = np.expand_dims(temp_nonc, -1)

        # Multiply all raw detector images by a single non-central collimation map
        self.Detector_images_2D_raw = self.Detector_images_2D_raw * temp_nonc

    # -----------------------------------------------------

    def do_photon_allotment(self):
        # PHOTON ALLOTMENT:
        # First doll out the photons based on the inverse square law:
        # Calculate proportion of photons each pixel gets from each slice, and each slice from the total
        scaled_a_values = np.zeros_like(self.a_values)
        for i in range(self.num_slices):
            slice_intensity_sum = np.sum(self.slices[:, :, i])
            scaled_a_values[i] = slice_intensity_sum * (1.0 / self.a_values[i] ** 2)  # This made sense a few months ago

        # How much photon count to allot for each slice IN RELATIVE MEASURE:
        self.photon_allotment = scaled_a_values / np.sum(scaled_a_values)

        # Share the photon allotment among the slices
        self.Detector_images_2D_raw = self.photon_allotment * self.Detector_images_2D_raw

    def get_summed_detector_image_conv(self):
        """
        Returns the detector image of the entire 3D object and simulates poisson noise
        """
        if np.sum(self.Detector_images_2D_raw) == 0:
            print("No signal on detector!")

        # Sum the detector image stack into a single multiplexed image
        self.Detector_image_noisefree = np.sum(self.Detector_images_2D_raw, 2)

        # Adjust detector image to given photon count:
        # Split between transitioned photons (noise) and photons going through the pinholes (signal):
        p_pattern = np.array(self.Photon_count_total / (1 + self.transmission_rate), np.float32)
        p_noise = p_pattern * self.transmission_rate

        # Added a small epsilon to the divisor:
        self.Detector_image_noisefree = self.Detector_image_noisefree / (np.sum(self.Detector_image_noisefree) + 1e-8)
        self.Detector_image_noisefree = self.Detector_image_noisefree * p_pattern

        # Add transmission noise as uniform noise and convert it to float32:
        trans_noise = np.ones_like(self.Detector_image_noisefree) / np.prod(self.Detector_image_noisefree.shape)
        trans_noise = trans_noise.astype(np.float32)
        trans_noise = trans_noise * p_noise
        self.Detector_image_noisefree = self.Detector_image_noisefree + trans_noise

        # Replace entire image with poisson noise where each pixel is sampled from a poisson distribution with pixel
        # value as the mean
        assert self.Detector_images_2D_raw.min() >= 0.0, \
            "Value below 0 found in Detector_images_2D_raw: min found: %f" % self.Detector_images_2D_raw.min()
        assert self.Detector_image_noisefree.min() >= 0.0, \
            "Value below 0 found in Detector_image_noisefree: min found: %f" % self.Detector_image_noisefree.min()

        assert np.any(np.isnan(self.Detector_images_2D_raw)) == False, \
            "NAN found in Detector_images_2D_raw: Sum of nans: %i" % np.isnan(self.Detector_images_2D_raw).sum()
        assert np.any(np.isnan(self.Detector_image_noisefree)) == False, \
            "NAN found in Detector_image_noisefree: Sum of nans: %i" % np.isnan(self.Detector_image_noisefree).sum()

        # Apply poisson noise!
        self.Detector_image_noisy = np.random.poisson(self.Detector_image_noisefree).astype(np.float32)

    def conv_in_fourier(self, _input, _filter):
        """ This function calculates the linear 2d convolution of the two inputs inside the Fourier domain.
        To yield the result of the LINEAR convolution of two images, careful zero-padding must be carried out first.
        The result might be one or two pixels shifted due to odd image sizes.
        ATTENTION: Due to numerical issues, small negative numbers can occur. They are dealt with somewhere else!
        ATTENTION: Also, many very small non-zero numbers can appear!
        ATTENTION: The filter MUST NOT be zero-padded!
        """
        size_input = np.shape(_input)[0]
        size_filter = np.shape(_filter)[0]

        # Transform to Fourier domain:
        S = np.fft.rfft2(_input, (size_input + size_filter - 1, size_input + size_filter - 1))
        F = np.fft.rfft2(_filter, (size_input + size_filter - 1, size_input + size_filter - 1))
        R = S * F
        # Transform back and crop the center:
        r = np.fft.irfft2(R).astype(np.float32)
        startx = np.shape(r)[0] // 2 - (size_input // 2)
        starty = np.shape(r)[1] // 2 - (size_input // 2)
        r_cropped = r[starty:starty + size_input, startx:startx + size_input]
        return r_cropped


if __name__ == "__main__":
    import os
    import time

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from pyevtk.hl import gridToVTK
    from Simulation import Simulation
    from MaskObject import MuraMaskObject
    from DetectorObject import DetectorObject
    from TobisSources import EllipsoidSourceGenerator, plotp, plot, norm
    from draw_from_phsp import p_random_photons
    from scipy.signal import correlate2d

    hd = 24.64  # mm. Smallest detector dimension
    hm = 9.92
    b = 22  # in mm. Detector-mask-distance

    # Figure which shows position and radii of all simulations in a 2D sketch
    sg = EllipsoidSourceGenerator(448, size_xy_mm=200 * hd / b, size_z_mm=195, centers_mm=[[0, 0, 63.2],
                                                                                           [7, 2, 15],
                                                                                           [-20, -70, 190]],
                                  radii_mm=[[2, 2, 2],
                                            [3, 3, 3],
                                            [5, 5, 5]], intens=[1, 1, 1], origin_mm=[0, 0, (5 + 200) / 2])
    # Tiny point source:
    sg = EllipsoidSourceGenerator(448, size_xy_mm=200 * hd / b, size_z_mm=195, centers_mm=[[10, 20, 63.2]],
                                  radii_mm=[[0.01, 0.01, 0.01]], intens=[1, 1, 1], origin_mm=[0, 0, (5 + 200) / 2])
    plotp(sg.source_array, xyres=None, zres=None, title="Test volume before morphing").show()
    sg.draw_sketch(b=b, hd=hd, hm=hm).show()
    sg.convert_cube_to_pyramide(b=b, hd=hd)

    mask = MuraMaskObject(style_or_rank="russo2020", size_mm=9.92, thickness=0.11, aperture_shape="circle",
                          aperture_radius_mm=0.08 / 2.0)
    detector = DetectorObject(size_px=448, size_mm=min(24.64, 28.16))
    sim = Simulation(Mask=mask, Detector=detector, a=sg.origin_mm[2], b=b, photon_count=1_000_000, Source=sg,
                     simulation_name="sim",
                     cos3_effect=True, center_coll_effect=False, transmission_rate=0.01)

