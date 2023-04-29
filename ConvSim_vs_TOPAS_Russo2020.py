"""
This script runs ConvSim simulations as described in the given publication, loads the detector images from TOPAS and
creates the comparative figures from the paper.

If you use this software, please cite it as follows:
T. Meißner and S. Pietrantonio et al., "Towards a fast and accurate simulation framework for 3D spherical source
localization in the near field of a coded aperture gamma camera," 2023.

"""

import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from Simulation import Simulation
from MaskObject import MuraMaskObject
from DetectorObject import DetectorObject
from SourceGenerators import EllipsoidSourceGenerator


def norm(img):
    """ Normalizes the given image to the range [0, 1]"""
    return (img - img.min()) / (img.max() - img.min())


def plotp(data_cube, title="", projection_fnc=np.sum, colorbar=True, subtitles=None):
    """
    subtitles: subtitles which go over the left and right sub-plots
    colorbar: if a colorbar should be depicted or not.
    data_cube: 3d numpy array to make projection figure of.
    title: str. Comes at the top of figure.
    projection_fnc: Should be either np.sum or np.max. Different styles of projecting the cube to a 2D
    representation. It is somewhat hard to get the actual voxel values from np.sum.
    Returns a plt.figure which can then be saved, shown or closed.
    """
    data_cube = np.array(data_cube).squeeze()

    if subtitles is None:
        subtitles = ["Sum Projection along Z direction.", "Sum Projection along Y direction."]

    plt.close('all')
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    im = axs[0].imshow(projection_fnc(data_cube, 2))
    if colorbar:
        fig.colorbar(im, ax=axs[0])
    axs[0].set_title(subtitles[0])

    axs[0].set_xlabel("y [Pixel]")
    axs[0].set_ylabel("x [Pixel]")
    im = axs[1].imshow(projection_fnc(data_cube, 1))
    if colorbar:
        fig.colorbar(im, ax=axs[1])
    axs[1].set_title(subtitles[1])
    axs[1].set_xlabel("z [Pixel]")
    axs[1].set_ylabel("x [Pixel]")
    fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot(img, title="", ticks=True, colorbar=True, cmap=None, clim=None, filename="", dpi=None, cb_format=None,
         fontsize=None):
    """ Plot a single image."""
    plt.clf()
    plt.cla()
    # plt.figure(figsize=(5, 5))
    plt.imshow(np.squeeze(img), cmap=cmap)
    if ticks == False:
        plt.xticks([])
        plt.yticks([])
    if colorbar:
        cb = plt.colorbar(format=cb_format)
    if clim:
        plt.clim(clim)
    plt.title(str(title), fontsize=fontsize)

    if fontsize:
        cb.ax.tick_params(labelsize=fontsize)
    if dpi:
        plt.gcf().set_dpi(dpi)
    plt.tight_layout()
    fig = plt.gcf()
    if filename == "":
        plt.show()
    else:
        plt.savefig(filename)
    return fig


def triplot(images, titles=None, suptitle="", ticks=True, cmap=None, show=True):
    """ Plots 3 images in a 1x3 subplot."""
    if titles is None:
        titles = ["", "", ""]
    plt.clf()
    # New: figsize
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs = axs.ravel()

    plt.suptitle(suptitle)
    for i in range(3):
        im = axs[i].imshow(np.array(images[i]).squeeze(), cmap=cmap)
        if titles[i]:
            axs[i].set_title(titles[i], fontsize=18)
        cb = fig.colorbar(im, ax=axs[i], shrink=0.8, format="%i")
        [t.set_fontsize(14) for t in cb.ax.get_yticklabels()]

    # Newer:
    if not ticks:
        [a.set_xticks([]) for a in axs]
        [a.set_yticks([]) for a in axs]
    # New:
    plt.tight_layout()

    if show:
        plt.show()
        plt.clf()
    else:
        return fig


hd = 24.64  # mm. Smallest detector dimension
hm = 9.92
b = 22  # in mm. Detector-mask-distance

# Disable any GPU:
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# +++++++++++++++++++++++++++++++++++++++++++++ Sphere_central_0_1mm +++++++++++++++++++++++++++++++++++++++++++++
sg = EllipsoidSourceGenerator(448, size_xy_mm=63.2 * hd / b, size_z_mm=10 * 0.1, centers_mm=[0, 0, 63.2],
                              radii_mm=[0.1, 0.1, 0.1], intens=[1, ], origin_mm=[0, 0, 63.2])
sg.draw_sketch(b, hd, hm=hm).show()
sg.convert_cube_to_pyramide(b=b, hd=hd)

# Load Topas simulation result:
topas_img = np.load("TOPAS_sphere_central_0_1mm.npy")

# --- Simulation ---
mask = MuraMaskObject(style_or_rank="russo2020", size_mm=9.92, thickness=0.11, aperture_shape="circle",
                      aperture_radius_mm=0.08 / 2.0)
detector = DetectorObject(size_px=448, size_mm=min(24.64, 28.16))
t0 = time.time()
sim = Simulation(Mask=mask, Detector=detector, a=sg.origin_mm[2], b=b, photon_count=topas_img.sum(), Source=sg,
                 simulation_name="sim",
                 non_central_cos3_effect=True,
                 non_central_coll_effect=True,
                 transmission_rate=0.01)
print("Elapsed time: %.2fs." % (time.time() - t0))
dect = sim.Detector_image_noisy

# --- Plotting ---
n_msssim = tf.image.ssim_multiscale(norm(topas_img[None, ..., None]), norm(dect[None, ..., None]), 1.0)
fig = triplot([topas_img, dect, topas_img - dect], suptitle="Sphere_central_0_1mm",
              titles=["TOPAS Simulation: %s" % format(int(topas_img.sum()), ","),
                      "ConvSim: %s" % format(int(dect.sum()), ","),
                      "nMS-SSIM: %.2f" % n_msssim], ticks=False, show=False)
plt.text(10, 435, "%.2f" % n_msssim, c="white", fontsize=30)
fig.savefig("Sphere_central_0_1mm.png")

# +++++++++++++++++++++++++++++++++++++++++++++ Sphere_central_1mm +++++++++++++++++++++++++++++++++++++++++++++
sg = EllipsoidSourceGenerator(448, size_xy_mm=63.2 * hd / b, size_z_mm=10 * 1.0, centers_mm=[0, 0, 63.2],
                              radii_mm=[1.0, 1.0, 1.0], intens=[1, ], origin_mm=[0, 0, 63.2])
sg.draw_sketch(b, hd, hm=hm).show()
sg.convert_cube_to_pyramide(b=b, hd=hd)

# Load Topas simulation result:
topas_img = np.load("TOPAS_sphere_central_1mm.npy")

# --- Simulation ---
mask = MuraMaskObject(style_or_rank="russo2020", size_mm=9.92, thickness=0.11, aperture_shape="circle",
                      aperture_radius_mm=0.08 / 2.0)
detector = DetectorObject(size_px=448, size_mm=min(24.64, 28.16))
t0 = time.time()
sim = Simulation(Mask=mask, Detector=detector, a=sg.origin_mm[2], b=b, photon_count=topas_img.sum(), Source=sg,
                 simulation_name="sim",
                 non_central_cos3_effect=True,
                 non_central_coll_effect=True,
                 transmission_rate=0.01)
print("Elapsed time: %.2fs." % (time.time() - t0))
dect = sim.Detector_image_noisy

# --- Plotting ---
n_msssim = tf.image.ssim_multiscale(norm(topas_img[None, ..., None]), norm(dect[None, ..., None]), 1.0)
fig = triplot([topas_img, dect, topas_img - dect], suptitle="Sphere_central_1mm",
              titles=["TOPAS Simulation: %s" % format(int(topas_img.sum()), ","),
                      "ConvSim: %s" % format(int(dect.sum()), ","),
                      "nMS-SSIM: %.2f" % n_msssim], ticks=False, show=False)
plt.text(10, 435, "%.2f" % n_msssim, c="white", fontsize=30)
fig.savefig("Sphere_central_1mm.png")

# +++++++++++++++++++++++++++++++++++++++++++++ Sphere_central_10mm +++++++++++++++++++++++++++++++++++++++++++++
sg = EllipsoidSourceGenerator(448, size_xy_mm=63.2 * hd / b, size_z_mm=10 * 10.0, centers_mm=[0, 0, 63.2],
                              radii_mm=[10, 10, 10], intens=[1, ], origin_mm=[0, 0, 63.2])
sg.draw_sketch(b, hd, hm=hm).show()
sg.convert_cube_to_pyramide(b=b, hd=hd)

# Load Topas simulation result:
topas_img = np.load("TOPAS_sphere_central_10mm.npy")

# --- Simulation ---
mask = MuraMaskObject(style_or_rank="russo2020", size_mm=9.92, thickness=0.11, aperture_shape="circle",
                      aperture_radius_mm=0.08 / 2.0)
detector = DetectorObject(size_px=448, size_mm=min(24.64, 28.16))
t0 = time.time()
sim = Simulation(Mask=mask, Detector=detector, a=sg.origin_mm[2], b=b, photon_count=topas_img.sum(), Source=sg,
                 simulation_name="sim",
                 non_central_cos3_effect=True,
                 non_central_coll_effect=True,
                 transmission_rate=0.01)
print("Elapsed time: %.2fs." % (time.time() - t0))
dect = sim.Detector_image_noisy

# --- Plotting ---
n_msssim = tf.image.ssim_multiscale(norm(topas_img[None, ..., None]), norm(dect[None, ..., None]), 1.0)
fig = triplot([topas_img, dect, topas_img - dect], suptitle="Sphere_central_10mm",
              titles=["TOPAS Simulation: %s" % format(int(topas_img.sum()), ","),
                      "ConvSim: %s" % format(int(dect.sum()), ","),
                      "nMS-SSIM: %.2f" % n_msssim], ticks=False, show=False)
plt.text(10, 435, "%.2f" % n_msssim, c="white", fontsize=30)
fig.savefig("Sphere_central_10mm.png")

# +++++++++++++++++++++++++++++++++++++++++++++ Sphere_close_0_1mm +++++++++++++++++++++++++++++++++++++++++++++
# I have to switch X and Y again -.-
sg = EllipsoidSourceGenerator(448, size_xy_mm=15 * hd / b, size_z_mm=10 * 0.1, centers_mm=[7, 2, 15],
                              radii_mm=[0.1, 0.1, 0.1], intens=[1, ], origin_mm=[0, 0, 15])
# sg.draw_sketch(b, hd, hm=hm).show()
sg.convert_cube_to_pyramide(b=b, hd=hd)
# plotp(sg.source_array).show()

# Load Topas simulation result:
topas_img = np.load("TOPAS_sphere_close_0_1mm.npy")

# --- Simulation ---
mask = MuraMaskObject(style_or_rank="russo2020", size_mm=9.92, thickness=0.11, aperture_shape="circle",
                      aperture_radius_mm=0.08 / 2.0)
detector = DetectorObject(size_px=448, size_mm=min(24.64, 28.16))
t0 = time.time()
sim = Simulation(Mask=mask, Detector=detector, a=sg.origin_mm[2], b=b, photon_count=topas_img.sum(), Source=sg,
                 simulation_name="sim",
                 non_central_cos3_effect=True,
                 non_central_coll_effect=True,
                 transmission_rate=0.01)
print("Elapsed time: %.2fs." % (time.time() - t0))
dect = sim.Detector_image_noisy

# --- Plotting ---
n_msssim = tf.image.ssim_multiscale(norm(topas_img[None, ..., None]), norm(dect[None, ..., None]), 1.0)
fig = triplot([topas_img, dect, topas_img - dect], suptitle="Sphere_close_0_1mm",
              titles=["TOPAS Simulation: %s" % format(int(topas_img.sum()), ","),
                      "ConvSim: %s" % format(int(dect.sum()), ","),
                      "nMS-SSIM: %.2f" % n_msssim], ticks=False, show=False)
plt.text(10, 435, "%.2f" % n_msssim, c="white", fontsize=30)
fig.savefig("Sphere_close_0_1mm.png")

# +++++++++++++++++++++++++++++++++++++++++++++ Sphere_close_1mm +++++++++++++++++++++++++++++++++++++++++++++
# I have to switch X and Y again -.-
sg = EllipsoidSourceGenerator(448, size_xy_mm=15 * hd / b, size_z_mm=10 * 1.0, centers_mm=[7, 2, 15],
                              radii_mm=[1, 1, 1], intens=[1, ], origin_mm=[0, 0, 15])
# sg.draw_sketch(b, hd, hm=hm).show()
sg.convert_cube_to_pyramide(b=b, hd=hd)
# plotp(sg.source_array).show()

# Load Topas simulation result:
topas_img = np.load("TOPAS_sphere_close_1mm.npy")

# --- Simulation ---
mask = MuraMaskObject(style_or_rank="russo2020", size_mm=9.92, thickness=0.11, aperture_shape="circle",
                      aperture_radius_mm=0.08 / 2.0)
detector = DetectorObject(size_px=448, size_mm=min(24.64, 28.16))
t0 = time.time()
sim = Simulation(Mask=mask, Detector=detector, a=sg.origin_mm[2], b=b, photon_count=topas_img.sum(), Source=sg,
                 simulation_name="sim",
                 non_central_cos3_effect=True,
                 non_central_coll_effect=True,
                 transmission_rate=0.01)
print("Elapsed time: %.2fs." % (time.time() - t0))
dect = sim.Detector_image_noisy

# --- Plotting ---
n_msssim = tf.image.ssim_multiscale(norm(topas_img[None, ..., None]), norm(dect[None, ..., None]), 1.0)
fig = triplot([topas_img, dect, topas_img - dect], suptitle="Sphere_close_1mm",
              titles=["TOPAS Simulation: %s" % format(int(topas_img.sum()), ","),
                      "ConvSim: %s" % format(int(dect.sum()), ","),
                      "nMS-SSIM: %.2f" % n_msssim], ticks=False, show=False)
plt.text(10, 435, "%.2f" % n_msssim, c="white", fontsize=30)
fig.savefig("Sphere_close_1mm.png")

# +++++++++++++++++++++++++++++++++++++++++++++ Sphere_close_10mm +++++++++++++++++++++++++++++++++++++++++++++
# I have to switch X and Y again -.-
sg = EllipsoidSourceGenerator(448, size_xy_mm=15 * hd / b, size_z_mm=100, centers_mm=[7, 2, 15], radii_mm=[10, 10, 10],
                              intens=[1, ], origin_mm=[0, 0, 55])
# sg.draw_sketch(b, hd, hm=hm).show()
sg.convert_cube_to_pyramide(b=b, hd=hd)
# plotp(sg.source_array).show()

# Load Topas simulation result:
topas_img = np.load("TOPAS_sphere_close_10mm.npy")

# --- Simulation ---
mask = MuraMaskObject(style_or_rank="russo2020", size_mm=9.92, thickness=0.11, aperture_shape="circle",
                      aperture_radius_mm=0.08 / 2.0)
detector = DetectorObject(size_px=448, size_mm=min(24.64, 28.16))
t0 = time.time()
sim = Simulation(Mask=mask, Detector=detector, a=sg.origin_mm[2], b=b, photon_count=topas_img.sum(), Source=sg,
                 simulation_name="sim",
                 non_central_cos3_effect=True,
                 non_central_coll_effect=True,
                 transmission_rate=0.01)
print("Elapsed time: %.2fs." % (time.time() - t0))
dect = sim.Detector_image_noisy

# --- Plotting ---
n_msssim = tf.image.ssim_multiscale(norm(topas_img[None, ..., None]), norm(dect[None, ..., None]), 1.0)
fig = triplot([topas_img, dect, topas_img - dect], suptitle="Sphere_close_10mm",
              titles=["TOPAS Simulation: %s" % format(int(topas_img.sum()), ","),
                      "ConvSim: %s" % format(int(dect.sum()), ","),
                      "nMS-SSIM: %.2f" % n_msssim], ticks=False, show=False)
plt.text(10, 435, "%.2f" % n_msssim, c="white", fontsize=30)
fig.savefig("Sphere_close_10mm.png")

# +++++++++++++++++++++++++++++++++++++++++++++ Sphere_far_0_1mm +++++++++++++++++++++++++++++++++++++++++++++
# I have to switch X and Y again -.-
sg = EllipsoidSourceGenerator(448, size_xy_mm=190 * hd / b, size_z_mm=10 * 1.0, centers_mm=[-20, -70, 190],
                              radii_mm=[0.1, 0.1, 0.1], intens=[1, ], origin_mm=[0, 0, 190])
# sg.draw_sketch(b, hd, hm=hm).show()
sg.convert_cube_to_pyramide(b=b, hd=hd)
# plotp(sg.source_array).show()

# Load Topas simulation result:
topas_img = np.load("TOPAS_sphere_far_0_1mm.npy")

# --- Simulation ---
mask = MuraMaskObject(style_or_rank="russo2020", size_mm=9.92, thickness=0.11, aperture_shape="circle",
                      aperture_radius_mm=0.08 / 2.0)
detector = DetectorObject(size_px=448, size_mm=min(24.64, 28.16))
t0 = time.time()
sim = Simulation(Mask=mask, Detector=detector, a=sg.origin_mm[2], b=b, photon_count=topas_img.sum(), Source=sg,
                 simulation_name="sim",
                 non_central_cos3_effect=True,
                 non_central_coll_effect=True,
                 transmission_rate=0.01)
print("Elapsed time: %.2fs." % (time.time() - t0))
dect = sim.Detector_image_noisy

# --- Plotting ---
n_msssim = tf.image.ssim_multiscale(norm(topas_img[None, ..., None]), norm(dect[None, ..., None]), 1.0)
fig = triplot([topas_img, dect, topas_img - dect], suptitle="Sphere_far_0_1mm",
              titles=["TOPAS Simulation: %s" % format(int(topas_img.sum()), ","),
                      "ConvSim: %s" % format(int(dect.sum()), ","),
                      "nMS-SSIM: %.2f" % n_msssim], ticks=False, show=False)
plt.text(10, 435, "%.2f" % n_msssim, c="white", fontsize=30)
fig.savefig("Sphere_far_0_1mm.png")

# +++++++++++++++++++++++++++++++++++++++++++++ Sphere_far_1mm +++++++++++++++++++++++++++++++++++++++++++++
# I have to switch X and Y again -.-
sg = EllipsoidSourceGenerator(448, size_xy_mm=190 * hd / b, size_z_mm=10 * 1.0, centers_mm=[-20, -70, 190],
                              radii_mm=[1, 1, 1], intens=[1, ], origin_mm=[0, 0, 190])
# sg.draw_sketch(b, hd, hm=hm).show()
sg.convert_cube_to_pyramide(b=b, hd=hd)
# plotp(sg.source_array).show()

# Load Topas simulation result:
topas_img = np.load("TOPAS_sphere_far_1mm.npy")

# --- Simulation ---
mask = MuraMaskObject(style_or_rank="russo2020", size_mm=9.92, thickness=0.11, aperture_shape="circle",
                      aperture_radius_mm=0.08 / 2.0)
detector = DetectorObject(size_px=448, size_mm=min(24.64, 28.16))
t0 = time.time()
sim = Simulation(Mask=mask, Detector=detector, a=sg.origin_mm[2], b=b, photon_count=topas_img.sum(), Source=sg,
                 simulation_name="sim",
                 non_central_cos3_effect=True,
                 non_central_coll_effect=True,
                 transmission_rate=0.01)
print("Elapsed time: %.2fs." % (time.time() - t0))
dect = sim.Detector_image_noisy

# --- Plotting ---
n_msssim = tf.image.ssim_multiscale(norm(topas_img[None, ..., None]), norm(dect[None, ..., None]), 1.0)
fig = triplot([topas_img, dect, topas_img - dect], suptitle="Sphere_far_1mm",
              titles=["TOPAS Simulation: %s" % format(int(topas_img.sum()), ","),
                      "ConvSim: %s" % format(int(dect.sum()), ","),
                      "nMS-SSIM: %.2f" % n_msssim], ticks=False, show=False)
plt.text(10, 435, "%.2f" % n_msssim, c="white", fontsize=30)
fig.savefig("Sphere_far_1mm.png")

# +++++++++++++++++++++++++++++++++++++++++++++ Sphere_far_10mm +++++++++++++++++++++++++++++++++++++++++++++
# I have to switch X and Y again -.-
sg = EllipsoidSourceGenerator(448, size_xy_mm=190 * hd / b, size_z_mm=10 * 10.0, centers_mm=[-20, -70, 190],
                              radii_mm=[10, 10, 10], intens=[1, ], origin_mm=[0, 0, 190])
# sg.draw_sketch(b, hd, hm=hm).show()
sg.convert_cube_to_pyramide(b=b, hd=hd)
# plotp(sg.source_array).show()

# Load Topas simulation result:
topas_img = np.load("TOPAS_sphere_far_10mm.npy")

# --- Simulation ---
mask = MuraMaskObject(style_or_rank="russo2020", size_mm=9.92, thickness=0.11, aperture_shape="circle",
                      aperture_radius_mm=0.08 / 2.0)
detector = DetectorObject(size_px=448, size_mm=min(24.64, 28.16))
t0 = time.time()
sim = Simulation(Mask=mask, Detector=detector, a=sg.origin_mm[2], b=b, photon_count=topas_img.sum(), Source=sg,
                 simulation_name="sim",
                 non_central_cos3_effect=True,
                 non_central_coll_effect=True,
                 transmission_rate=0.01)
print("Elapsed time: %.2fs." % (time.time() - t0))
dect = sim.Detector_image_noisy

# --- Plotting ---
n_msssim = tf.image.ssim_multiscale(norm(topas_img[None, ..., None]), norm(dect[None, ..., None]), 1.0)
fig = triplot([topas_img, dect, topas_img - dect], suptitle="Sphere_far_10mm",
              titles=["TOPAS Simulation: %s" % format(int(topas_img.sum()), ","),
                      "ConvSim: %s" % format(int(dect.sum()), ","),
                      "nMS-SSIM: %.2f" % n_msssim], ticks=False, show=False)
plt.text(10, 435, "%.2f" % n_msssim, c="white", fontsize=30)
fig.savefig("Sphere_far_10mm.png")



