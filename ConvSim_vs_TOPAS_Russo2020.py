"""


DISCLAIMER:
This entire software is provided "as is" and no support will be provided.

"""

import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from Simulation import Simulation
from MaskObject import MuraMaskObject
from DetectorObject import DetectorObject
from SourceGenerators import EllipsoidSourceGenerator


def norm(img):
    return (img - img.min()) / (img.max() - img.min())


def triplot(images, titles=None, suptitle="", ticks=True, cmap=None, show=True):
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

topas_img = np.save("TOPAS_sphere_central_1mm.npy")
topas_img = np.save("TOPAS_sphere_central_10mm.npy")
topas_img = np.save("TOPAS_sphere_close_0_1mm.npy")
topas_img = np.save("TOPAS_sphere_close_1mm.npy")
topas_img = np.save("TOPAS_sphere_close_10mm.npy")
topas_img = np.save("TOPAS_sphere_far_0_1mm.npy")
topas_img = np.save("TOPAS_sphere_far_1mm.npy")
topas_img = np.save("TOPAS_sphere_far_10mm.npy")

# +++++++++++++++++++++++++++++++++++++++++++++ Sphere_central_0_1mm +++++++++++++++++++++++++++++++++++++++++++++
sg = EllipsoidSourceGenerator(448, size_xy_mm=63.2 * hd / b, size_z_mm=10 * 0.1, centers_mm=[0, 0, 63.2],
                              radii_mm=[0.1, 0.1, 0.1], intens=[1, ], origin_mm=[0, 0, 63.2])
sg.draw_sketch(b, hd, hm=hm).show()
sg.convert_cube_to_pyramide(b=b, hd=hd)

# Load Topas simulation result:
topas_img = np.save("TOPAS_sphere_central_0_1mm.npy")

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
# plot(sim.Detector_image_noisefree)
# Output is of int32 type:
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
sg.convert_cube_to_pyramide(b=b, hd=hd)

# Load Topas simulation result:
topas_img = np.save("TOPAS_sphere_central_1mm.npy")

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
sg.convert_cube_to_pyramide(b=b, hd=hd)

# Load Topas simulation result:
topas_img = np.save("TOPAS_sphere_central_10mm.npy")

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
topas_img = np.save("TOPAS_sphere_close_0_1mm.npy")

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
topas_img = np.save("TOPAS_sphere_close_1mm.npy")

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
topas_img = np.save("TOPAS_sphere_close_10mm.npy")

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
topas_img = np.save("TOPAS_sphere_far_0_1mm.npy")

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
topas_img = np.save("TOPAS_sphere_far_1mm.npy")

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
topas_img = np.save("TOPAS_sphere_far_10mm.npy")

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
