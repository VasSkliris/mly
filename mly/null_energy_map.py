"""
    This module is a part of the mly machine learning package which allows for
    the creation of null_energy_maps using a mly plugin. Null energy maps can be
    used for sky localisation. The healpix coordinate system is used throughout.
"""

import tempfile
from math import *
import tensorflow as tf
import healpy as hp
import numpy as np
from gwpy.timeseries import TimeSeries
from pycbc.detector import Detector
from ligo.skymap.io import fits
from .projectwave import *
from .plugins import *
from .validators import *
from .datatools import *
import matplotlib.pyplot as plt

from scipy.stats import chi2

__author__ = "Wasim Javed, Michael Norman, Vasileios Skliris, Kyle Willetts, and"
"Patrick Sutton."


############################################################################


def GPStoGMST(gps):

    # Function to convert a time in GPS seconds to Greenwich Mean Sidereal Time
    # Necessary for converting RA to longitude

    gps0 = 630763213   # Start of J2000 epoch, Jan 1 2000 at 12:00 UTC

    D = (gps - gps0) / 86400.0   # Days since J2000 

    # Days between J2000 and last 0h at Greenwich (always integer + 1/2)
    d_u = np.floor(D) + 0.5

    df_u = D - d_u

    T_u = d_u / 36525.0
    gmst0h = 24110.54841 + 8640184.812866 * \
        T_u + 0.093104 * T_u**2 - 6.2e-6 * T_u**3

    gmst = gmst0h + 1.00273790935 * 86400 * df_u

    if (gmst >= 86400):
        gmst = gmst - floor(gmst / 86400) * 86400

    return gmst


def radectoearth(ra, dec, gps):

    # Script to convert Right Ascension, Declination to earth-centered coordinates
    #
    # phi is azimuthal angle (longitude), -pi <= phi < pi
    # theta is polar angle (latitude),   0 <= theta <= pi
    #
    # Inputs (RA,dec) are in degrees, RA from zero (GMT) to 360, dec from 90
    # (north pole) to -90 (south pole)

    gmst = GPStoGMST(gps)  # get time in sidereal day

    gmst_deg = gmst / 86400.0 * 360.0   # convert time in sidereal day to

    # longitude is right ascension minus location of midnight relative to
    # greenwich
    phi_deg = ra - gmst_deg
    # latitude is the same as declination, but theta (earth-centered
    # coordinate) is zero
    theta_deg = 90 - dec
    # at the north pole and pi at the south pole

    # convert to radians
    phi = phi_deg * (2 * pi / 360)
    # make sure phi is -pi to pi, not zero to 2*pi, which would make sense but
    # whatever
    phi = fmod(phi + pi, 2 * pi) - pi
    theta = theta_deg * 2 * pi / 360

    return phi, theta


def earthtoradec(phi, theta, gps):

    # Script to convert Right Ascension, Declination to earth-centered coordinates
    #
    # phi is azimuthal angle (longitude), -pi <= phi < pi
    # theta is polar angle (latitude),   0 <= theta <= pi
    #
    # Inputs (RA,dec) are in degrees, RA from zero (GMT) to 360, dec from 90
    # (north pole) to -90 (south pole)

    gmst = GPStoGMST(gps)  # get time in sidereal day

    gmst_deg = gmst / 86400.0 * 360.0   # convert time in sidereal day to

    # longitude is right ascension minus location of midnight relative to
    # greenwich
    ra = (phi * (180 / pi)) + gmst_deg
    # latitude is the same as declination, but theta (earth-centered
    # coordinate) is zero
    dec = 90 - (theta * (180 / pi))
    # at the north pole and pi at the south pole

    # convert to degrees
    #phi = phi_deg * (2 * pi / 360)
    # make sure phi is -pi to pi, not zero to 2*pi, which would make sense but
    # whatever
    ra = np.mod(ra, 360)
    #theta = theta_deg * 2 * pi / 360

    return ra * (np.pi / 180), dec * (np.pi / 180)


def timeDelayShiftTensorflow(strain, freq, shift):    # sample frequency

    # ---- FFT the original timeseries.
    strainFFT = tf.signal.rfft(strain)
    phaseFactor = tf.exp(
        tf.complex(
            tf.cast(
                0.0, tf.float64), -2 * np.pi * freq * shift))
    
    return phaseFactor * strainFFT


# Convert function to tensorflow graph:
tf_fft = tf.function(timeDelayShiftTensorflow)

def calculateEnergies(
    strain, 
    noise_spectrum,
    frequency_axis,
    dt,
    null_vector,
    num_pixels,
    num_detectors,
    num_samples,
    fs
):
    # Reshape tensors into compatible shape for operation:
    strain = tf.reshape(
        strain,
        [1, num_detectors, num_samples]
    )
    frequency_axis = tf.reshape(
        frequency_axis,
        [1, 1, (int)(num_samples / 2) + 1]
    )
    dt = tf.reshape(
        dt,
        [num_pixels, num_detectors, 1]
    )
    null_vector = tf.reshape(
        null_vector,
        [num_pixels, num_detectors, 1]
    )
    noise_spectrum = tf.reshape(
        noise_spectrum,
        [1, num_detectors, (int)(num_samples / 2) + 1]
    )

    # Calculate shifted positions using phase correction:
    shifted_strain = tf_fft(strain, frequency_axis, dt)
    
    shifted_strain = tf.scalar_mul(
        tf.cast((1.0/fs),tf.complex128), shifted_strain
    )
    
    # Multiply by the null_vector:
    null_stream = \
        tf.math.multiply(shifted_strain, tf.cast(null_vector, tf.complex128))
    
    # Calculate map:
    null_stream_noise_spectrum = \
        tf.reduce_sum(
            tf.math.multiply(tf.math.square(null_vector), noise_spectrum),
            axis = 1)
    
    coherent_null_energy = tf.divide(
        tf.math.square(
            tf.math.abs(
                tf.math.reduce_sum(
                    null_stream,
                    axis=1
                ))
            ),
        null_stream_noise_spectrum)
    
    coherent_null_energy = tf.math.reduce_sum(coherent_null_energy, axis=1) 
    coherent_null_energy = tf.math.scalar_mul(tf.cast((2.0*num_samples),tf.float64), coherent_null_energy)
    
    incoherent_null_energy = tf.divide(
        tf.math.reduce_sum(
            tf.math.square(
                tf.math.abs(
                    null_stream
                )
            ),
            axis=1
        ),
        null_stream_noise_spectrum)
    
    incoherent_null_energy = tf.math.reduce_sum(incoherent_null_energy, axis=1) 
    incoherent_null_energy = tf.math.scalar_mul(tf.cast((2.0*num_samples),tf.float64), incoherent_null_energy)
    
    energy       = tf.math.reduce_sum(tf.divide(tf.math.square(tf.math.abs(shifted_strain[0])), noise_spectrum[0]), axis = 0)
    total_energy = tf.math.reduce_sum(energy)
    total_energy = tf.math.scalar_mul(tf.cast((2.0*num_samples),tf.float64), total_energy)
    
    return coherent_null_energy, incoherent_null_energy, total_energy

# Convert function to tensroflow graph:
calc_energies = tf.function(calculateEnergies)

def returnNULLVector(num_pixels, theta, phi, detectors, gps_time):
    
    num_detectors = len(detectors)
    
    null_vector                = np.zeros([num_pixels, num_detectors])
    antenna_sensitivity_factor = np.zeros(num_pixels)
    for pixel_index in range(num_pixels):

        ra, dec = earthtoradec(phi[pixel_index], theta[pixel_index], gps_time)

        fp = np.zeros(num_detectors)
        fc = np.zeros(num_detectors)

        for detector_index, detector in enumerate(detectors):
            fp[detector_index], fc[detector_index] = detector.antenna_pattern(
                ra, dec, 0, gps_time)
            
        antenna_sensitivity_factor[pixel_index] = np.sum((fp*fp + fc*fc)**1.5)
            
        K = np.cross(fp, fc)
        norm_K = np.linalg.norm(K)
        null_vector[pixel_index] = (K / (norm_K))
    
    return null_vector, antenna_sensitivity_factor


def returnTimeDelayVector(num_pixels, theta, phi, detectors, gps_time):

    dt_vector = np.zeros([num_pixels, len(detectors)])
    for pixel_index in range(num_pixels):
        ra, dec = earthtoradec(phi[pixel_index], theta[pixel_index], gps_time)

        for detector_index, detector in enumerate(detectors):
            dt_vector[pixel_index][detector_index] = - \
                detector.time_delay_from_detector(detectors[0], ra, dec, gps_time)

    return dt_vector


def signaltoskymap(
    strain,
    dt_vector,
    frequency_axis,
    num_pixels,
    num_detectors,
    num_samples,
    null_vector,
    antenna_sensitivity_factor,
    fs
):
    
    noise_spectrum = np.empty([num_detectors, len(frequency_axis)])    
    for i, ts in enumerate(strain):        
        noise_spectrum[i] = plt.psd(ts, NFFT=num_samples)[0]
            
    background_samples = len(strain[0])
    strain_start = int((background_samples-num_samples)/2)
    strain_end  = int((background_samples+num_samples)/2)
        
    strain = strain[:,strain_start:strain_end]
        
    # Convert required arrays into tensorflow tensors:
    frequency_axis = tf.convert_to_tensor(frequency_axis)
    strain = tf.convert_to_tensor(strain)
    dt_vector = tf.convert_to_tensor(dt_vector)
    null_vector = tf.convert_to_tensor(null_vector)
    noise_spectrum = tf.convert_to_tensor(noise_spectrum)

    # Run tensorflow graph:
    coherent_null_energy, incoherent_null_energy, total_energy = calc_energies(
        strain,
        noise_spectrum,
        frequency_axis,
        dt_vector,
        null_vector,
        num_pixels,
        num_detectors,
        num_samples,
        fs
    )
    
    energy_combination = tf.math.multiply(
                        (1.0 - tf.math.divide(
                            coherent_null_energy, 
                            incoherent_null_energy
                        )),
                        (total_energy - coherent_null_energy)
                      )   
    
    alpha = 1
    scale_factor = fs*alpha
    
    antenna_sensitivity_factor = tf.convert_to_tensor(antenna_sensitivity_factor)
    probability_map = tf.multiply(
                        tf.exp(
                          tf.math.divide(
                            (energy_combination - tf.math.reduce_max(energy_combination)),
                            tf.cast(scale_factor, tf.float64)
                           )
                        ),
                        antenna_sensitivity_factor
                      )

    probability_map = probability_map.numpy()
#     coherent_null_energy = coherent_null_energy.numpy()
#     incoherent_null_energy = incoherent_null_energy.numpy()
    
#     plt.figure()
#     hp.mollview(coherent_null_energy, coord='C')
#     plt.savefig("coherent_null_energy.png")
    
#     plt.figure()
#     hp.mollview(incoherent_null_energy, coord='C')
#     plt.savefig("incoherent_null_energy.png")
    
#     plt.figure()
#     hp.mollview(incoherent_null_energy, coord='C')
#     plt.savefig("probability_map.png")
    
#     print(f"Total Energy: {total_energy}")
        
    return probability_map


def plotsignaltoskymap(strain, data=None):
    
    null_energy_map = hp.mollview(data, coord='C')
    return null_energy_map


def createSkymapPlugin(nside, fs, duration, pluginName = "skymap"):

        # Unpack config:
    num_samples = fs * duration

    # Reference GPS so that GMST is within 1 second of 0 to make equivilent of
    # earth centre coordinates:
    gps_time = 1337069300

    # Calculate num pixels from nside:
    num_pixels = hp.nside2npix(nside)

    # Setup detector array:
    detectors = []
    for initial in 'HLV':
        detectors.append(Detector(f"{initial}1"))

    # Assign number of detectors:
    num_detectors = len('HLV')

    # Create theta and phi arrays:
    theta, phi = hp.pix2ang(nside, range(num_pixels))

    # Create Antenna and TimeDelay maps:
    null_vector, antenna_sensitivity_factor = returnNULLVector(num_pixels, theta, phi, detectors, gps_time)
    dt_vector = returnTimeDelayVector(
        num_pixels, theta, phi, detectors, gps_time)
    frequency_axis = np.fft.rfftfreq(num_samples, d=1 / fs)

    # Create mly plugin:
    skymap_plugin = PlugIn(
        name="skymap",
        genFunction=signaltoskymap,
        attributes=['strain'],
        plotAttributes=['strain'],
        plotFunction=plotsignaltoskymap,
        dt_vector=dt_vector,
        frequency_axis=frequency_axis,
        num_pixels=num_pixels,
        num_detectors=num_detectors,
        num_samples=num_samples,
        null_vector=null_vector,
        antenna_sensitivity_factor=antenna_sensitivity_factor,
        fs=fs
    )

    return skymap_plugin

def saveFitsFile(pod, file_path):
    null_energy_map = pod.null_energy_map

    with open(file_path, "w") as f:
        fits.write_sky_map(f.name, null_energy_map, nest=False)
