# Construction of null-stream sky map using unwhitened strain data.

from tqdm import tqdm
#import tensorflow_probability as tfp
import tensorflow as tf
import time
from scipy.stats import norm
import healpy as hp
import pycbc.psd
import pycbc.noise
from gwpy.timeseries import TimeSeries
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from pycbc.detector import Detector, get_available_detectors
from pycbc.waveform import get_td_waveform
from math import *
import numpy as np
from .projectwave import *
from .plugins import *
from .datatools import *
import sys

# GPS > GPS, GMST > GMST, ra = RA, dec = declination


def GPStoGMST(GPS):

    # Function to convert a time in GPS seconds to Greenwich Mean Sidereal Time
    # Necessary for converting RA to longitude

    GPS0 = 630763213   # Start of J2000 epoch, Jan 1 2000 at 12:00 UTC

    D = (GPS-GPS0)/86400.0   # Days since J2000

    # Days between J2000 and last 0h at Greenwich (always integer + 1/2)
    d_u = np.floor(D)+0.5

    df_u = D-d_u

    T_u = d_u/36525.0
    GMST0h = 24110.54841 + 8640184.812866*T_u + 0.093104*T_u**2 - 6.2e-6*T_u**3

    GMST = GMST0h + 1.00273790935*86400*df_u

    if (GMST >= 86400):
        GMST = GMST - floor(GMST/86400)*86400

    return GMST


def RAdectoearth(RA, dec, GPS):

    # Script to convert Right Ascension, Declination to earth-centered coordinates
    #
    # phi is azimuthal angle (longitude), -pi <= phi < pi
    # theta is polar angle (latitude),   0 <= theta <= pi
    #
    # Inputs (RA,dec) are in degrees, RA from zero (GMT) to 360, dec from 90 (north pole) to -90 (south pole)

    GMST = GPStoGMST(GPS)  # get time in sidereal day

    GMST_deg = GMST / 86400.0 * 360.0   # convert time in sidereal day to

    # longitude is right ascension minus location of midnight relative to greenwich
    phi_deg = RA - GMST_deg
    # latitude is the same as declination, but theta (earth-centered coordinate) is zero
    theta_deg = 90 - dec
    # at the north pole and pi at the south pole

    # convert to radians
    phi = phi_deg * (2 * pi / 360)
    # make sure phi is -pi to pi, not zero to 2*pi, which would make sense but whatever
    phi = fmod(phi+pi, 2*pi)-pi
    theta = theta_deg * 2 * pi / 360

    return phi, theta


def earthtoRAdec(phi, theta, GPS):

    # Script to convert Right Ascension, Declination to earth-centered coordinates
    #
    # phi is azimuthal angle (longitude), -pi <= phi < pi
    # theta is polar angle (latitude),   0 <= theta <= pi
    #
    # Inputs (RA,dec) are in degrees, RA from zero (GMT) to 360, dec from 90 (north pole) to -90 (south pole)

    GMST = GPStoGMST(GPS)  # get time in sidereal day

    GMST_deg = GMST / 86400.0 * 360.0   # convert time in sidereal day to

    # longitude is right ascension minus location of midnight relative to greenwich
    RA = (phi*(180/pi)) + GMST_deg
    # latitude is the same as declination, but theta (earth-centered coordinate) is zero
    dec = 90 - (theta*(180/pi))
    # at the north pole and pi at the south pole

    # convert to degrees
    #phi = phi_deg * (2 * pi / 360)
    # make sure phi is -pi to pi, not zero to 2*pi, which would make sense but whatever
    RA = np.mod(RA, 360)
    #theta = theta_deg * 2 * pi / 360

    return RA*(np.pi/180), dec*(np.pi/180)


def timeDelayShiftTensorflow(strain, freq, shift):    # sample frequency

    strainFFT = tf.signal.rfft(strain)
    phaseFactor = tf.exp(tf.complex(
        tf.cast(0.0, tf.float64), -2*np.pi*freq*shift))
    y = tf.math.multiply(phaseFactor, tf.cast(strainFFT, tf.complex128))

    #CONFLICT
    return tf.math.real(y)


#CONFLICT this needs to change
#tf_fft = tf.function(timeDelayShiftTensorflow)
timeDelayShiftTensorflow = tf.function(timeDelayShiftTensorflow)


#shape of the output of this fuction has changed to [num_pixels, 1]
def antennaResponseRMS(num_pixels, theta, phi, detectors, GPS_time):

    antenna_response_rms = np.zeros([num_pixels])
    for pixel_index in range(num_pixels):

        RA, dec = earthtoRAdec(phi[pixel_index], theta[pixel_index], GPS_time)

        fp = np.zeros(len(detectors))
        fc = np.zeros(len(detectors))

        for detector_index, detector in enumerate(detectors):
            fp[detector_index], fc[detector_index] = detector.antenna_pattern(
                RA, dec, 0, GPS_time)
#        print('fp', fp.shape)
#        print('fc', fc.shape)
        totalResponse = np.square(np.abs(fp)) + np.square(np.abs(fc))
#        print('totalResponse', totalResponse.shape)
        #sum of square of fp and fc over all the detectors
        antenna_response_rms[pixel_index] = (np.sum(totalResponse))**0.5

    #print(antenna_response_rms)
    #print('antenna_response_rms', antenna_response_rms.shape)

    return antenna_response_rms


#this is a tensorflow graph
def EnergySkyMapsGRF(
    strain,
    frequency_axis,
    noise_psd,
    time_delay,
    null_coefficient,
    antenna_response_rms,
    num_pixels,
    num_detectors,
    fs
):
    """
    THIS FUNCTION IS A TENSORFLOW GRAPH AND CONVERTS APPLE TO ORANGE 

    Parameters
    ----------

    strain: numpy.ndarray
        The strain data used....
    
    frequency axis: xxxx
        descriptio xxxxxxxx


    """
    #Reshape tensors into compatible shape for operation:
    strain = tf.reshape(
        strain,
        [1, num_detectors, fs]
    )
    frequency_axis = tf.reshape(
        frequency_axis,
        [1, 1, (int)(fs/2) + 1]
    )
    noise_psd = tf.reshape(
        noise_psd,
        [1, num_detectors, len(noise_psd[0])]

    )
    time_delay = tf.reshape(
        time_delay,
        [num_pixels, num_detectors, 1]
    )
    null_coefficient = tf.reshape(
        null_coefficient,
        [num_pixels, num_detectors, 1]
    )
    # antenna_response_rms = tf.reshape(
    #     antenna_response_rms,
    #     [num_pixels, 1, 1] # shape used to be [num_pixels, num_detectors, 1]
    # )
    # print('noise_psd', noise_psd.shape)
    #Calculating spectral density of background noise
    noise_psd = tf.math.reduce_sum(tf.math.multiply(
        tf.math.square(null_coefficient), noise_psd), axis=1)

    #Calculate shifted positions using phase correction:
    shifted_strain = tf.math.divide(timeDelayShiftTensorflow(
        strain, frequency_axis, time_delay), fs)

    #Calculate null_energy by multiplying detectors antenna response with time shifted_strain
    null_stream_components = tf.math.multiply(null_coefficient, shifted_strain)

    #Construct null stream, using Chaterji et al 2006 method
    null_stream = tf.math.reduce_sum(null_stream_components, axis=1)

    #calculating coherent null energy by taking absolute square of the null_stream and dividing it by noise_psd then suming over detectors
    coherentNullEnergy = tf.math.reduce_sum(tf.math.divide(
        tf.math.square(tf.math.abs(null_stream)), noise_psd), axis=1)

    #To calcuate incorehent energy we apply following method
    incoherentNullEnergy = tf.math.reduce_sum(
        tf.math.divide(tf.math.reduce_sum(tf.math.square(tf.math.abs(null_stream_components)), axis=1), noise_psd),         axis=1)

    '''
    fft the data from one detector, modulus square it,
    divide by the noise spectrum for that detector, and sum over all 
    frequencies. Do the same for the other detectors and add together.
    '''

    E_hanford = tf.math.reduce_sum(tf.divide(tf.math.square(
        tf.math.abs(shifted_strain[0][0])), noise_psd[0][0]), axis=0)

    E_livingston = tf.math.reduce_sum(tf.divide(tf.math.square(
        tf.math.abs(shifted_strain[0][1])), noise_psd[0][1]), axis=0)
    E_virgo = tf.math.reduce_sum(tf.divide(tf.math.square(
        tf.math.abs(shifted_strain[0][2])), noise_psd[0][2]), axis=0)

#    print('E_hanford', E_hanford.shape)

    totalEnergy = E_hanford + E_livingston + E_virgo

    return (coherentNullEnergy, incoherentNullEnergy, totalEnergy)


EnergySkyMapsGRF = tf.function(EnergySkyMapsGRF)


def nullCoefficient(num_pixels, theta, phi, detectors, GPS_time):

    null_coefficient = np.zeros([num_pixels, 3])
    for pixel_index in range(num_pixels):

        RA, dec = earthtoRAdec(phi[pixel_index], theta[pixel_index], GPS_time)

        fp = np.zeros(len(detectors))
        fc = np.zeros(len(detectors))

        for detector_index, detector in enumerate(detectors):
            fp[detector_index], fc[detector_index] = detector.antenna_pattern(
                RA, dec, 0, GPS_time)

        cross_product = np.cross(fp, fc)
        norm_cross_product = np.linalg.norm(cross_product)
        null_coefficient[pixel_index] = (cross_product/(norm_cross_product))

    return null_coefficient
# is it used?


def timeDelayMap(num_pixels, theta, phi, detectors, GPS_time):

    time_delay_map = np.zeros([num_pixels, len(detectors)])
    for pixel_index in range(num_pixels):
        RA, dec = earthtoRAdec(phi[pixel_index], theta[pixel_index], GPS_time)

        for detector_index, detector in enumerate(detectors):
            time_delay_map[pixel_index][detector_index] = - \
                detector.time_delay_from_detector(
                    detectors[0], RA, dec, GPS_time)

    return time_delay_map


def EnergySkyMaps(
    strain,
    time_delay_map,
    frequency_axis,
    noise_psd,
    num_pixels,
    num_detectors,
    fs,
    null_coefficient,
    antenna_response_rms
):
    #incorporate this to the EnergySkymaps function

    #Convert to tensorflow tensors:
    frequency_axis = tf.convert_to_tensor(frequency_axis)
    strain = tf.convert_to_tensor(strain)
    time_delay_map = tf.convert_to_tensor(time_delay_map)
    null_coefficient = tf.convert_to_tensor(null_coefficient)
    antenna_response_rms = tf.convert_to_tensor(antenna_response_rms)
    noise_psd = tf.convert_to_tensor(noise_psd)

    #Run tensorflow graph:
    coherentNullEnergy, incoherentNullEnergy, totalEnergy = EnergySkyMapsGRF(
        strain,
        frequency_axis,
        noise_psd,
        time_delay_map,
        null_coefficient,
        antenna_response_rms,
        num_pixels,
        num_detectors,
        fs
    )

    coherentNullEnergy = np.array(coherentNullEnergy)
    incoherentNullEnergy = np.array(incoherentNullEnergy)
    totalEnergy = np.array(totalEnergy)

    return (coherentNullEnergy, incoherentNullEnergy, totalEnergy)






def skymap_gen_function(fs, uwstrain, noise_psd, gps
                        , alpha = None, beta=None, sigma=None
                        , nside = None
                        , **kwargs):

    if alpha is None or beta is None or sigma is None:
        print(alpha, beta, sigma)
        raise ValueError('alpha, beta and sigma must be defined')
    
    sigma = fs/sigma
    
    
    print(alpha, beta, sigma)
    if nside is None:
        nside = 64
    

    
    num_pixels = hp.nside2npix(nside)
    # print(num_pixels)

    start = time.time()

    # ---------------> Could we possibly save this somewhere and only load it?

    detector_initials = ["H1", "L1", "V1"]

    gps_time = gps[0]

    #Setup detector array:
    detectors = []
    for initial in detector_initials:
        detectors.append(Detector(initial))

    num_detectors = len(detector_initials)

    #Create theta and phi arrays:
    theta, phi = hp.pix2ang(nside, range(num_pixels), nest = True)

    #Create Antenna and TimeDelay maps:
    null_coefficient = nullCoefficient(num_pixels,
                                       theta, phi,
                                       detectors,
                                       gps_time)
    
    antenna_response_rms = antennaResponseRMS(
        num_pixels, theta, phi, detectors, gps_time)

    time_delay_map = timeDelayMap(num_pixels, theta, phi, detectors, gps_time)

    frequency_axis = np.fft.rfftfreq(fs, d=1/fs)

    # < ------------------------------------------


    
    prob_map_total = []

    start = time.time()

    sky_map = EnergySkyMaps(uwstrain,
                            time_delay_map, 
                            frequency_axis,
                            noise_psd,
                            num_pixels, 
                            len(detectors), 
                            fs, 
                            null_coefficient,
                            antenna_response_rms)

    sky_null = sky_map[0]
    sky_inc = sky_map[1]

    L_sky = (1-(sky_null/sky_inc)) * (np.max(sky_null) - sky_null)
    
    prob_map = ((antenna_response_rms)**alpha) * np.exp(
                 -((np.amax(L_sky) - L_sky)/sigma)**beta)


    prob_map = (prob_map)/np.sum(prob_map)
    # print('prob_map', prob_map)

    pixel_index = np.argsort(prob_map)[::-1]
    # print('pixel_index', pixel_index)

    sorted_pixels = prob_map[pixel_index]
    # print('sorted_pixels', sorted_pixels)

    unsorted_pixels = np.argsort(pixel_index)
    # print('unsorted_pixels', unsorted_pixels)

    # print('z', z)
    c_est = np.cumsum(sorted_pixels)
    # print('c_est', c_est)

    # print(len(c_est))
    # c_true = c_est - 0.05*np.sin(np.pi*c_est)
    c_true = c_est**0.6  # --------> ???
    #print('c_true', c_true)

    p_true = np.diff(c_true, prepend=0)
    # print('p_true', p_true)    

    corrected_map = p_true[unsorted_pixels]
    # print('corrected_map', corrected_map)
    prob_map_total.append(corrected_map)
    print('skymap generation time: ',time.time() - start)

    prob_map_total = np.array(prob_map_total)
    
    # if isinstance(map_save, str):
    #     file_name = f'{map_save}.npy'

    #     np.save(file_name, prob_map_total)
    
    # if plot_map is True:
    #     plt.figure()
    #     hp.mollview(prob_map_total[0], coord = 'C', nest= None, title = "probability_map")
    #     #plt.savefig(f"prob_skymap_{time.time()}.png")
    
    return(prob_map_total)


def skymap_plot_function(strain,data=None):

    hp.mollview(data[0], coord = 'C', nest= True, title = "probability_map")