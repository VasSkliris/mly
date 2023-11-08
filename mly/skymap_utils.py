# Construction of null-stream sky map using unwhitened strain data.

from tqdm import tqdm
import tensorflow as tf
import time
import healpy as hp
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from pycbc.detector import Detector
from math import *
import numpy as np
from .plugins import *
from .datatools import *
from scipy.signal import welch
from scipy.signal import iirnotch, lfilter, butter

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
def antennaResponseRMS(num_pixels, RA, dec, detectors, GPS_time):
    
    antenna_response_rms = np.zeros([num_pixels])
    for pixel_index in range(num_pixels):

        #RA, dec = earthtoRAdec(phi[pixel_index], theta[pixel_index], GPS_time)

        fp = np.zeros(len(detectors))
        fc = np.zeros(len(detectors))

        for detector_index, detector in enumerate(detectors):
            fp[detector_index], fc[detector_index] = detector.antenna_pattern(
                RA[pixel_index], dec[pixel_index], 0, GPS_time)
#        print('fp', fp.shape)
#        print('fc', fc.shape)
        totalResponse = np.square(np.abs(fp)) + np.square(np.abs(fc))
#        print('totalResponse', totalResponse.shape)
        #sum of square of fp and fc over all the detectors
        antenna_response_rms[pixel_index] = (np.sum(totalResponse))**0.5

    #print(antenna_response_rms)
    #print('antenna_response_rms', antenna_response_rms.shape)

    return antenna_response_rms

def mask_window_tf(data_length, fs, start_time, end_time, ramp_duration):
    """
    Creates a windowed mask for a given time series.
    
    Parameters:
    - data_length: Length of the data series.
    - fs: Sampling frequency.
    - start_time: Starting time of the window.
    - end_time: Ending time of the window.
    - ramp_duration: Duration of the ramp (smooth transition) in the window.
    
    Returns:
    - A windowed mask with values between 0 and 1.
    """
    tnp = tf.experimental.numpy
    pi = tnp.pi 
    PI_HALF = pi / 2

    # Convert times to indices and ensure they are float32
    window_start_idx = tf.cast((start_time - ramp_duration) * fs, tf.float32)
    window_end_idx = tf.cast((end_time + ramp_duration) * fs, tf.float32)
    start_idx = tf.cast(start_time * fs, tf.float32)
    end_idx = tf.cast(end_time * fs, tf.float32)
    
    indices = tf.range(data_length, dtype=tf.float32)
    
    # Create the rolling mask for the main windowed region
    main_mask = tf.where((indices >= start_idx) & (indices < end_idx), 1., indices * 0.)
    
    # Create the ramp-up transition
    ramp_up = 0.5 * (1 + tf.math.sin(-PI_HALF + (indices - window_start_idx) / (start_idx - window_start_idx) * pi))
    mask_on = tf.where((indices >= window_start_idx) & (indices < start_idx), ramp_up, 0.)

    # Create the ramp-down transition
    ramp_down = 0.5 * (1 + tf.math.sin(PI_HALF + (indices - end_idx) / (window_end_idx - end_idx) * pi))
    mask_off = tf.where((indices >= end_idx) & (indices < window_end_idx), ramp_down, 0.)

    # Combine the masks
    windowed_data = main_mask + mask_on + mask_off

    return windowed_data


mask_window_tf = tf.function(mask_window_tf)


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
    fs,
    window_parameter = None
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
    
    if window_parameter is not None:
        print('activate window', window_parameter)
        start_time, end_time, ramp_duration = window_parameter
        
        # data_shape = tf.shape(strain)
        # num_det = data_shape[1]
        # data_len = data_shape[2]
        
        # Apply mask_window_tf function on strain data
        windowed_data = mask_window_tf(fs, fs, start_time, end_time, ramp_duration)
        print('windowed_data', windowed_data.shape)
        print ('strain shape', strain.shape)
        print('start_time_util', start_time)
        print('end_time_until', start_time)
    
        
        # Reshape windowed_data to match the shape of strain
        windowed_data_reshaped = tf.reshape(windowed_data, [1, 1, -1])
        #window_shape_broadcast = tf.broadcast_to(windowed_data_reshaped, data_shape)
        
        #print('window_shape_broadcast', window_shape_broadcast.shape)
        
        strain = tf.math.multiply(tf.cast(windowed_data_reshaped, dtype = tf.float64), strain)
        print('windowed_strain', strain)
    else:
        print('window not activated', window_parameter)

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

    return (coherentNullEnergy, incoherentNullEnergy)


EnergySkyMapsGRF = tf.function(EnergySkyMapsGRF)


def nullCoefficient(num_pixels, RA, dec, detectors, GPS_time):
    if len(detectors) < 3:
        dominant_polarisation_p = np.zeros([num_pixels, len(detectors)])
        dominant_polarisation_c = np.zeros([num_pixels, len(detectors)])
        for pixel_index in range(num_pixels):
            
            #RA, dec = earthtoRAdec(phi[pixel_index], theta[pixel_index], GPS_time)

            fp = np.zeros(len(detectors))
            fc = np.zeros(len(detectors))
            
            for detector_index, detector in enumerate(detectors):
                
                fp[detector_index], fc[detector_index] = detector.antenna_pattern(
                    RA[pixel_index], dec[pixel_index], 0, GPS_time)
            
            
            # add the angle on this following Fp2 and Fc2
            Fp2 = np.square(fp[0]) + np.square(fp[1])
            Fc2 = np.square(fc[0]) + np.square(fc[1])
            
            dot_product  = np.dot(fp, fc)
            # dot_product  = np.dot(Fp2, Fc2)
                  
            # compute the dominant polarisation angle 
            dpa = 1/4*(np.arctan2( (2*dot_product) , (Fp2 - Fc2)))

            # Compute antenna response tensor for each pixel and detector
            dominant_polarisation_p[pixel_index] = [np.cos(2*dpa)*fp[0] + sin(2*dpa)*fc[0], np.cos(2*dpa)*fp[1] + sin(2*dpa)*fc[1]]
            dominant_polarisation_c[pixel_index] = [-np.sin(2*dpa)*fp[0] + cos(2*dpa)*fc[0], -np.sin(2*dpa)*fp[1] + cos(2*dpa)*fc[1]]
            
            if pixel_index ==0:
                print('dp', dominant_polarisation_p[0][:3])
                print('dc', dominant_polarisation_c[0][:3])
            
            if np.dot(dominant_polarisation_c[pixel_index], dominant_polarisation_c[pixel_index])> np.dot(dominant_polarisation_p[pixel_index], dominant_polarisation_p[pixel_index]):
                temp = dominant_polarisation_p[pixel_index]
                dominant_polarisation_p[pixel_index] = dominant_polarisation_c[pixel_index]
                dominant_polarisation_c[pixel_index] = -temp
            
        null_coefficient = (np.divide((dominant_polarisation_p[:, 1], - dominant_polarisation_p[:, 0]), np.sqrt(np.square(dominant_polarisation_p[:, 0]) + np.square(dominant_polarisation_p[:, 1]))))
        
        null_coefficient = null_coefficient.T
            
            # norm_dominant_polarisation = np.linalg.norm(dominant_polarisation)
        
    else:
        null_coeff = np.zeros([num_pixels, len(detectors)])
        for pixel_index in range(num_pixels):

            #RA, dec = earthtoRAdec(phi[pixel_index], theta[pixel_index], GPS_time)

            fp = np.zeros(len(detectors))
            fc = np.zeros(len(detectors))

            for detector_index, detector in enumerate(detectors):
                fp[detector_index], fc[detector_index] = detector.antenna_pattern(
                    RA[pixel_index], dec[pixel_index], 0, GPS_time)

            cross_product = np.cross(fp, fc)
            norm_cross_product = np.linalg.norm(cross_product)
            null_coeff[pixel_index] = (cross_product/(norm_cross_product))
        
        null_coefficient = null_coeff

    return null_coefficient



def timeDelayMap(num_pixels, RA, dec, detectors, GPS_time):
    
    time_delay_map = np.zeros([num_pixels, len(detectors)])
    for pixel_index in range(num_pixels):
        
        #RA, dec = earthtoRAdec(phi[pixel_index], theta[pixel_index], GPS_time)

        for detector_index, detector in enumerate(detectors):
            time_delay_map[pixel_index][detector_index] = - \
                detector.time_delay_from_detector(
                    detectors[0], RA[pixel_index], dec[pixel_index], GPS_time)
    
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
    antenna_response_rms,
    window_parameter = None
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
    coherentNullEnergy, incoherentNullEnergy = EnergySkyMapsGRF(
        strain,
        frequency_axis,
        noise_psd,
        time_delay_map,
        null_coefficient,
        antenna_response_rms,
        num_pixels,
        num_detectors,
        fs,
        window_parameter = window_parameter
    )

    coherentNullEnergy = np.array(coherentNullEnergy)
    incoherentNullEnergy = np.array(incoherentNullEnergy)
    

    return (coherentNullEnergy, incoherentNullEnergy)

def bandpass_prior_to_notching(data, fs, f_min, f_max):
    filter_order = 10
    b, a = butter(filter_order, [f_min, f_max], btype='bandpass', output='ba', fs=fs)
    samples_to_crop = 1 * fs  # 2 seconds * fs samples/second

    bandpassed_data = np.zeros((data.shape[0], data.shape[1] - 2*samples_to_crop))
    for index in range(data.shape[0]):    
        # Apply the filter
        filtered_data = lfilter(b, a, data[index])
        # Crop the first 2 seconds
        cropped_data = filtered_data[samples_to_crop:-samples_to_crop]
        # Store the cropped data
        bandpassed_data[index] = cropped_data
        
    return bandpassed_data

def remove_line(data, fs, f_min, f_max, Q, factor):
    
    data = bandpass_prior_to_notching(data, fs, f_min, f_max)
    
    # FFT length chosen 4 times the sample frequency
    Nfft = 4 * fs  
    notched_data = np.zeros_like(data)  # Array to hold the smoothed time series
    originalPSD = []
    notch_centre_bin = []

    # Loop through the rows in the data (each row is a separate time series)
    for i in range(data.shape[0]):
        # Calculate the PSD of the signal
        frequencies, psd = welch(data[i], fs=fs, nperseg=Nfft)
        originalPSD.append(psd)
        smothBins = int(9 / (fs/Nfft))

        median_psd = np.zeros_like(psd)
        for bins in range(len(psd)):
            start = max(0, bins - smothBins)
            end = min(len(psd), bins + smothBins)
            median_psd[bins] = np.median(psd[start:end])
    
        # Identify lines
        line_centre_bin = []
        line_width_bins = []
        line_height = []
        prev_bin = False
        for j in range(len(psd)):
            if (psd[j] > factor*median_psd[j]) and (frequencies[j] >= f_min) and (frequencies[j] <= f_max):
                if prev_bin == False:
                    width_bins = 1
                    max_height = psd[j] / median_psd[j]
                    centre_bin = j
                else:
                    width_bins = width_bins+1
                    if psd[j] / median_psd[j] > max_height:
                        max_height = psd[j] / median_psd[j]
                        centre_bin = j
                prev_bin = True
            else:
                if prev_bin==True:
                    line_centre_bin.append(centre_bin)
                    line_width_bins.append(width_bins)
                    line_height.append(max_height)
                    prev_bin = False
        notch_centre_bin.append(line_centre_bin)

        filtered_data = np.copy(data[i])
        # Loop over frequencies and apply notch filter
        for centre_bin in notch_centre_bin[i]:
            # Create a notch filter
            centre_frequency = frequencies[centre_bin]
            b, a = iirnotch(centre_frequency, Q, fs)
            # Apply the filter to data
            filtered_data = lfilter(b, a, filtered_data)
        notched_data[i] = filtered_data
    w = int(notched_data.shape[1]/fs)
    return notched_data[:,int(((w-1)/2)*fs):int(((w+1)/2)*fs)]




def skymap_gen_function(fs, uwstrain, psd, gps, detectors
                        , alpha = None, beta=None, sigma=None
                        , nside = None
                        , window_parameter = None
                        , **kwargs):

    if alpha is None or beta is None or sigma is None:
        print(alpha, beta, sigma)
        raise ValueError('alpha, beta and sigma must be defined')
    
    #sigma = fs*sigma
    
    
    print(alpha, beta, sigma)
    if nside is None:
        nside = 64
    

    
    num_pixels = hp.nside2npix(nside)
    # print(num_pixels)

    start = time.time()

    # ---------------> Could we possibly save this somewhere and only load it?

    detector_initials = list(det + '1' for det in detectors)

    gps_time = gps[0]
    #gps_time = 1337069300

    #Setup detector array:
    detectors_objects = []
    for initial in detector_initials:
        detectors_objects.append(Detector(initial))


    #Create theta and phi arrays:
    theta, phi = hp.pix2ang(nside, range(num_pixels), nest = True)
    RA = phi
    dec = np.pi/2 - theta
    # print(type(theta))
    # print(type(phi))
    # print(type(RA))
    # print(type(dec))

    #Create Antenna and TimeDelay maps:
    null_coefficient = nullCoefficient(num_pixels,
                                       RA, dec,
                                       detectors_objects,
                                       gps_time)
    
    antenna_response_rms = antennaResponseRMS(
        num_pixels, RA, dec, detectors_objects, gps_time)
    
    time_delay_map = timeDelayMap(num_pixels, RA, dec, detectors_objects, gps_time)
    
    frequency_axis = np.fft.rfftfreq(fs, d=1/fs)

    # < ------------------------------------------
    
    notched_strain = remove_line(uwstrain, fs, f_min=20, f_max=480, Q=30.0, factor=10)

    
    start = time.time()
    sky_map = EnergySkyMaps(notched_strain,
                            time_delay_map, 
                            frequency_axis,
                            psd,
                            num_pixels, 
                            len(detectors), 
                            fs, 
                            null_coefficient,
                            antenna_response_rms,
                            window_parameter = window_parameter)
    
    sky_null = sky_map[0]
    sky_inc = sky_map[1]

    Lsky = (1-(sky_null/sky_inc)) * (np.max(sky_null) - sky_null)

    prob_map = ((antenna_response_rms)**alpha) * np.exp(
                 -((np.amax(Lsky) - Lsky)/sigma)**beta)


    prob_map = (prob_map)/np.sum(prob_map)
    print('prob_map', prob_map)

#     pixel_index = np.argsort(prob_map)[::-1]
#     # print('pixel_index', pixel_index)

#     sorted_pixels = prob_map[pixel_index]
#     # print('sorted_pixels', sorted_pixels)

#     unsorted_pixels = np.argsort(pixel_index)
#     # print('unsorted_pixels', unsorted_pixels)

#     # print('z', z)
#     c_est = np.cumsum(sorted_pixels)
#     # print('c_est', c_est)

#     # print(len(c_est))
#     # c_true = c_est - 0.05*np.sin(np.pi*c_est)
#     c_true = c_est**1  # --------> ???
#     #print('c_true', c_true)

#     p_true = np.diff(c_true, prepend=0)
#     # print('p_true', p_true)    

#     corrected_map = p_true[unsorted_pixels]
#     # print('corrected_map', corrected_map)
#     #prob_map_total.append(corrected_map)
#     print('skymap generation time: ',time.time() - start)

    prob_map_total = np.array(prob_map)
    Lsky_array = np.array(Lsky)
    
    # if isinstance(map_save, str):
    #     file_name = f'{map_save}.npy'

    #     np.save(file_name, prob_map_total)
    
    # if plot_map is True:
    #     plt.figure()
    #     hp.mollview(prob_map_total[0], coord = 'C', nest= None, title = "probability_map")
    #     #plt.savefig(f"prob_skymap_{time.time()}.png")
    
    return(prob_map_total, Lsky_array, antenna_response_rms)


def skymap_plot_function(strain,data=None):

    hp.mollview(data[0][0], coord = 'C', nest= True, title = "probability_map")
    hp.mollview(data[1][0], coord = 'C', nest = True, title = 'Lsky_map')


def skymap_plugin(alpha = 0.75, beta=0.128, sigma = 64*1024, nside =64, window_parameter = None):

    return PlugIn('sky_map', genFunction=skymap_gen_function , attributes= ['fs', 'uwstrain', 'psd', 'gps', 'detectors'],
                       plotFunction=skymap_plot_function, plotAttributes=['strain'], alpha = alpha, beta = beta, sigma = sigma, nside = nside, window_parameter = window_parameter)



def compute_prob_map_from_lsky(lsky_array, antenna_rms_array, alpha, beta, sigma):
    
    """
    lsky_array: it is a numpy array of likelihood maps computed from EnergySkyMaps function.
    
    antenna_rms_array: it is a numpy array of the root mean squared values of antenna response computed from antennaResponseRMS function.
    
    alpha, beta, sigma: hyper parameter, scaler value.
    
    """
    
    prob_maps = []
    max_prob = []
    for Lsky, antenna_response_rms in zip(lsky_array, antenna_rms_array):
        prob_map = (antenna_response_rms**alpha) * np.exp(-((np.amax(Lsky) - Lsky)/sigma)**beta)
        prob_map = prob_map - np.min(prob_map)
        prob_map = prob_map/np.sum(prob_map)
        prob_maps.append(prob_map)
        max_prob.append(np.max(prob_map))
    #np.save('prob_map_array', np.array(prob_maps))
    return np.array(prob_maps), np.array(max_prob)
# Outputs of this functions are also numpy arrays.




def compute_containment_and_search_area(prob_array, inj_pixel_array, thresholds=[0.9, 0.5]):
    
    """
    prob_array: it is a numpy array of probability maps computed from Lsky_array.
    
    inj_pixel_array: this is an numpy array of injection pixel for synthetic injections
    
    thresholds: it is a dictionary of two scaler threshold values which defines the containment regions.
    
    
    """
    
    # Comnpute area of each pixel in square degrees
    pixel_area = (4 * np.pi * (180/np.pi)**2) / len(prob_array[0])  
    
    # Initialise dictionaries to store results
    containment_results = {thresh: [] for thresh in thresholds}
    search_area = []
    containment_area = {thresh: [] for thresh in thresholds}
    search_prob = []
    
    for prob_map, inj_pixel_value in zip(prob_array, inj_pixel_array):        
        sorted_indices = np.argsort(prob_map)[::-1]
        cum_sum = np.cumsum(prob_map[sorted_indices]) / np.sum(prob_map)  # Normalize cum_sum
        search_area.append(np.sum(np.where(prob_map >= prob_map[inj_pixel_value], 1,0)) * pixel_area)
        search_prob.append(np.sum(np.where(prob_map >= prob_map[inj_pixel_value], prob_map,0)))
        
        for thresh in thresholds:
            # Identify containment region
            index_containment = np.argmax(cum_sum >= thresh)
            containment_region = sorted_indices[:index_containment + 1]
            
            # Check if the probability at the injection pixel is within the containment region
            is_in_containment = prob_map[inj_pixel_value] >= prob_map[containment_region[-1]]
            
            # Compute search area for the containment region
            containment_region_area = len(containment_region) * pixel_area

            
            containment_results[thresh].append(is_in_containment)
            containment_area[thresh].append(containment_region_area)
    
    return containment_results, search_area, containment_area, np.array(search_prob)

#outputs of the function are also in mumpy arrays.
