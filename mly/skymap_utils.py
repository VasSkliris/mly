# Construction of null-stream sky map using unwhitened strain data.

from tqdm import tqdm
import tensorflow as tf
import time
import healpy as hp
from healpy.newvisufunc import projview, newprojplot
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import time 
import ligo.skymap.plot
from ligo.skymap import io, plot,postprocess
from ligo.skymap.plot.marker import earth

import time 
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


def mask_window_tf(T, fs, start_time, end_time, ramp_duration,ramp_centre, duration_limit = 1/32):
    """
    Creates a windowed mask for a given time series.
    
    Parameters:
    - T: Duration of the final data series.
    - fs: Sampling frequency.
    - start_time: Starting time of the window.
    - end_time: Ending time of the window.
    - ramp_duration: Duration of the ramp (smooth transition) in the window.
    - ramp_centre: Where will the center of the ramp be in respect of start and end time [0,1]
                    0 corresponds for the center to be -0.5*ramp_duration away from the signal 
                    1 corresponds for the center to be +0.5*ramp_duration in the signal (dumping the edge)
    Returns:
    - A windowed mask with values between 0 and 1.
    """
    start_time = tf.cast(start_time, tf.float32)
    end_time = tf.cast(end_time, tf.float32)
    ramp_duration = tf.cast(ramp_duration, tf.float32)
    ramp_centre = tf.cast(ramp_centre, tf.float32)
    duration_limit = tf.cast(duration_limit, tf.float32)
    fs = tf.cast(fs, tf.float32)
    tnp = tf.experimental.numpy
    PI = tnp.pi 
    PI = tf.cast( PI, tf.float32)
    
    # Duration limit is used as limiter on the duration of signals
    # also it is used for the edges so that they are smooth and they
    # do not fold on the other side.
    if end_time - start_time < duration_limit:

        diff = duration_limit - (end_time - start_time)
        
        start_time = start_time -diff/2
        end_time = end_time + diff/2

        print(f"Duration smaller than duration limit, and it was adjusted by {diff} s")

    if start_time < ramp_duration:
        # The first 32 ms will be folded to the end so we move the start time 
        # enough to avoid that.
        start_time = ramp_duration + duration_limit 
        print("Start time close too close to the biggining")

        # Adjusting duration in case the above adjustment 
        # made duration negative or too small
        if end_time-start_time < duration_limit: # Max difference between detectors
            end_time = start_time + duration_limit
            print("End time too close to start time, changed to start_time + 0.032")

    if T-end_time < ramp_duration:
        # The last 32 ms will be folded to the end so we move the start time 
        # enough to avoid that.
        end_time = T - ramp_duration - duration_limit
        print("End time close too close to the biggining")
        # Adjusting duration in case the above adjustment 
        # made duration negative or too small
        if end_time-start_time < duration_limit: # Max difference between detectors
            start_time = end_time - duration_limit
            print("Start time too close to end time, changed to end_time - 0.032")


    signal_duration = end_time - start_time
    
    # Check that ramp_duration is a power of two:
    # if not (ramp_duration != 0 
    #     and ramp_duration<T/2 
    #     and tf.subtract(tf.experimental.numpy.log2(ramp_duration) , tf.math.round(tf.experimental.numpy.log2(ramp_duration))).numpy() == 0.0):

    #     raise ValueError("Ramp duration must be a power of two"
    #                      ", and one fourth of the duration")

    # Adjusting the ramp_duration when signals are too small
    # Note: Possibly adjust this with the duration of the signal
    #       to include cases where signal is small and at the edge.
    while signal_duration <= 2 * ramp_duration * ramp_centre:

        ramp_duration = ramp_duration/2

        print(f"Ramp duration changed to 1/{int(fs/(ramp_duration*fs))}")

        if ramp_duration == 1/128:
            print("Ramp duration reached minimum value of 1/128")
            break

    # Creating the central platoe
    centre_lentgh = int(( signal_duration - 2 * ramp_duration * ramp_centre ) * fs )
    centre = tf.ones( centre_lentgh)

    # Creating the ramps
    t = tf.range( 0 , ramp_duration, 1/fs )
    left_ramp  = tf.math.sin( PI * t * (1/ramp_duration) - PI/2 )/2 + 0.5
    right_ramp = tf.math.sin( PI * t * (1/ramp_duration) + PI/2 )/2 + 0.5

    # Creating the pads
    #print(start_time, ramp_centre,ramp_duration,fs)
    

    left_pad =  tf.zeros( tf.cast(tf.math.ceil((start_time - (1 - ramp_centre) * ramp_duration) * fs),tf.int32) )
    right_pad = tf.zeros( tf.cast(tf.math.ceil((T - end_time - (1 - ramp_centre) * ramp_duration) * fs ),tf.int32))

    # print(len(left_pad),len(left_ramp),len(centre),len(right_ramp),len(right_pad))
    # print(len(left_pad)+len(left_ramp)+len(centre)+len(right_ramp)+len(right_pad))

    # Adjusting the length when sometimes a pixel or two missing due to the use of int and ceil.
    if len(left_pad)+len(left_ramp)+len(centre)+len(right_ramp)+len(right_pad) < tf.cast(T*fs,tf.float32):
        diff = tf.cast( T*fs -  len(left_pad)+len(left_ramp)+len(centre)+len(right_ramp)+len(right_pad) , tf.int32 ) 

        centre = tf.concat([centre, tf.ones(diff)],0)
    
    elif len(left_pad)+len(left_ramp)+len(centre)+len(right_ramp)+len(right_pad) > tf.cast(T*fs,tf.float32):

        diff = tf.cast( len(left_pad)+len(left_ramp)+len(centre)+len(right_ramp)+len(right_pad) - T*fs , tf.int32 )

        centre = centre[:-diff]
    
    else:
        diff = 0

    # Putting the mask together.
    mask = tf.concat([ left_pad, left_ramp , centre, right_ramp, right_pad ],0)

    return(mask)






# def mask_window_tf_old(data_length, fs, start_time, end_time, ramp_duration):
#     """
#     Creates a windowed mask for a given time series.
    
#     Parameters:
#     - data_length: Length of the data series.
#     - fs: Sampling frequency.
#     - start_time: Starting time of the window.
#     - end_time: Ending time of the window.
#     - ramp_duration: Duration of the ramp (smooth transition) in the window.
    
#     Returns:
#     - A windowed mask with values between 0 and 1.
#     """
#     tnp = tf.experimental.numpy
#     pi = tnp.pi 
#     PI_HALF = pi / 2

#     # Convert times to indices and ensure they are float32
#     window_start_idx = tf.cast((start_time - ramp_duration) * fs, tf.float32)
#     window_end_idx = tf.cast((end_time + ramp_duration) * fs, tf.float32)
#     start_idx = tf.cast(start_time * fs, tf.float32)
#     end_idx = tf.cast(end_time * fs, tf.float32)
    
#     indices = tf.range(data_length, dtype=tf.float32)
    
#     # Create the rolling mask for the main windowed region
#     main_mask = tf.where((indices >= start_idx) & (indices < end_idx), 1., indices * 0.)
    
#     # Create the ramp-up transition
#     ramp_up = 0.5 * (1 + tf.math.sin(-PI_HALF + (indices - window_start_idx) / (start_idx - window_start_idx) * pi))
#     mask_on = tf.where((indices >= window_start_idx) & (indices < start_idx), ramp_up, 0.)

#     # Create the ramp-down transition
#     ramp_down = 0.5 * (1 + tf.math.sin(PI_HALF + (indices - end_idx) / (window_end_idx - end_idx) * pi))
#     mask_off = tf.where((indices >= end_idx) & (indices < window_end_idx), ramp_down, 0.)

#     # Combine the masks
#     windowed_data = main_mask + mask_on + mask_off

#     return windowed_data


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
    window_parameter = None,
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

        start_time, end_time, ramp_duration, ramp_centre, duration_limit = window_parameter
        
        # data_shape = tf.shape(strain)
        # num_det = data_shape[1]
        # data_len = data_shape[2]
        
        # Apply mask_window_tf function on strain data
        windowed_data = mask_window_tf(1.0, fs, start_time, end_time, ramp_duration, ramp_centre, duration_limit = duration_limit)
        # print('windowed_data', windowed_data.shape)
        # print ('strain shape', strain.shape)
        # print('start_time_util', start_time)
        # print('end_time_until', start_time)
    
        
        # Reshape windowed_data to match the shape of strain
        windowed_data_reshaped = tf.reshape(windowed_data, [1, 1, -1])
        #window_shape_broadcast = tf.broadcast_to(windowed_data_reshaped, data_shape)
        
        #print('window_shape_broadcast', window_shape_broadcast.shape)
        
        strain = tf.math.multiply(tf.cast(windowed_data_reshaped, dtype = tf.float64), strain)
        #print('windowed_strain', strain)
    else:
        print('window not activated', window_parameter)

    #Calculate shifted positions using phase correction:
    shifted_strain = tf.math.divide(timeDelayShiftTensorflow(
        strain, frequency_axis, time_delay), fs) #WHY divide by fs?

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
            dominant_polarisation_p[pixel_index] = [ np.cos(2*dpa)*fp[0] + sin(2*dpa)*fc[0],  np.cos(2*dpa)*fp[1] + sin(2*dpa)*fc[1]]
            dominant_polarisation_c[pixel_index] = [-np.sin(2*dpa)*fp[0] + cos(2*dpa)*fc[0], -np.sin(2*dpa)*fp[1] + cos(2*dpa)*fc[1]]
            
            # if pixel_index ==0:
            #     print('dp', dominant_polarisation_p[0][:3])
            #     print('dc', dominant_polarisation_c[0][:3])
            
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




def bandpass(data, fs, f_min, f_max, filter_order=10):

    # ---- Construct bandpass filter.
    b, a = butter(filter_order, [f_min, f_max], btype='bandpass', output='ba', fs=fs)

    # ---- Assign storage for bandpassed data. Remove samples_to_crop samples from 
    #      each end to avoid filter transients. 
    samples_to_crop = 1 * fs  # 1 second at each end
    bandpassed_data = np.zeros((data.shape[0], data.shape[1] - 2*samples_to_crop))
    for index in range(data.shape[0]):    
        # ---- Apply the filter.
        filtered_data = lfilter(b, a, data[index])
        # ---- Crop the ends and store.
        bandpassed_data[index] = filtered_data[samples_to_crop:-samples_to_crop]
        # # Crop the ends
        # cropped_data = filtered_data[samples_to_crop:-samples_to_crop]
        # # Store the cropped data
        # bandpassed_data[index] = cropped_data
        
    return bandpassed_data



def remove_lines(data, fs, f_min, f_max, Q=30.0, factor=10.0, smoothing=9.0):

    # data  FORMAT? Timeseries data.
    # fs  Integer. Sample rate [Hz] of timeseries data.
    # f_min  Float. Minimum frequency [Hz] for bandpassing.
    # f_max  Float. Maximum frequency [Hz] for bandpassing.
    # Q  Float. Notch filter quality factor. Default 30.0.
    # factor  Float. Minimum line height above median PSD. Default 10.0.
    # smoothing  Float. Width [Hz] for smoothing. Default 9.0.

    # ---- Parameters for line removal.
    smoothingWindowHz = smoothing
    minLineHeightFactor = factor
    
    # ---- FFT length. 
    Nfft = 4 * fs    # FFT resolution 0.25 Hz

    # ---- Bandpass the data.
    data = bandpass(data, fs, f_min, f_max)
    
    confirmed_clean = False

    while confirmed_clean == False:

        notch_centre_bin = []
        notch_width_bins = []
        notch_height     = []  # really depth ...

        # ---- Check all timeseries for lines.

        # ---- Loop through the rows in the data (each row is a separate time series).
        for ifo in range(data.shape[0]):

            # ---- Compute the PSD of the data.
            frequency, psd = welch(data[ifo], fs=fs, nperseg=Nfft)
        
            # ---- Construct and plot the "smoothed" PSD.
            smoothingWindowBins = int(smoothingWindowHz / (fs/Nfft))
            median_psd = np.zeros_like(psd)
            for bins in range(len(psd)):
                start = max(0, bins - smoothingWindowBins)
                end = min(len(psd), bins + smoothingWindowBins)
                median_psd[bins] = np.median(psd[start:end])
            
            # ---- Identify lines.
            line_centre_bin = []
            line_width_bins = []
            line_height     = []
            prev_bin = False
            for j in range(len(psd)):
                if (psd[j] > minLineHeightFactor*median_psd[j]) and (frequency[j] >= f_min) and (frequency[j] <= f_max):
                    #print(frequency[j])
                    if prev_bin==False:
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
            notch_width_bins.append(line_width_bins)
            notch_height.append(line_height)

        # ---- Remove the lines.
        confirmed_clean = True
        for ifo in range(data.shape[0]):
            frequencies_to_remove = frequency[notch_centre_bin[ifo]]
            if notch_centre_bin[ifo]:
                print('notches found for detector',ifo,':')
                print(notch_centre_bin[ifo])
                confirmed_clean = False # resets to loop again if any lines found
            # ---- Copy the original data.
            filtered_data = np.copy(data[ifo])
            # ---- Loop over frequencies and apply notch filter.
            for f0 in frequencies_to_remove:
                # ---- Create and apply a notch filter.
                b, a = iirnotch(f0, Q, fs)
                filtered_data = lfilter(b, a, filtered_data)
            # ---- Replace background timeseries in the pod.
            data[ifo] = filtered_data            

    # ---- Return central 1 second of notched data.
    # w = int(data.shape[1]/fs)
    # # print('w =',w)
    # # print('original shape of notched data:',notched_data.shape)
    # data = data[:,int(((w-1)/2)*fs):int(((w+1)/2)*fs)]
    # # print('shape of notched data:',notched_data.shape)
    return data

def remove_line(data, fs, f_min, f_max, Q=30.0, factor=10.0):
    
    # print('shape of raw data:',data.shape)

    data = bandpass(data, fs, f_min, f_max)
    
    # print('shape of bandpassed data:',data.shape)

    # FFT length chosen 4 times the sample frequency (FFT resolution 0.25 Hz)
    Nfft = 4 * fs  
    notched_data = np.zeros_like(data)  # Array to hold the smoothed time series
    # originalPSD = []
    notch_centre_bin = []

    # Loop through the rows in the data (each row is a separate time series)
    for i in range(data.shape[0]):

        # Calculate the PSD of the data.
        frequencies, psd = welch(data[i], fs=fs, nperseg=Nfft)
        # originalPSD.append(psd)

        # Calculate the smoothed PSD.
        smoothBins = int(9 / (fs/Nfft))
        median_psd = np.zeros_like(psd)
        for bins in range(len(psd)):
            start = max(0, bins - smoothBins)
            end = min(len(psd), bins + smoothBins)
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
            b, a = iirnotch(frequencies[centre_bin], Q, fs)
            # Apply the filter to data
            filtered_data = lfilter(b, a, filtered_data)
        notched_data[i] = filtered_data

    # # ---- Return central 1 second of notched data.
    # w = int(notched_data.shape[1]/fs)
    # # print('w =',w)
    # # print('original shape of notched data:',notched_data.shape)
    # # print('shape of notched data:',notched_data.shape)
    return notched_data




def skymap_gen_function(strain,fs, uwstrain, psd, gps, detectors,PE
                        , alpha = None, beta=None, sigma=None
                        , nside = None
                        , window_parameter = None
                        , **kwargs):

    if alpha is None or beta is None or sigma is None:
        #print(alpha, beta, sigma)
        raise ValueError('alpha, beta and sigma must be defined')
    
    #sigma = fs*sigma
    
    
    #print(alpha, beta, sigma)
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

    #Create Antenna and TimeDelay maps:
    null_coefficient = nullCoefficient(num_pixels,
                                       RA, dec,
                                       detectors_objects,
                                       gps_time)

    #print('nc shape:',null_coefficient.shape)
    
    antenna_response_rms = antennaResponseRMS(num_pixels
                                              , RA, dec
                                              , detectors_objects
                                              , gps_time)
    
    #print('ar shape:',antenna_response_rms.shape)

    time_delay_map = timeDelayMap(num_pixels
                                  , RA, dec
                                  , detectors_objects
                                  , gps_time)
    
    #print('td shape:',time_delay_map.shape)

    frequency_axis = np.fft.rfftfreq(fs, d=1/fs)

    # < ------------------------------------------
    
    notched_strain = remove_line(uwstrain, fs, f_min=20, f_max=480, Q=30.0, factor=10)

    w = int(notched_strain.shape[1]/fs)

    notched_strain= notched_strain[:,int(((w-1)/2)*fs):int(((w+1)/2)*fs)]

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
                 -((np.amax(Lsky) - Lsky)/(sigma/np.sqrt(PE['bandwidth'])))**beta)


    prob_map = (prob_map)/np.sum(prob_map)

    prob_map_total = np.array(prob_map)
    Lsky_array = np.array(Lsky)

    containment_region_50 = containment_region(prob_map,threshold=0.5)
    containment_region_90 = containment_region(prob_map,threshold=0.9)

    return [ prob_map_total, Lsky_array, antenna_response_rms , containment_region_50, containment_region_90]

def skymap_plot_function(strain,data=None):
        
    """
    Function to format and display a LIGO skymap plot.

    Parameters
    ----------
    probmap: A HEALPix probability map to be plotted.
    projection: The projection type for the plot.
    nested: Boolean indicating if the HEALPix data is in 'nested' format.
    cmap: The colormap for the plot.
    xlabel: Label for the x-axis.
    ylabel: Label for the y-axis.
    title: The title for the plot.
    save: Boolean to determine whether to save the plot or not.
    inj_locations: Array of injection locations in (RA, Dec) format.

    """

    probmap = data[0]
    containment_region_50 = data[3]
    containment_region_90 = data[4]
    # Plot settings
    projection='astro hours mollweide'
    nested=True
    cmap='cylon'

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection=projection)

    # Plot the sky map
    img = ax.imshow_hpx(probmap, nested=nested, cmap=cmap)

    # Add grid lines
    ax.grid(True, which='major', color='k', linestyle='-', linewidth=0)

    ax.text(0.8, 1, f"50% area: {containment_region_50:.3f} deg²", weight='bold', transform=ax.transAxes)  
    ax.text(0.8, 0.95, f"90% area: {containment_region_90:.3f} deg²", weight='bold', transform=ax.transAxes) 

    c = 100 * postprocess.find_greedy_credible_levels(probmap)
    cs = ax.contour_hpx(c, nested=True, colors='k', linewidths=0.5, levels=[50,90])
    plt.clabel(cs, fmt='%g%%', fontsize=6, inline=True)


def skymap_plot_function_with_inj(strain,RA,declination,data=None):
        
    """
    Function to format and display a LIGO skymap plot.

    Parameters
    ----------
    probmap: A HEALPix probability map to be plotted.
    projection: The projection type for the plot.
    nested: Boolean indicating if the HEALPix data is in 'nested' format.
    cmap: The colormap for the plot.
    xlabel: Label for the x-axis.
    ylabel: Label for the y-axis.
    title: The title for the plot.
    save: Boolean to determine whether to save the plot or not.
    inj_locations: Array of injection locations in (RA, Dec) format.

    """

    probmap = data[0]
    containment_region_50 = data[3]
    containment_region_90 = data[4]
    # Plot settings
    projection='astro hours mollweide'
    nested=True
    cmap='viridis'
    xlabel='Right Ascension'
    ylabel='Declination'

    # Create a new figure and subplot with a Mollweide projection
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection=projection)

    # Plot the sky map
    img = ax.imshow_hpx(probmap, nested=nested, cmap=cmap)

    # Add grid lines
    ax.grid(True, which='major', color='white', linestyle='-', linewidth=0.2)

    ax.text(0, 0, xlabel, ha='center', va='center', transform=ax.transAxes)  
    ax.set_ylabel(ylabel)
    ax.text(0.8, 1, f"50% area: {containment_region_50:.3f} deg²", weight='bold', transform=ax.transAxes)  
    ax.text(0.8, 0.95, f"90% area: {containment_region_90:.3f} deg²", weight='bold', transform=ax.transAxes) 
    # Add color bar
    cbar = plt.colorbar(img, ax=ax, orientation='horizontal', fraction=0.05, pad=0.1)
    cbar.set_label('Probability')

    c = 100 * postprocess.find_greedy_credible_levels(probmap)
    cs = ax.contour_hpx(c, nested=True, colors='k', linewidths=0.5, levels=[50,90])
    plt.clabel(cs, fmt='%g%%', fontsize=6, inline=True)

    # Plot injection location if injection is present
    ax.plot(np.degrees(RA), np.degrees(declination),
        transform=ax.get_transform('world'),
        marker=ligo.skymap.plot.reticle(),
        markersize=30,
        markeredgewidth=3)

    
def skymap_plugin(alpha = None, beta=None, sigma = None, nside =None, window_parameter = None, injection = False):

    if injection:

        return PlugIn('sky_map', genFunction=skymap_gen_function , attributes= ['strain','fs', 'uwstrain', 'psd', 'gps', 'detectors','PE'],
                        plotFunction=skymap_plot_function_with_inj, plotAttributes=['strain','RA','declination'], alpha = alpha, beta = beta, sigma = sigma, nside = nside, window_parameter = window_parameter)
    else:
        
        return PlugIn('sky_map', genFunction=skymap_gen_function , attributes= ['strain','fs', 'uwstrain', 'psd', 'gps', 'detectors','PE'],
                        plotFunction=skymap_plot_function, plotAttributes=['strain'], alpha = alpha, beta = beta, sigma = sigma, nside = nside, window_parameter = window_parameter)


def compute_prob_map_from_lsky(lsky_array, antenna_rms_array, alpha, beta, sigma):
    
    """
    lsky_array: it is a numpy array of likelihood maps computed from EnergySkyMaps function.
    
    antenna_rms_array: it is a numpy array of the root mean squared values of antenna response computed from antennaResponseRMS function.
    
    alpha, beta, sigma: hyper parameter, scaler value.
    
    
    RETURN:
    Outputs of this functions are also numpy arrays.
    
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


def search_area(prob_map,inj_pixel_index):
    pixel_area = (4 * np.pi * (180/np.pi)**2) / len(prob_map)

    s_area = np.sum(np.where(prob_map >= prob_map[inj_pixel_index], 1,0)) * pixel_area

    return s_area

def search_probability(prob_map,inj_pixel_index):

    s_prob = (np.sum(np.where(prob_map >= prob_map[inj_pixel_index], prob_map,0)))

    return s_prob


def containment_region(prob_map, threshold = 0.5):
    pixel_area = (4 * np.pi * (180/np.pi)**2) / len(prob_map)

    sorted_indices = np.argsort(prob_map)[::-1]
    cum_sum = np.cumsum(prob_map[sorted_indices]) / np.sum(prob_map) 

    index_containment = np.argmax(cum_sum >= threshold)
    containment_region = sorted_indices[:index_containment + 1]
    containment_region_area = len(containment_region) * pixel_area

    return containment_region_area



