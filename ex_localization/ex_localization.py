# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 14:19:30 2019

@author: WSJF7149
"""
import grid
import dsp
import numpy as np
from scipy.io.wavfile import read
from keras.models import load_model
import os

#%%
def getNormalizedIntensity(x_f):
    '''
    Compute the input feature needed by the localization neural network.
    
    Parameters
    ----------
    x_f: nd-array
        STFT of the HOA signal.
        Shape: (nBatch, nChannel, lSentence, nBand)
    
    Returns
    -------
    inputFeat_f: nd-array
        Shape: (nBatch, lSentence, nBand, nFeat)
    '''
    (lBatch, nChannel, lSentence, nBand) = x_f.shape
    nFeat = 6
    inputFeat_f = np.empty((lBatch, lSentence, nBand, nFeat), dtype=np.float32)
    
    for nBatch, sig_f in enumerate(x_f):       # Process all examples in the batch
        # Compute the intensity vector in each TF bin
        intensityVect = np.empty((lSentence, nBand, 3), dtype=complex)
        intensityVect[:,:,0] = sig_f[0].conj()*sig_f[1]
        intensityVect[:,:,1] = sig_f[0].conj()*sig_f[2]
        intensityVect[:,:,2] = sig_f[0].conj()*sig_f[3]
        
        # Normalize it in each TF bin
        coeffNorm = (abs(sig_f[0])**2+np.sum(abs(sig_f[1:])**2/3, axis=0))[:,:,np.newaxis]    
        inputFeat_f[nBatch, :, :, :nFeat//2] = np.real(intensityVect)/coeffNorm
        inputFeat_f[nBatch, :, :, nFeat//2:] = np.imag(intensityVect)/coeffNorm

    return inputFeat_f

#%%
def main():    
    nSrc = 2        # examples provided for 1 or 2
    
    # Load the FOA signal and the network
    if nSrc == 1:
        filename = r'src0__el_00_r_1.30_az_-63__snr02_rt387.wav'
        modelname = r'foadoa_1src_0516.h5'
    else:       # Model trained for 2 sources, but you can try with more
        filename = r'src0__el_00_r_1.30_az_-63_src1__el_68_r_1.30_az_179__snr20_sir02_rt387.wav'
        modelname = r'foadoa_2src_0611.h5'
    fs, x = read(filename)
    x = x.T
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    model = load_model(modelname)
    
    # Compute the STFT
    nChannel, nSmp = x.shape
    lFrame = 1024                   # size of the STFT window, in samples
    nBand = lFrame//2+1
    nOverlap = lFrame//2
    hop = lFrame - nOverlap
    lSentence = nSmp//hop
    x_f = np.empty((nChannel, lSentence, nBand), dtype=complex)
    for iChannel in range(nChannel):
        x_f[iChannel] = dsp.stft(x[iChannel], lWindow=lFrame)
        
    # The neural network can only process buffers of 25 frames
    lBuffer = 25
    nBatch = 1
    x_f = x_f[np.newaxis,:,:lBuffer,:]      # First axis corresponding to the batches of 25 frames to process; here there's only 1
        
    # Get the input feature for the neural network
    inputFeat_f = getNormalizedIntensity(x_f)
    
    # Make the prediction
    predictProbaByFrame = model.predict(inputFeat_f)        

    # Average the predictions over a sequence of 25 frames
    predictProbaByBatch = np.mean(predictProbaByFrame, axis=1)
    
    # Load the discretized sphere corresponding to the network's output classes
    gridStep = 10                           # Approximate angle between two DoAs on the grid, in degrees
    neighbour_tol = 2*gridStep              # angular diameter of a neighborhood, for spatial smoothing and peak selection
    el_grid, az_grid = grid.makeDoaGrid(gridStep)
    
    # Get the main peaks and the corresponding elevations and azimuths
    el_pred = np.empty((nBatch, nSrc))
    az_pred = np.empty((nBatch, nSrc))
    for iBatch in range(nBatch):    
        # Find all the main peaks of the prediction (spatial smoothing is performed)
        peaks, iPeaks = grid.peaksOnGrid(predictProbaByBatch[iBatch], el_grid, az_grid, neighbour_tol)
        
        # Select the right number of sources
        for iSrc in range(nSrc):
            iMax = np.argmax(peaks)
            peaks[iMax] = 0         # enables to find other sources
            predIdx = iPeaks[iMax]

            el_pred[iBatch, iSrc] = el_grid[predIdx]
            az_pred[iBatch, iSrc] = az_grid[predIdx]
    print(el_pred)
    print(az_pred)

if __name__ == '__main__':
  main()

