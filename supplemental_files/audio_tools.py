# Audio processing tools
#
# Liberty Hamilton 2020
# 
# Some code modified from original MATLAB rastamat package.
# 

import numpy as np
from scipy.signal import hanning, spectrogram, resample, hilbert, butter, filtfilt
from scipy.io import wavfile
# import spectools
# from .fbtools import fft2melmx
from matplotlib import pyplot as plt
import parselmouth as pm
# from soundsig import sound



def get_meanF0s_v2(fileName, steps=1/128.0, f0min=50, f0max=300):
	"""
	Uses parselmouth Sound and Pitch object to generate frequency spectrum of
	wavfile, 'fileName'.  Mean F0 frequencies are calculated for each phoneme
	in 'phoneme_times' by averaging non-zero frequencies within a given
	phoneme's time segment.  A range of 10 log spaced center frequencies is
	calculated for pitch classes. A pitch belongs to the class of the closest
	center frequency bin that falls below one standard deviation of the center
	frequency range.
	"""
	#fileName = wav_dirs + wav_name
	sound_obj =  pm.Sound(fileName)
	pitch = sound_obj.to_pitch(steps, f0min, f0max) #create a praat pitch object
	pitch_values = pitch.selected_array['frequency']
	return pitch_values


def fft2melmx(nfft, sr=8000, nfilts=0, bwidth=1.0, minfreq=0, maxfreq=4000, htkmel=False, constamp=0):
  '''
  #      Generate a matrix of weights to combine FFT bins into Mel
  #      bins.  nfft defines the source FFT size at sampling rate sr.
  #      Optional nfilts specifies the number of output bands required 
  #      (else one per "mel/width"), and width is the constant width of each 
  #      band relative to standard Mel (default 1).
  #      While wts has nfft columns, the second half are all zero. 
  #      Hence, Mel spectrum is fft2melmx(nfft,sr)*abs(fft(xincols,nfft));
  #      minfreq is the frequency (in Hz) of the lowest band edge;
  #      default is 0, but 133.33 is a common standard (to skip LF).
  #      maxfreq is frequency in Hz of upper edge; default sr/2.
  #      You can exactly duplicate the mel matrix in Slaney's mfcc.m
  #      as fft2melmx(512, 8000, 40, 1, 133.33, 6855.5, 0);
  #      htkmel=1 means use HTK's version of the mel curve, not Slaney's.
  #      constamp=1 means make integration windows peak at 1, not sum to 1.
  #      frqs returns bin center frqs.
  # 2004-09-05  dpwe@ee.columbia.edu  based on fft2barkmx
  '''
  
  if nfilts == 0:
    nfilts = np.ceil(hz2mel(maxfreq, htkmel)/2);
  
  wts = np.zeros((nfilts, nfft))
  
  # Center freqs of each FFT bin
  fftfrqs = np.arange(0,nfft/2.)/nfft*sr
  
  # 'Center freqs' of mel bands - uniformly spaced between limits
  minmel = hz2mel(minfreq, htkmel)
  maxmel = hz2mel(maxfreq, htkmel)
  binfrqs = mel2hz(minmel+np.arange(0.,nfilts+2)/(nfilts+2.)*(maxmel-minmel), htkmel);
  
  binbin = np.round(binfrqs/sr*(nfft-1.))
  
  for i in np.arange(nfilts):
    fs = binfrqs[i+[0, 1, 2]]

    #print fs
    # scale by width
    fs = fs[1]+bwidth*(fs - fs[1]);
    # lower and upper slopes for all bins
    loslope = (fftfrqs - fs[0])/(fs[1] - fs[0])
    hislope = (fs[2] - fftfrqs)/(fs[2] - fs[1])
    w = np.min((loslope, hislope), axis=0)
    w[w<0] = 0
    # .. then intersect them with each other and zero
    wts[i, 0:np.int(nfft/2)] = w
    
  if constamp == 0:
    # Slaney-style mel is scaled to be approx constant E per channel
    wts = np.dot(np.diag(2./(binfrqs[2+np.arange(nfilts)]-binfrqs[np.arange(nfilts)])),wts)
    #wts = np.atleast_2d(2/(binfrqs[2+np.arange(nfilts)]-binfrqs[np.arange(nfilts)]))*wts
    #wts = np.dot(2/(binfrqs[2+np.arange(nfilts)]-binfrqs[np.arange(nfilts)]),wts)
  
  # Make sure 2nd half of FFT is zero
  wts[:,np.int(nfft/2+2):np.int(nfft)] = 0
  # seems like a good idea to avoid aliasing
  return wts, binfrqs

def hz2mel(f,htk=False):
  '''
  #  z = hz2mel(f,htk)
  #  Convert frequencies f (in Hz) to mel 'scale'.
  #  Optional htk = 1 uses the mel axis defined in the HTKBook
  #  otherwise use Slaney's formula
  # 2005-04-19 dpwe@ee.columbia.edu
  '''

  if htk:
    z = 2595 * log10(1+f/700)
  else:
    # Mel fn to match Slaney's Auditory Toolbox mfcc.m

    f_0 = 0.; # 133.33333;
    f_sp = 200./3; # 66.66667;
    brkfrq = 1000.;
    brkpt  = (brkfrq - f_0)/f_sp;  # starting mel value for log region
    logstep = np.exp(np.log(6.4)/27.); # the magic 1.0711703 which is the ratio needed to get from 1000 Hz to 6400 Hz in 27 steps, and is *almost* the ratio between 1000 Hz and the preceding linear filter center at 933.33333 Hz (actually 1000/933.33333 = 1.07142857142857 and  exp(log(6.4)/27) = 1.07117028749447)

    linpts = [f < brkfrq];

    z = 0*f;

    # fill in parts separately
    if len(linpts) == 1:
      if linpts[0]:
        z = f - f_0/f_sp
      else:
        z = brkpt+(np.log(f/brkfrq))/np.log(logstep)
    else:
      z[linpts==True] = (f[linpts==True] - f_0)/f_sp
      z[linpts==False] = brkpt+(np.log(f[linpts==False]/brkfrq))/np.log(logstep)

  return z

def mel2hz(z, htk=False):
#   f = mel2hz(z, htk)
#   Convert 'mel scale' frequencies into Hz
#   Optional htk = 1 means use the HTK formula
#   else use the formula from Slaney's mfcc.m
# 2005-04-19 dpwe@ee.columbia.edu


  if htk:
    f = 700.*(10**(z/2595.)-1);
  else:
    f_0 = 0.; # 133.33333;
    f_sp = 200./3.; # 66.66667;
    brkfrq = 1000.;
    brkpt  = (brkfrq - f_0)/f_sp;  # starting mel value for log region
    logstep = np.exp(np.log(6.4)/27.); # the magic 1.0711703 which is the ratio needed to get from 1000 Hz to 6400 Hz in 27 steps, and is *almost* the ratio between 1000 Hz and the preceding linear filter center at 933.33333 Hz (actually 1000/933.33333 = 1.07142857142857 and  exp(log(6.4)/27) = 1.07117028749447)

    linpts = [z < brkpt]
    nonlinpts = [z >= brkpt]

    f = 0*z;

    # fill in parts separately
    f[linpts] = f_0 + f_sp*z[linpts];
    f[nonlinpts] = brkfrq*np.exp(np.log(logstep)*(z[nonlinpts]-brkpt));
  return f

def get_envelope(audio, audio_fs, new_fs, cof=25, bef_aft=[0, 0], pad_next_pow2=False):
    ''' Get the envelope of a sound file
    Inputs:
        w [float] : audio signal vector
        fs [int] : sampling rate of audio signal
        new_fs [int] : desired sampling rate of the envelope (same as your EEG, for example)
    Outputs:
        envelope [array-like] : returns the envelope of the sound as an array
    '''
    
    if pad_next_pow2:
        print("Padding the signal to the nearest power of two...this should speed things up")
        orig_len = len(audio)
        sound_pad = np.hstack((audio, np.zeros((2**np.int(np.ceil(np.log2(len(audio))))-len(audio),))))
        audio = sound_pad

    print("calculating hilbert transform")
    env_hilb = np.abs(hilbert(audio))

    nyq = audio_fs/2. #Nyquist frequency
    b, a = butter(3, cof/nyq, 'low'); #this designs a 3-pole low-pass filter
    
    print("Low-pass filtering hilbert transform to get audio envelope")
    envelope_long = np.atleast_2d(filtfilt(b, a, env_hilb, axis=0)) #filtfilt makes it non-causal (fwd/backward)

    envelope = resample(envelope_long.T, np.int(np.floor(envelope_long.shape[1]/(audio_fs/new_fs))))
    if pad_next_pow2:
        print("Removing padding")
        final_len = np.int((orig_len/audio_fs)*new_fs)
        envelope = envelope[:final_len,:]
        print(envelope.shape)

    if bef_aft[0] < 0:
        print("Adding %.2f seconds of silence before"%bef_aft[0])
        envelope = np.vstack(( np.zeros((np.int(np.abs(bef_aft[0])*new_fs), 1)), envelope ))
    if bef_aft[1] > 0:
        print("Adding %.2f seconds of silence after"%bef_aft[1])
        envelope = np.vstack(( envelope, np.zeros((np.int(bef_aft[1]*new_fs), 1)) ))

    return envelope

def get_cse_onset(audio, audio_fs, wins = [0.04], nfilts=80, pos_deriv=True, spec_noise_thresh=1.04):
    """
    Get the onset based on cochlear scaled entropy

    Inputs:
        audio [np.array] : your audio
        audio_fs [float] : audio sampling rate
        wins [list] : list of windows to use in the boxcar convolution
        pos_deriv [bool] : whether to detect onsets only (True) or onsets and offsets (False)

    Outputs:
        cse [np.array] : rectified cochlear scaled entropy over window [wins]
        auddiff [np.array] : instantaneous derivative of spectrogram
    """
    new_fs = 100 # Sampling frequency of spectrogram
    specgram = get_mel_spectrogram(audio, audio_fs, nfilts=nfilts)
    specgram[specgram<spec_noise_thresh] = 0
    nfilts, ntimes = specgram.shape

    if pos_deriv is False:
        auddiff= np.sum(np.diff(np.hstack((np.atleast_2d(specgram[:,0]).T, specgram)))**2, axis=0)
    else:
        all_diff = np.diff(np.hstack((np.atleast_2d(specgram[:,0]).T, specgram)))
        all_diff[all_diff<0] = 0
        auddiff = np.sum(all_diff**2, axis=0)
    cse = np.zeros((len(wins), ntimes))

    # Get the windows over which we are summing as bins, not times
    win_segments = [np.int(w*new_fs) for w in wins]

    for wi, w in enumerate(win_segments):
        box = np.hstack((np.atleast_2d(boxcar(w)), -np.ones((1, np.int(0.15*new_fs))))).ravel()
        cse[wi,:] = convolve(auddiff, box, 'full')[:ntimes]

    cse[cse<0] = 0
    cse = cse/cse.max()

    return cse, auddiff

def get_peak_rate(envelope):
    env_diff = np.diff(np.concatenate((0, envelope), axis=None))
    env_diff[env_diff<0] = 0

    return env_diff

def get_mel_spectrogram(w, fs, wintime=0.025, steptime=0.010, nfilts=80, minfreq=0, maxfreq=None):
    ''' Make mel-band spectrogram
    Inputs:
        w [float] : audio signal vector
        fs [int] : sampling rate of audio signal
        wintime [float] : window size
        steptime [float] : step size (time resolution)
        nfilts [int] : number of mel-band filters
        minfreq [int] : Minimum frequency to analyze (in Hz)
        maxfreq [int] : Maximum frequency to analyze (in Hz). If none, defaults to fs/2
    
    Outputs:
        mel_spectrogram [array]: mel-band spectrogram
        freqs [array] : array of floats, bin edges of spectrogram

    '''
    if maxfreq is None:
        maxfreq = np.int(fs/2)

    pspec, e = powspec(w, sr=fs, wintime=wintime, steptime=steptime, dither=1)
    aspectrum, wts, freqs = audspec(pspec, sr=fs, nfilts=nfilts, fbtype='mel', minfreq=minfreq, maxfreq=maxfreq, sumpower=True, bwidth=1.0)
    mel_spectrogram = aspectrum**0.001

    return mel_spectrogram, freqs

def powspec(x, sr=8000, wintime=0.025, steptime=0.010, dither=1):
    '''
    # compute the powerspectrum and frame energy of the input signal.
    # basically outputs a power spectrogram
    #
    # each column represents a power spectrum for a given frame
    # each row represents a frequency
    #
    # default values:
    # sr = 8000Hz
    # wintime = 25ms (200 samps)
    # steptime = 10ms (80 samps)
    # which means use 256 point fft
    # hamming window
    #
    # $Header: /Users/dpwe/matlab/rastamat/RCS/powspec.m,v 1.3 2012/09/03 14:02:01 dpwe Exp dpwe $

    # for sr = 8000
    #NFFT = 256;
    #NOVERLAP = 120;
    #SAMPRATE = 8000;
    #WINDOW = hamming(200);
    '''

    winpts = int(np.round(wintime*sr))
    steppts = int(np.round(steptime*sr))

    NFFT = 2**(np.ceil(np.log(winpts)/np.log(2)))
    WINDOW = hanning(winpts).T

    # hanning gives much less noisy sidelobes
    NOVERLAP = winpts - steppts
    SAMPRATE = sr

    # Values coming out of rasta treat samples as integers,
    # not range -1..1, hence scale up here to match (approx)
    f,t,Sxx = spectrogram(x*32768, nfft=NFFT, fs=SAMPRATE, nperseg=len(WINDOW), window= WINDOW, noverlap=NOVERLAP)
    y = np.abs(Sxx)**2

    # imagine we had random dither that had a variance of 1 sample
    # step and a white spectrum.  That's like (in expectation, anyway)
    # adding a constant value to every bin (to avoid digital zero)
    if dither:
        y = y + winpts

    # ignoring the hamming window, total power would be = #pts
    # I think this doesn't quite make sense, but it's what rasta/powspec.c does

    # 2012-09-03 Calculate log energy - after windowing, by parseval
    e = np.log(np.sum(y))

    return y, e 

def audspec(pspectrum, sr=16000, nfilts=80, fbtype='mel', minfreq=0, maxfreq=8000, sumpower=True, bwidth=1.0):
    '''
    perform critical band analysis (see PLP)
    takes power spectrogram as input
    '''

    [nfreqs,nframes] = pspectrum.shape

    nfft = int((nfreqs-1)*2)
    freqs = []

    if fbtype == 'mel':
        wts, freqs = fft2melmx(nfft=nfft, sr=sr, nfilts=nfilts, bwidth=bwidth, minfreq=minfreq, maxfreq=maxfreq);
    elif fbtype == 'htkmel':
        wts = fft2melmx(nfft, sr, nfilts, bwidth, minfreq, maxfreq, 1, 1);
    elif fbtype == 'fcmel':
        wts = fft2melmx(nfft, sr, nfilts, bwidth, minfreq, maxfreq, 1, 0);
    elif fbtype == 'bark':
        wts = fft2barkmx(nfft, sr, nfilts, bwidth, minfreq, maxfreq);
    else:
        error(['fbtype ' + fbtype + ' not recognized']);

    wts = wts[:, 0:nfreqs]
    #figure(1)
    #plt.imshow(wts)

    # Integrate FFT bins into Mel bins, in abs or abs^2 domains:
    if sumpower:
        aspectrum = np.dot(wts, pspectrum)
    else:
        aspectrum = np.dot(wts, np.sqrt(pspectrum))**2.

    return aspectrum, wts, freqs

def get_mps(audio, audio_fs, window=0.5):
    '''
    Calculate the modulation power spectrum based on Theunissen lab code from soundsig package
    Inputs:
        audio [array]: sound pressure waveform
        audio_fs [int]: sampling rate of the sound
        window [float] : Time window for the modulation power spectrum
    
    Outputs:
        mps [array] : modulation power spectrum matrix, dimensions are spectral modulation x temporal modulation
        wt [array] : array of temporal modulation values (Hz) to go with temporal dimension axis of mps
        wf [array] : array of spectral modulation values (cyc/kHz) to go with spectral dimension axis of mps

    Example:
    
    import librosa
    s, sample_rate =librosa.load('/Users/liberty/Box/stimulidb/distractors/use_these_ITU_R/stim167_gargling__ITU_R_BS1770-3_12ms_200ms_-29LUFS.mp3')
    mps, wt, wf = get_mps(s, sample_rate)

    '''
    soundobj = sound.BioSound(audio, audio_fs)
    soundobj.mpsCalc(window=window)

    # Return the modulation power spectrum, which is in units of spectral modulation x temporal modulation
    # The actual temporal and spectral modulation values are given by wt and wf

    return soundobj.mps, soundobj.wt, soundobj.wf

if __name__=='main':
    stim_wav = '/Users/liberty/Documents/Austin/code/semantic_EOG/stimuli/blah.wav'
    print("Reading in %s"%stim_wav)
    [stim_fs, w] = wavfile.read(stim_wav)
    
    stim_fs = 44100 # Should be this for everyone, some of them were 48000 but played at 44kHz
    eeg_fs = 128

    print(stim_fs)
    envelope = get_envelope(w[:,0], stim_fs, eeg_fs, pad_next_pow2=True)
