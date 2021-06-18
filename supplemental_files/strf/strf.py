#!/usr/bin/env python
# from gkTools.strf import strf

"""
strf.py (formerly strfTools)
Created by Garret Kurteff for Liberty Hamilton's Lab
v1.2
11/14/2019
"""
import scipy.io # For .mat files
import h5py # For loading hf5 files
import mne # For loading BrainVision files (EEG)
import numpy as np
from scipy.io import wavfile
import os
import matplotlib.pyplot as plt # For plotting
from matplotlib import cm, rcParams
import random
import itertools as itools
import csv
import logging
from supplemental_files.strf import spectools, fbtools, phn_tools, ridge, utils

# user paths
# gk = True
# if gk == True:
#     username = 'glk455'
#     data_dir = '/Users/%s/Desktop/serverData/' % username
#     eeg_file = '%s/eeg/%s/%s/%s_%s'%(data_dir, subj, blockid, blockid, data_suffix)
#     mic_file = '%saudio/%s/%s/%s_mic.wav'%(data_dir, subj, blockid, blockid)
#     spkr_file = '%saudio/%s/%s/%s_spkr.wav'%(data_dir, subj, blockid, blockid)
#     log_file = '/Users/%s/Desktop/git/onsetProd/transcription/logs/%s.txt' % (username, blockid)
#     spkr_env_file = '/Users/%s/Desktop/git/onsetProd/eeg/onsetProd_STRF/envelopes/%s_spkr.hf5' % (username, blockid)
#     mic_env_file = '/Users/%s/Desktop/git/onsetProd/eeg/onsetProd_STRF/envelopes/%s_mic.hf5' % (username, blockid)

# liberty = False
# if liberty == True:
#     data_dir = '/Users/liberty/Documents/Austin/data/EEG/onsetProd/'
#     eeg_file = '%s/%s/%s/%s_%s'%(data_dir, subj, blockid, blockid, data_suffix)
#     mic_file = '%s/%s/%s/%s_mic.wav'%(data_dir, subj, blockid, blockid)
#     spkr_file = '%s/%s/%s/%s_spkr.wav'%(data_dir, subj, blockid, blockid)
#     log_file = '/Users/liberty/Documents/Austin/code/onsetProd/transcription/logs/%s.txt'%(blockid)
#     spkr_env_file = '/Users/liberty/Documents/Austin/code/onsetProd/eeg/onsetProd_STRF/envelopes/%s_spkr.hf5' % (blockid)
#     mic_env_file = '/Users/liberty/Documents/Austin/code/onsetProd/eeg/onsetProd_STRF/envelopes/%s_mic.hf5' % (blockid)

# Functions
def raw(user, subj, block, data_suffix, preload=True, montage_path='../montage/AP-128.bvef'):
	'''
	use mne to load in raw data
	'''
	blockid = subj + '_' + block
	if user == 'liberty':
		data_dir = '/Users/%s/Documents/Austin/data/EEG/onsetProd/' % user
		eeg_file = '%s/%s/%s/%s_%s'%(data_dir, subj, blockid, blockid, data_suffix)
	elif user == 'glk455' or user == 'kfsh':
		data_dir = '/Users/%s/Desktop/serverData/' % user
		eeg_file = '%s/eeg/%s/%s/%s_%s'%(data_dir, subj, blockid, blockid, data_suffix)
	print("Loading the neural data for %s." % blockid)
	if eeg_file[-4:] == 'vhdr':
		raw = mne.io.read_raw_brainvision(eeg_file, preload=preload)
	if eeg_file[-3:] == 'fif':
		raw = mne.io.read_raw_fif(eeg_file, preload=preload)
	#raw.notch_filter(np.arange(60, 128, 60))
	# Print which are the bad channels, but don't get rid of them yet...
	raw.pick_types(eeg=True, meg=False, exclude=[])
	bad_chans = raw.info['bads']
	print("Bad channels are: ")
	print(bad_chans)
	# Get rid of stimtrak channel
	if 'STI 014' in raw.info['ch_names']:
		print("Dropping StimTrak channel.")
		raw.drop_channels(['STI 014'])
	if 'vEOG' in raw.info['ch_names']:
		print("Dropping vEOG channel.")
		raw.drop_channels(['vEOG'])
	if 'hEOG' in raw.info['ch_names']:
		print("Dropping hEOG channel.")
		raw.drop_channels(['hEOG'])
	# Set the montage from the cap file provided by BrainVision that tells you where
	# each sensor is located on the head (for topomaps)
	montage_file = os.path.abspath(montage_path)
	raw.set_montage(montage_file)
	if raw.info['custom_ref_applied'] == False:
		print("Re-referencing to TP9 and TP10 because it doesn't seem to have been done")
		raw.set_eeg_reference(ref_channels=['TP9','TP10'])
		raw.filter(2., 8.) # Filter between 2 and 8 Hz

	return(raw, bad_chans)
def log(user, subj, block, task_start):
	'''
	Read in information from the log file to help with segmenting the STRF envelopes.
	'''
	blockid = subj + '_' + block
	if user == 'liberty':
		log_file = '/Users/%s/Documents/Austin/code/onsetProd/transcription/logs/%s.txt'%(user, blockid)
	elif user == 'glk455' or user == 'kfsh':
		log_file = '/Users/%s/Desktop/git/onsetProd/transcription/logs/%s.txt' % (user, blockid)
	# Read in information from the log file
	print("Looking for log %s" % log_file)
	condition = []
	block = []
	times = []
	with open(log_file) as tsvfile:
		next(tsvfile), next(tsvfile), next(tsvfile) # skip the header
		reader = csv.DictReader(tsvfile, delimiter='\t')
		for row in reader:
			condition.append(row['TrialPart'])
			block.append(row['CurrentBlock'])
			times.append(float(row['Time']))
	# convert device time to task time 
	times = [x+task_start-times[0] for x in times]
	print("Done reading from log file. %d time points extracted." % (len(times)))
	return(condition, block, times)
def raw_by_condition(raw, task_times, task_block, task_condition):
	'''
	Splits a raw EEG file by behavioral condition.
	'''
	# empty lists of np arrays to hstack after we loop
	el_list = []
	sh_list = []
	prod_list = []
	# Get np arrays for our three conditions
	for i in range(len(task_times)-1):
		if task_condition[i] == 'listen':
			if task_block[i] == 'echolalia': # el
				# get eeg
				start = int(task_times[i]*128) # convert to sfreq
				stop = int(task_times[i+1]*128) # convert to sfreq
				raw_segment, el_times = raw.get_data(start=start, stop=stop, return_times=True)
				el_list.append(raw_segment)
			if task_block[i] == 'shuffled': # sh
				# get eeg
				start = int(task_times[i]*128) # convert to sfreq
				stop = int(task_times[i+1]*128) # convert to sfreq
				raw_segment, sh_times = raw.get_data(start=start, stop=stop, return_times=True)
				sh_list.append(raw_segment)
		if task_condition[i] == 'readRepeat': # prod
			# get eeg
			start = int(task_times[i]*128) # convert to sfreq
			stop = int(task_times[i+1]*128) # convert to sfreq
			raw_segment, prod_times = raw.get_data(start=start, stop=stop, return_times=True)
			prod_list.append(raw_segment)
	# make our data arrays using information from our log loop
	el_array = np.hstack((el_list))
	sh_array = np.hstack((sh_list))
	prod_array = np.hstack((prod_list))
	return(el_array, sh_array, prod_array)
def get_envelope(raw, channel, wav_file, env_file):
	'''
	Returns the envelope of a sound given a wav file.
	'''
	print(env_file)
	print(os.path.isfile(env_file))
	# load wavfile

	fs, wav = wavfile.read(wav_file)
	if os.path.isfile(env_file) == True:
		print("Loading previously generated %s_env_file" % channel)
		with h5py.File(env_file,'r') as fh:
			print("Successfully opened %s." % env_file)
			if channel == 'mic':
				env2 = fh['mic_env'][:]
			if channel == 'spkr':
				env2 = fh['spkr_env'][:]
	else:
		print(raw.info['sfreq'])
		new_fs = np.int(raw.info['sfreq'])
		all_envs = dict()
		all_sounds = dict()

		print("Making sound envelopes for %s.. this may take awhile...." % channel)
		wav = wav/wav.max()
		if channel == 'mic':
			all_sounds['mic'] = wav
		if channel == 'spkr':
			all_sounds['spkr'] = wav
		envelopes = []
		# make the chunk length a power of two
		chunk_len = 2.**15
		# Calculate how many chunks there should be based on this length
		nchunks = np.int(len(wav)/chunk_len)
		print(len(wav)/nchunks)
		
		for n in np.arange(nchunks):
			if np.mod(n,100):
				print("Chunk %d out of %d"%(n, nchunks))
				envelopes.append(spectools.get_envelope(wav[np.int(n*chunk_len):np.int((n+1)*chunk_len)], fs, new_fs))    
			if np.int(nchunks*chunk_len) < len(wav):
				print("Still have stuff at the end")
				envelopes.append(spectools.get_envelope(wav[np.int(nchunks*chunk_len):], fs, new_fs))  
				
		print("Done with %s envelopes" % channel)
		print(np.vstack((envelopes)).shape)

		# Concatenate all the envelope chunks
		print("Concatenating envelope chunks")
		env=np.vstack((envelopes))
		# Re-scale to maximum
		print("Rescaling to maximum")
		env = env/env.max()

		# Resample the envelope to the same size as the response
		print("Resampling the data to be the same length as the EEG")
		env = scipy.signal.resample(env, raw.get_data().shape[1])

		# soft de-spiking - audio more than 5 STD away from 0 will be squashed, 
		# then we multiply back to get into orig. range
		# This puts the speech into a better range
		print("Soft de-spiking")
		mm1 = 5*np.std(env)
		env2 = mm1*np.tanh(env/mm1);
		env2 = env2/env2.max()

		print("Setting small values of envelope to 0, because audio can be noisy")
		env2[env2<0.02] = 0

		# Save the envelope so we don't have to run the thing again
		with h5py.File(env_file,'w') as fh:
			if channel == 'mic':
				fh.create_dataset('/mic_env', data = np.array(env2))
			if channel == 'spkr':
				fh.create_dataset('/spkr_env', data = np.array(env2))
	return(env2, fs)
def plt_env_chunk(env2, fs, raw):
	'''
	Plots the envelope from get_envelope() superimposed on the raw waveform in a 40s window.
	'''
	plt.figure(figsize=(10,2))
	plt.plot(env2[np.int(raw.info['sfreq']*500):np.int(raw.info['sfreq']*520)])
	# LPF
	nyq = fs/2. #Nyquist frequency
	cof = 500.
	b, a = scipy.signal.butter(3, cof/nyq, 'low'); #this designs a 3-pole low-pass filter
	env3 = scipy.signal.filtfilt(b, a, env2.T).T
	#print(env3.shape)
	plt.plot(env3[np.int(raw.info['sfreq']*500):np.int(raw.info['sfreq']*520)])
def plt_sound_envelope(subj, block, raw, spkr_env2, mic_env2, user="glk455", start_time=500, end_time=530, task_start=0):
	blockid = subj + '_' + block
	# get path to wavs based on who is running the function
	if user == "glk455" or user == "kfsh":
		spkr_file = ('/Users/%s/Desktop/serverData/audio/%s/%s/%s_spkr.wav' % 
			(user, subj, blockid, blockid))
		mic_file = ('/Users/%s/Desktop/serverData/audio/%s/%s/%s_mic.wav' % 
			(user, subj, blockid, blockid))
	elif user == "liberty":
		spkr_file = ('/Users/%s/Documents/Austin/data/EEG/onsetProd/audio/%s/%s/%s_spkr.wav' % 
			(user, subj, blockid, blockid))
		mic_file = ('/Users/%s/Documents/Austin/data/EEG/onsetProd/audio/%s/%s/%s_mic.wav' % 
			(user, subj, blockid, blockid))
	# load wavfiles
	spkr_fs, spkr_wav = wavfile.read(spkr_file)
	spkr_wav = spkr_wav[np.int(task_start*spkr_fs):]
	mic_fs, mic_wav = wavfile.read(mic_file)
	mic_wav = mic_wav[np.int(task_start*mic_fs):]
	spkr_wav = spkr_wav/spkr_wav.max()
	mic_wav = mic_wav/mic_wav.max()
	# get our stuff to plot
	new_fs = np.int(raw.info['sfreq'])
	t1 = np.linspace(0, len(spkr_env2)/new_fs, len(spkr_env2))
	t2 = np.linspace(0, len(spkr_env2)/new_fs, len(spkr_wav))
	trange1 = np.arange(np.int(new_fs*start_time), np.int(new_fs*end_time))
	trange2 = np.arange(np.int(spkr_fs*start_time), np.int(spkr_fs*end_time))
	plt.figure(figsize=(15,3))
	plt.subplot(2,1,1)
	# plot the waveforms
	plt.plot(t2[trange2],spkr_wav[trange2])
	plt.plot(t2[trange2],mic_wav[trange2])
	# plot the envelopes
	plt.plot(t1[trange1],spkr_env2[trange1])
	plt.plot(t1[trange1],mic_env2[trange1])
	plt.axis([start_time, end_time, -1, 1]) 
def envelopes_by_condition(spkr_envelope, mic_envelope, task_times, task_block, task_condition):
	# segment envelope by condition
	el_env = []
	sh_env = []
	prod_env = []
	# chunk the long envelope into smaller envelopes (sh, el, prod)
	for i in range(len(task_times)-1):
		if task_condition[i] == 'listen':
			if task_block[i] == 'echolalia': # el
				# get env
				start = int(task_times[i]*128) # convert to sfreq
				stop = int(task_times[i+1]*128) # convert to sfreq
				env_segment = spkr_envelope[start:stop]
				el_env.append(env_segment)
			if task_block[i] == 'shuffled': # sh
				# get env
				start = int(task_times[i]*128) # convert to sfreq
				stop = int(task_times[i+1]*128) # convert to sfreq
				env_segment = spkr_envelope[start:stop]
				sh_env.append(env_segment)
		if task_condition[i] == 'readRepeat': # prod
			# get env
			start = int(task_times[i]*128) # convert to sfreq
			stop = int(task_times[i+1]*128) # convert to sfreq
			env_segment = mic_envelope[start:stop]
			#env_segment = env3[start:stop]
			prod_env.append(env_segment)
	# make a fat stack
	el_env = np.vstack((el_env))
	sh_env = np.vstack((sh_env))
	prod_env = np.vstack((prod_env))
	return(el_env, sh_env, prod_env)
def strf(resp, stim,
	delay_min=0, delay_max=0.6, wt_pad=0.0, alphas=np.hstack((0, np.logspace(-3,5,20))),
	use_corr=True, single_alpha=True, nboots=20, sfreq=128, vResp=[],vStim=[], flip_resp=False):
	'''
	Run the STRF model.
	* wt_pad: Amount of padding for delays, since edge artifacts can make weights look weird
	* use_corr: Use correlation between predicted and validation set as metric for goodness of fit
	* single_alpha: Use the same alpha value for all electrodes (helps with comparing across sensors)
	* Logspace was previously -1 to 7, changed for smol stim strf in may 20
	'''
	# Populate stim and resp lists (these will be used to get tStim and tResp, or vStim and vResp)
	stim_list = []
	stim_sum= []
	train_or_val = [] # Mark for training or for validation set
	np.random.seed(6655321)
	if flip_resp == True:
		resp = resp.T
		if len(vResp) >= 1:
			vResp = vResp.T
	# Load stimulus and response
	if resp.shape[1] != stim.shape[0]:
		logging.warning("Resp and stim do not match! This is a problem")
		print("Resp shape: %r || Stim shape: %r" % (resp.shape[1],stim.shape[0]))
	nchans, ntimes = resp.shape
	print(nchans, ntimes)
	# RUN THE STRFS
	# For logging compute times, debug messages
	logging.basicConfig(level=logging.DEBUG)
	delays = np.arange(np.floor((delay_min-wt_pad)*sfreq), np.ceil((delay_max+wt_pad)*sfreq), dtype=np.int) 
	print("Delays:", delays)
	# Regularization parameters (alphas - also sometimes called lambda)
	# MIGHT HAVE TO CHANGE ALPHAS RANGE... e.g. alphas = np.hstack((0, np.logspace(-2,5,20)))
	# alphas = np.hstack((0, np.logspace(2,8,20))) # Gives you 20 values between 10^2 and 10^8
	# alphas = np.hstack((0, np.logspace(-1,7,20))) # Gives you 20 values between 10^-2 and 10^5
	# alphas = np.logspace(1,8,20) # Gives you 20 values between 10^1 and 10^8
	nalphas = len(alphas)
	all_wts = []
	all_corrs = []
	# Train on 80% of the trials, test on 
	# the remaining 20%.
	# Z-scoring function (assumes time is the 0th dimension)
	zs = lambda x: (x-x[np.isnan(x)==False].mean(0))/x[np.isnan(x)==False].std(0)
	resp = zs(resp.T).T
	if len(vResp) >= 1 and len(vStim) >= 1:
		# Create training and validation response matrices.
		# Time must be the 0th dimension.
		tResp = resp[:,:np.int(0.8*ntimes)].T
		vResp = resp[:,np.int(0.8*ntimes):].T
		# Create training and validation stimulus matrices
		tStim_temp = stim[:np.int(0.8*ntimes),:]
		vStim_temp = stim[np.int(0.8*ntimes):,:]
		tStim = utils.make_delayed(tStim_temp, delays)
		vStim = utils.make_delayed(vStim_temp, delays)
	else: # if vResp and vStim were passed into the function
		tResp = resp
		tStim = stim
	chunklen = np.int(len(delays)*4) # We will randomize the data in chunks 
	nchunks = np.floor(0.2*tStim.shape[0]/chunklen).astype('int')
	nchans = tResp.shape[1] # Number of electrodes/sensors

	# get a strf
	wt, corrs, valphas, allRcorrs, valinds, pred, Pstim = ridge.bootstrap_ridge(tStim, tResp, vStim, vResp, 
																		  alphas, nboots, chunklen, nchunks, 
																		  use_corr=use_corr,  single_alpha = single_alpha, 
																		  use_svd=False, corrmin = 0.05,
																		  joined=[np.array(np.arange(nchans))])
	print("wt shape:")
	print(wt.shape)
	
	# If we decide to add some padding to our model to account for edge artifacts, 
	# get rid of it before returning the final strf
	if wt_pad>0:
		print("Reshaping weight matrix to get rid of padding on either side")
		orig_delays = np.arange(np.floor(delay_min*sfreq), np.ceil(delay_max*sfreq), dtype=np.int) 

		# This will be a boolean mask of only the "good" delays (not part of the extra padding)
		good_delays = np.zeros((len(delays), 1), dtype=np.bool)
		int1, int2, good_inds = np.intersect1d(orig_delays,delays,return_indices=True)
		for g in good_inds:
			good_delays[g] = True	#wt2 = wt.reshape((len(delays), -1, wt.shape[1]))[len(np.where(delays<0)[0]):-(len(np.where(delays<0)[0])),:,:]
		print(delays)
		print(orig_delays)
		# Reshape the wt matrix so it is now only including the original delay_min to delay_max time period instead
		# of delay_min-wt_pad to delay_max+wt_pad
		wt2 = wt.reshape((len(delays), -1, wt.shape[1])) # Now this will be ndelays x nfeat x nchans
		wt2 = wt2[good_delays.ravel(), :, :].reshape(-1, wt2.shape[2])
	else:
		wt2 = wt
	print(wt2.shape)
	all_wts.append(wt2)
	all_corrs.append(corrs)
	return(all_corrs, all_wts, tStim, tResp, vStim, vResp, valphas, pred)	
def bootstrap(raw, tStim, tResp, vStim, vResp, valphas, all_corrs,
		nboots=20, nboots_shuffle=100, delay_min=0, delay_max=0.6, wt_pad=0.1):
	'''
	Determine whether the correlations we see for model performance are significant
	by shuffling the data and re-fitting the models to see what "random" performance
	would look like.
	* tstim: training stimulus matrix
	* vstim: validation stimulus matrix
	* nboots: How many bootstraps to do
	* nboots_shuffle: How many bootstraps to do on our shuffled dataset.
	  The minimum p-value you can get from this is 1/nboots_shuffle.
	  So for nboots_shuffle = 100, you can get p_values from 0.01 to 1.0.
	'''
	np.random.seed(0)
	all_corrs_shuff = [] # List of lists
	logging.basicConfig(level=logging.DEBUG)
	# get info from the raw
	fs = raw.info['sfreq']
	delays = np.arange(np.floor((delay_min-wt_pad)*fs), np.ceil((delay_max+wt_pad)*fs), dtype=np.int) 
	print("Delays:", delays)
	chunklen = np.int(len(delays)*4) # We will randomize the data in chunks 
	nchunks = np.floor(0.2*tStim.shape[0]/chunklen).astype('int')
	# How many bootstraps to do for determining bootstrap significance
	nresp, nelecs = tStim.shape
	allinds = range(nresp)
	print("Determining significance of the correlation values using a bootstrap with %d iterations"%(nboots_shuffle))
	for n in np.arange(nboots_shuffle):
		print("Bootstrap %d/%d"%(n+1, nboots_shuffle))
		indchunks = list(zip(*[iter(allinds)]*chunklen))
		random.shuffle(indchunks)
		shuff_inds = list(itools.chain(*indchunks[:nchunks]))
		tStim_shuff = tStim.copy()
		tResp_shuff = tResp.copy()
		tStim_shuff = tStim_shuff[shuff_inds,:]
		tResp_shuff = tResp_shuff[:len(shuff_inds),:]
		corrs_shuff = ridge.ridge.eigridge_corr(tStim_shuff, vStim, tResp_shuff, vResp, [valphas[0]])
		all_corrs_shuff.append(corrs_shuff)
	# all_corrs_shuff is a list of length nboots_shuffle
	# Each element is the correlation for a random model for each of the 64 electrodes for that iteration
	# We use this to figure out [nboots_shuffle] possible values of random correlations for each electrode,
	# then use this to determine if the correlations we're actually measuring with the non-shuffled data are 
	# significantly higher than this
	# Get the p values of each of the significant correlations
	all_pvals = [] 
	all_c_s=np.vstack((all_corrs_shuff)) # Get the shuffled random correlations for this model
	# Is the correlation of the model greater than the shuffled correlation for random data?
	h_val = np.array([all_corrs[0] > all_c_s[c] for c in np.arange(len(all_c_s))])
	print(h_val.shape)
	# Count the number of times out of nboots_shuffle that the correlation is greater than 
	# random, subtract from 1 to get the bootstrapped p_val (one per electrode)
	p_val = 1-h_val.sum(0)/nboots_shuffle
	all_pvals.append(p_val)
	# Find the maximum correlation across the shuffled and real data
	max_corr = np.max(np.vstack((all_corrs_shuff[0], all_corrs[0])))+0.01
	return(all_pvals, max_corr, all_corrs_shuff)
def plt_correlations(raw, all_corrs, all_pvals, max_corr, all_corrs_shuff, tResp):
	nchans = tResp.shape[1] # Number of electrodes/sensors
	# Plot the correlations for each channel separately
	plt.figure(figsize=(15,3));
	plt.plot(all_corrs[0]);
	# Plot an * if the correlation is significantly higher than chance at p<0.05
	for i,p in enumerate(all_pvals[0]):
		if p<0.05:
			plt.text(i, max_corr, '*');
	# Plot the shuffled correlation distribution
	shuffle_mean = np.vstack((all_corrs_shuff)).mean(0)
	shuffle_stderr = np.vstack((all_corrs_shuff)).std(0)#/np.sqrt(nboots_shuffle)
	plt.fill_between(np.arange(nchans), shuffle_mean-shuffle_stderr, 
					 shuffle_mean+shuffle_stderr, color=[0.5, 0.5, 0.5]);
	plt.plot(shuffle_mean, color='k');
	plt.gca().set_xticks(np.arange(len(all_corrs[0])));
	plt.gca().set_xticklabels(raw.info['ch_names'], rotation=90);
	plt.xlabel('Channel');
	plt.ylabel('Model performance');
	plt.legend(['Actual data','Null distribution']);
def plt_topo(raw, max_corr, all_corrs, all_pvals, montage_path='../montage/AP-128.bvef'):
	significant_corrs = np.array(all_corrs[0])
	significant_corrs[np.array(all_pvals[0])>0.05] = 0
	plt.figure(figsize=(5,5))
	print(['eeg']*2)
	info = mne.create_info(ch_names=raw.info['ch_names'][:64], sfreq=raw.info['sfreq'], ch_types=64*['eeg'])
	raw2 = mne.io.RawArray(np.zeros((64,10)), info)
	raw2.set_montage(montage_path)
	mne.viz.plot_topomap(significant_corrs, raw2.info, vmin=-max_corr, vmax=max_corr)
def plt_corr_hist(all_corrs, all_corrs_shuff, max_corr):
	np.vstack((all_corrs_shuff)).ravel().shape
	plt.hist(np.hstack((all_corrs_shuff)).ravel(), bins=np.arange(-0.2,max_corr,0.005), alpha=0.5, density=True)
	plt.hist(all_corrs[0], bins=np.arange(-0.2,max_corr,0.005), alpha=0.5, density=True)
	plt.xlabel('Model fits (r-values)')
	plt.ylabel('Number')
	plt.title('Correlation histograms')
	plt.legend(['Null distribution', 'EEG data'])
def plt_montage(raw, array, all_wts, montage_path='../montage/AP-128.bvef', delay_min=0, delay_max=0.6):
	fs = raw.info['sfreq']
	resp = array
	nchans, ntimes = resp.shape
	print(1.0/fs)
	t = np.linspace(delay_min, delay_max, all_wts[0].shape[0])
	wmax = np.abs(all_wts[0]).max()
	for i in np.arange(nchans):
		el_name = raw.info['ch_names'][i]
		mont = mne.channels.read_montage(montage_path, ch_names=[el_name]);
		xy = mont.get_pos2d()
		pos = [xy[0][0]+10, xy[0][1]+10, 0.3, 0.3]
		ax = plt.subplot(8,8,i+1)
		ax.plot(t,all_wts[0][:,i])
		ax.hlines(0, t[0], t[-1])
		ax.set_ylim([-wmax, wmax])
		ax.set_position(pos)
		plt.title(el_name)
def plt_avg_wts(all_wts, delay_min=0, delay_max=0.6):
	t = np.linspace(delay_min, delay_max, all_wts[0].shape[0])
	wmax = np.abs(all_wts[0].mean(1)).max()
	plt.plot(t,all_wts[0].mean(1))
	plt.hlines(0, delay_min, delay_max, color='k', linewidth=0.5)
	plt.axis([0, 0.6, -wmax, wmax])
	plt.gca().invert_xaxis()
	plt.xlabel('Time delay (s)')
	#tick_right()
def plt_scatter2(corrs1, corrs2):
	'''
	Scatterplot comparing the correlations of two conditions
	'''
	max_corr = np.max((corrs1, corrs2))
	min_corr = np.min((corrs1, corrs2))

	plt.plot(corrs1, corrs2, '.') # One dot per electrode
	plt.plot([min_corr, max_corr], [min_corr, max_corr], 'k--', linewidth=0.5)
	plt.xlabel('Corr 1')
	plt.ylabel('Corr 2')
def plt_scatter3(corrs1, corrs2, corrs3):
	'''
	Scatterplot comparing the correlations of three conditions
	'''
	print("WIP")


