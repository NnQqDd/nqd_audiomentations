import random
import numpy as np
from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations import HighPassFilter, LowPassFilter, Compose
import pyrubberband as pyrb
from scipy.signal import fftconvolve
import librosa
from .rir_sim import generate_rirs


class RubberBandPitchShift(BaseWaveformTransform):
	def __init__(self, min_semitones=-5.0, max_semitones=5.0, p=0.5):
		super().__init__(p)
		self.min_semitones = min_semitones
		self.max_semitones = max_semitones
		self.p = p

	def randomize_parameters(self, x, y): # Skip samples, sample_rate
		super().randomize_parameters(x, y)
		pitch = random.uniform(self.min_semitones, self.max_semitones)
		self.parameters["pitch"] = pitch

	def apply(self, samples, sample_rate):
		assert len(samples.shape) == 1, "Input must be a mono numpy waveform!"
		return pyrb.pitch_shift(
			samples, 
			sample_rate, 
			n_steps=self.parameters["pitch"]
		)


class RubberBandTimeStretch(BaseWaveformTransform):
	def __init__(self, min_rate=0.5, max_rate=2.0, leave_length_unchanged=True, p=0.5):
		super().__init__(p)
		self.min_rate = min_rate
		self.max_rate = max_rate
		self.fixed_length = leave_length_unchanged
		self.p = p

	def randomize_parameters(self, x, y):
		super().randomize_parameters(x, y)
		self.parameters["rate"] = random.uniform(self.min_rate, self.max_rate)

	def apply(self, pre_samples, sample_rate):
		assert len(pre_samples.shape) == 1, "Input must be a mono numpy waveform!"
		pre_length = len(pre_samples)
		post_samples = pyrb.time_stretch(pre_samples, sample_rate, rate=self.parameters["rate"])
		
		if not self.fixed_length:
			return post_samples

		if self.fixed_length and pre_length > len(post_samples):
			return np.pad(post_samples, (0, pre_length - len(post_samples)), mode='constant')
	 	
		return post_samples[:pre_length] 


class SyntheticReverb():
	# Following WHAMR
	DEFAULT_PRESETS = {
		"room": {
			"L": [5.0, 10.0],
			"W": [5.0, 10.0],
			"H": [3.0, 4.0],
		},
		"T60": {
			"low": [0.1, 0.3],
			"med": [0.2, 0.6],
			"high": [0.4, 1.0],
		},
		"mic": {
			"center_jitter": 0.2,
			"height": [0.9, 1.8],
		},
		"sources": {
			"height": [0.9, 1.8],
			"dist_from_mic": [0.66, 2.0],
			"angle": [0, 6.283185307179586],  # 2 * pi
		},
	}
	def __init__(self, presets=None, p=0.5):
		if not presets:
			self.presets = self.DEFAULT_PRESETS
		else:
			self.presets = self.DEFAULT_PRESETS
		self.p = p
	
	def apply(self, samples, sample_rate):
		assert len(samples.shape) == 1, "Input must be a mono numpy waveform!"
		if random.random() >= self.p:
			return samples
		pre_length = len(samples)
		rirs = generate_rirs(presets=self.presets, fs=sample_rate, n_sources=1, n_mics=1) 
		samples = fftconvolve(samples, rirs[0][0], mode="full")
		samples = samples[:pre_length]
		return samples

	def __call__(self, samples, sample_rate):
		return self.apply(samples, sample_rate)


class PeakNormalize():
	def __init__(self, eps=1e-9, optional=True):
		self.eps = eps
		self.optional = optional
	
	def apply(self, samples, _):		
		peak = np.max(np.abs(samples))
		if peak > 1 or not self.optional:
			samples = samples / (peak + self.eps)
		return samples

	def __init__(self, samples, sample_rate):
		return self.apply(samples, sample_rate)


class PhoneCallEffect():
	def __init__(self, p=0.5):
		self.p = p
		self.effect = Compose([
			HighPassFilter(min_cutoff_freq=300, max_cutoff_freq=300, p=1.0),
			LowPassFilter(min_cutoff_freq=3400, max_cutoff_freq=3400, p=1.0),
		])

	def apply(self, samples, sample_rate):
		if random.random() >= self.p:
			return samples
		pre_sample_rate = sample_rate
		samples_8k = librosa.resample(samples, orig_sr=pre_sample_rate, target_sr=8000)
		phone_samples = self.effect(samples=samples_8k, sample_rate=8000)
		phone_samples = librosa.resample(phone_samples, orig_sr=8000, target_sr=pre_sample_rate)
		return phone_samples
	
	def __call__(self, samples, sample_rate):
		return self.apply(samples, sample_rate)
	
'''
# Phonecall effect
from audiomentations import Compose, HighPassFilter, LowPassFilter

'''