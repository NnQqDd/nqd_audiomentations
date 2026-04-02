import random
import numpy as np
from audiomentations.core.transforms_interface import BaseWaveformTransform
import pyrubberband as pyrb


class RubberBandPitchShift(BaseWaveformTransform):
	def __init__(self, min_semitones=-5.0, max_semitones=5.0, p=0.5):
		super().__init__(p)
		self.min_semitones = min_semitones
		self.max_semitones = max_semitones
		self.p = p

	def randomize_parameters(self, a, b): # Skip samples, sample_rate
		super().randomize_parameters(a, b)
		pitch = np.random.uniform(self.min_semitones, self.max_semitones)
		self.parameters["pitch"] = pitch

	def apply(self, samples, sample_rate):
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

	def randomize_parameters(self, a, b):
		super().randomize_parameters(a, b)
		self.parameters["rate"] = np.random.uniform(self.min_rate, self.max_rate)

	def apply(self, pre_samples, sample_rate):
		pre_length = len(pre_samples)
		post_samples = pyrb.time_stretch(pre_samples, sample_rate, rate=self.parameters["rate"])
		
		if not self.fixed_length:
			return post_samples

		if self.fixed_length and pre_length > len(post_samples):
			return np.pad(post_samples, (0, pre_length - len(post_samples)), mode='constant')
	 	
		return post_samples[:pre_length] 



'''
# Phonecall effect
from audiomentations import Compose, HighPassFilter, LowPassFilter
phonecall = Compose([
	HighPassFilter(min_cutoff_freq=300, max_cutoff_freq=300, p=1.0),
	LowPassFilter(min_cutoff_freq=3400, max_cutoff_freq=3400, p=1.0),
])
'''