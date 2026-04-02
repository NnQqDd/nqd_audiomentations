import random
import numpy as np


def rms(wave: np.ndarray, eps=1e-9) -> np.ndarray:
    return np.sqrt(np.mean(wave ** 2, axis=0, keepdims=True) + eps)


def random_select_segment(wave, length):
    if random.random() <= 0.5:
        start = random.randint(0, wave.shape[0] - length)
        wave = wave[start:start + length]
    else:
        end = random.randint(length, wave.shape[0])
        wave = wave[end-length:end]
    return wave


def random_pad(wave, length):
    if len(wave) >= length:
        return wave
    pad_width = length - len(wave)
    pad_left = random.randint(0, pad_width)
    pad_right = pad_width - pad_left
    wave = np.pad(wave, (pad_left, pad_right), mode='constant')
    return wave


def peak_normalize(wave: np.ndarray, optional=True, eps=1e-9):
    peak = np.max(np.abs(wave))
    if peak > 1 or not optional:
        wave /= (peak + eps)
    return wave


def add_noise(
    speech_wave: np.ndarray, # 1-D, same sample rate as noise wave
    noise_wave: np.ndarray,
    min_snr: float,
    max_snr: float,
    repeat: bool=True,
) -> np.ndarray:
    speech_len = speech_wave.shape[0]
    noise_len = noise_wave.shape[0]
    if noise_len >= speech_len:
        noise_wave = random_select_segment(noise_wave, speech_len)
    elif repeat:
        reps = int(np.ceil(speech_len/ noise_len))
        noise_wave = np.tile(noise_wave, reps)[:speech_len]
    else:
        pad_left = random.randint(0, (speech_len - noise_len))
        pad_right = (speech_len - noise_len) - pad_left
        noise_wave = np.pad(noise_wave, (pad_left, pad_right), mode="constant")

    snr_db = random.uniform(min_snr, max_snr)
    rms_s = rms(speech_wave)
    rms_n = rms(noise_wave)

    gain = (rms_s / (rms_n + 1e-9)) * (10 ** (-snr_db / 20))
    noise_scaled = noise_wave * gain

    mixed = speech_wave + noise_scaled
    return mixed
