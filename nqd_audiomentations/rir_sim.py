import random
import math
import numpy as np
import pyroomacoustics as pra


def sabine_alpha_from_t60(t60, room_dim):
    """
    Compute mean absorption coefficient (alpha_bar) for target T60 using Sabine:
      T60 = 0.161 * V / (alpha_bar * S)
    Clip to (0.01, 0.99) to remain stable for shoebox sim.
    """
    L, W, H = room_dim
    V = L * W * H
    S = 2 * (L * W + L * H + W * H)
    if t60 <= 0:
        return 0.5
    alpha = 0.161 * V / (t60 * S)
    return float(np.clip(alpha, 0.01, 0.99))


def _sample_room_and_t60(presets, t60_class=None):
    # sample room dims
    L = random.uniform(*presets["room"]["L"])
    W = random.uniform(*presets["room"]["W"])
    H = random.uniform(*presets["room"]["H"])
    room_dim = [L, W, H]

    # pick t60 class if not provided
    classes = list(presets["T60"].keys())
    if t60_class is None:
        t60_class = random.choice(classes)
    if t60_class not in presets["T60"]:
        raise ValueError(f"t60_class must be one of {classes}")

    target_t60 = random.uniform(*presets["T60"][t60_class])
    return room_dim, target_t60, t60_class


def _place_mics_near_center(room_dim, n_mics, center_jitter, height_range):
    L, W, H = room_dim
    center_x = L / 2.0
    center_y = W / 2.0

    mic_positions = []
    for _ in range(n_mics):
        jitter_x = random.uniform(-center_jitter, center_jitter)
        jitter_y = random.uniform(-center_jitter, center_jitter)
        x = np.clip(center_x + jitter_x, 0, L)
        y = np.clip(center_y + jitter_y, 0, W)
        z = random.uniform(*height_range)
        mic_positions.append([x, y, z])
    return mic_positions


def _place_sources_around_mic(room_dim, mic_pos, n_sources, dist_range, angle_range, height_range, margin=0.3):
    L, W, H = room_dim
    sources = []
    for _ in range(n_sources):
        r = random.uniform(*dist_range)
        theta = random.uniform(*angle_range)
        x = mic_pos[0] + r * math.cos(theta)
        y = mic_pos[1] + r * math.sin(theta)
        z = random.uniform(*height_range)
        if margin > x:
            x = margin
        if x > L - margin:
            x = L - margin
        if margin > y:
            y = margin
        if y > W - margin:
            y = W - margin
        if 0 > z:
            z = 0
        if z > H - margin:
            z = H - margin
        sources.append([x, y, z])
    return sources


def generate_rirs(presets, fs, n_sources, n_mics, t60_class=None):
    """
    Generate RIRs following WHAMR sampling for a single mixture.

    Inputs:
      - fs: sampling rate (int)
      - n_sources: number of sources (int)
      - n_mics: number of microphones (int)
      - t60_class: optional string 'low'|'med'|'high' to force a class; otherwise chosen randomly

    Output:
      - rirs: a list of length n_sources; each element is a list length n_mics such that
              rirs[s][m] is a numpy array RIR for source s and mic m.
    """
    # sample room and T60
    room_dim, target_t60, chosen_class = _sample_room_and_t60(presets, t60_class)

    # compute absorption estimate from Sabine and add small randomization
    absorption = sabine_alpha_from_t60(target_t60, room_dim)
    
    # choose max_order (more T60 -> more image sources). Keep reasonable bounds.
    max_order = int(np.clip(3 + target_t60 * 6.0, 1, 30))

    # create shoebox room
    room = pra.ShoeBox(room_dim, fs=fs, max_order=max_order, absorption=absorption)

    # place mics near center with jitter (WHAMR style)
    mic_positions = _place_mics_near_center(room_dim, n_mics, presets["mic"]["center_jitter"], presets["mic"]["height"])

    # add mic array (shape 3 x n_mics)
    mic_array = np.array(mic_positions).T
    room.add_microphone_array(mic_array)

    # For WHAMR they often place sources relative to the (single) mic near center.
    # When multiple mics exist, use the first mic as the reference for distance sampling.
    ref_mic = mic_positions[0]

    # place sources around reference mic
    sources = _place_sources_around_mic(
        room_dim,
        ref_mic,
        n_sources,
        presets["sources"]["dist_from_mic"],
        presets["sources"]["angle"],
        presets["sources"]["height"],
    )

    # add sources to room
    for s_pos in sources:
        room.add_source(s_pos)

    # compute rirs
    room.compute_rir()

    # build return structure rirs[s][m] where s in [0..n_sources-1], m in [0..n_mics-1]
    rirs = []
    for s_idx in range(n_sources):
        row = []
        for m_idx in range(n_mics):
            # pyroomacoustics stores as room.rir[mic][source]
            rir = np.array(room.rir[m_idx][s_idx], dtype=float)
            # light normalization to avoid zero/huge scale (keeps relative energy)
            peak = np.max(np.abs(rir)) + 1e-9
            if peak > 0:
                rir = rir / peak * 0.95
            row.append(rir)
        rirs.append(row)

    return rirs



if __name__ == "__main__":
    whamr_presets = {
        "room": {
            "L": (5.0, 10.0),
            "W": (5.0, 10.0),
            "H": (3.0, 4.0),
        },
        "T60": {
            "low":  (0.1, 0.3),
            "med":  (0.2, 0.6),
            "high": (0.4, 1.0),
        },
        "mic": {
            "center_jitter": 0.2,
            "height": (0.9, 1.8),
        },
        "sources": {
            "height": (0.9, 1.8),
            "dist_from_mic": (0.66, 2.0),
            "angle": (0, 2 * math.pi),
        }
    }
    rirs = generate_rirs(presets=whamr_presets, fs=16000, n_sources=2, n_mics=2)
    print("Generated RIR matrix: sources x mics =", len(rirs), "x", len(rirs[0]))