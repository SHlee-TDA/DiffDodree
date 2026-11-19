"""Render a melody to WAV with the web demo's reed-like voice (site/js/audio.js), offline.

A saw plus a slightly detuned square, a vibrato that fades in, a resonant low-pass at
min(3200 Hz, 5 f) and a short attack/release; additive synthesis keeps it alias-free.
This is a synthesized timbre, not a 피리 recording.
"""
import numpy as np, soundfile as sf
from scipy.signal import lfilter

SR = 44100


def _lowpass(x, fc, q):
    w0 = 2 * np.pi * fc / SR; a = np.sin(w0) / (2 * q); c = np.cos(w0)
    b = np.array([(1 - c) / 2, 1 - c, (1 - c) / 2]); den = np.array([1 + a, -2 * c, 1 - a])
    return lfilter(b / den[0], den / den[0], x)


def note(midi, dur):
    f = 440 * 2 ** ((midi - 69) / 12)
    n = int(dur * SR); t = np.arange(n) / SR
    depth = f * 0.012 * np.clip(t / min(0.35, dur * 0.6), 0, 1)
    inst = f + depth * np.sin(2 * np.pi * 5.2 * t)
    ph = 2 * np.pi * np.cumsum(inst) / SR
    y = np.zeros(n)
    kmax = int(min(8000, SR / 2 - 200) // (f * 1.003))
    for k in range(1, kmax + 1):
        y += np.sin(k * ph) / k                       # sawtooth
        if k % 2: y += 0.35 * (4 / np.pi) * np.sin(k * ph * 1.003) / k  # detuned square
    y = _lowpass(y * 0.5, min(3200, f * 5), 2.5)
    env = np.ones(n); a = int(0.035 * SR); r = int(min(0.05, dur * 0.3) * SR)
    env[:a] = np.linspace(0, 1, a); env[n - r:] = np.linspace(1, 0, r)
    return y * env


def render(phrases, bpm, path, tail=1.2):
    spb = 60 / bpm
    total = sum(d for ev in phrases for _, d in ev) * spb
    out = np.zeros(int((total + tail) * SR))
    t = 0.0
    for ev in phrases:
        for p, d in ev:
            dur = d * spb
            if p is not None:
                y = note(p, max(dur - 0.015, 0.03)); i = int(t * SR)
                out[i:i + len(y)] += y
            t += dur
    # a small room: exponentially decaying noise as the impulse response
    rng = np.random.default_rng(1)
    ir = rng.standard_normal(int(0.9 * SR)) * np.exp(-np.arange(int(0.9 * SR)) / (0.22 * SR))
    wet = np.convolve(out, ir)[: len(out)]
    y = out + 0.12 * wet / (np.abs(wet).max() + 1e-9) * np.abs(out).max()
    y = y / (np.abs(y).max() + 1e-9) * 0.8
    sf.write(path, y.astype(np.float32), SR)
    return total
