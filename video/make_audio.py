"""Synthesize every narration line with Supertonic (on-device TTS) and record durations.

Usage: python video/make_audio.py [voice] [speed]   (default F1, 1.0)
Writes video/audio/<key>.wav and video/audio/timings.json. Lines whose text, voice and speed
are unchanged are not re-synthesized.
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np, soundfile as sf
sys.path.insert(0, str(Path(__file__).parent))
from narration import SCENES

voice = sys.argv[1] if len(sys.argv) > 1 else "F1"
speed = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
OUT = Path(__file__).parent / "audio"; OUT.mkdir(exist_ok=True)
tim_path = OUT / "timings.json"
old = json.loads(tim_path.read_text()) if tim_path.exists() else {}
tts = style = None
tim = {}
for scene, lines in SCENES.items():
    for key, speech, caption in lines:
        h = hashlib.sha1(f"{voice}|{speed}|{speech}".encode()).hexdigest()[:12]
        wav = OUT / f"{key}.wav"
        if old.get(key, {}).get("hash") == h and wav.exists():
            tim[key] = {**old[key], "scene": scene, "caption": caption}; continue   # captions can change without new audio
        if tts is None:
            from supertonic import TTS
            tts = TTS(); style = tts.get_voice_style(voice)
        w, _ = tts.synthesize(speech, voice_style=style, lang="ko", total_steps=16, speed=speed)
        w = np.asarray(w).reshape(-1).astype(np.float32)
        sr = tts.sample_rate if hasattr(tts, "sample_rate") else 44100
        # trim leading/trailing silence, then normalise to -1 dBFS peak
        idx = np.where(np.abs(w) > 0.01)[0]
        w = w[max(0, idx[0] - int(0.03 * sr)): idx[-1] + int(0.08 * sr)]
        w = w / max(1e-6, np.abs(w).max()) * 0.89
        sf.write(wav, w, sr)
        tim[key] = {"scene": scene, "dur": round(len(w) / sr, 3), "hash": h, "caption": caption}
        print(f"{key:5s} {tim[key]['dur']:5.2f}s  {speech[:40]}")
tim_path.write_text(json.dumps(tim, ensure_ascii=False, indent=1))
tot = sum(v["dur"] for v in tim.values())
print(f"total narration {tot:.1f}s over {len(tim)} lines")
