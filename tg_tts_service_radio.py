import gc
import io
import os
import random
import re
import subprocess
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("tts.api")

import librosa
import numpy as np
import pydub
import pysbd
import requests
import soundfile as sf
from blake3 import blake3
from flask import Flask, abort, make_response, request, send_file, jsonify
from pydub import AudioSegment
from pydub.silence import detect_leading_silence
from stftpitchshift import StftPitchShift
import torch
import torchaudio
app = Flask(__name__)

def audiosegment_to_numpy(seg):
    samples = np.array(seg.get_array_of_samples())

    if seg.channels == 2:
        samples = samples.reshape((-1, 2))

    samples = samples.astype(np.float32) / (1 << (8 * seg.sample_width - 1))

    return samples, seg.frame_rate


def numpy_to_audiosegment(samples, sr, sample_width=2, channels=1):
    samples_int16 = (samples * 32767).astype(np.int16)

    return AudioSegment(
        samples_int16.tobytes(),
        frame_rate=sr,
        sample_width=sample_width,
        channels=channels,
    )


from scipy.signal import butter, lfilter


def bandpass(x, sr, low=300, high=3000, order=4):
    nyq = sr * 0.5
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return lfilter(b, a, x)


def compress(x, threshold=0.2, ratio=4):
    y = x.copy()
    mask = np.abs(y) > threshold
    y[mask] = np.sign(y[mask]) * (threshold + (np.abs(y[mask]) - threshold) / ratio)
    return y


def saturate(x, drive=2.5):
    return np.tanh(drive * x)


def add_radio_noise(x, level=0.004):
    noise = np.random.normal(0, level, len(x))
    return x + noise


def am_modulate(x, sr, depth=0.08, freq=60):
    t = np.arange(len(x)) / sr
    return x * (1 + depth * np.sin(2 * np.pi * freq * t))


def squelch_tail(sr, length=0.15):
    n = int(sr * length)
    noise = np.random.normal(0, 0.02, n)
    fade = np.linspace(1, 0, n)
    return noise * fade


def normalize(x, peak=0.8):
    m = np.max(np.abs(x))
    if m > 0:
        x = x / m * peak
    return x


def ensure_mono(x):
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x


def load_and_match(path, target_sr):
    audio, sr = sf.read(path)
    audio = ensure_mono(audio)

    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    return audio

def numpy_to_torch_audio(audio_np):
    # Ensure float32
    if audio_np.dtype != np.float32:
        if np.issubdtype(audio_np.dtype, np.integer):
            audio_np = audio_np.astype(np.float32) / np.iinfo(audio_np.dtype).max
        else:
            audio_np = audio_np.astype(np.float32)

    # Shape to (channels, samples)
    if audio_np.ndim == 1:
        audio_np = audio_np[np.newaxis, :]
    else:
        audio_np = audio_np.T

    return torch.from_numpy(audio_np)

def audiosegment_to_torchaudio(seg):
    samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
    samples /= float(2 ** (seg.sample_width * 8 - 1))  # normalize to [-1, 1]
    
    waveform = torch.from_numpy(samples).unsqueeze(0)  # (1, T)
    
    # stereo: interleaved → (2, T)
    if seg.channels == 2:
        waveform = waveform.reshape(2, -1, seg.channels).squeeze(-1)  # cleaner split
        waveform = waveform.view(seg.channels, -1)
    
    return waveform, seg.frame_rate

def radio_effect(audio_path, raw_text, gibberish_text):
    base_audio = pydub.AudioSegment.from_file(audio_path, format="ogg")
    samples, numpy_sr = audiosegment_to_numpy(base_audio)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    click_on = load_and_match("mic_click_on.wav", numpy_sr)
    click_off = load_and_match("mic_click_off.wav", numpy_sr)
    static = load_and_match("diffstatic.wav", numpy_sr)
    speech = librosa.resample(samples, orig_sr=numpy_sr, target_sr=8000)
    speech = librosa.resample(speech, orig_sr=8000, target_sr=48000)
    speech = normalize(speech, 0.5)
    speech = bandpass(speech, numpy_sr)
    speech = saturate(speech, 2)
    speech = normalize(speech, 0.7)
    static = static[: len(speech)]
    speech = speech + static * 0.2
    output = np.concatenate([click_on, speech, static[: int(numpy_sr * 0.1)], click_off])

    output = normalize(output, 0.9)

    return output


@app.route("/radio")
def radio_handler():
    identifier = request.json.get("identifier", "")
    folder = request.json.get("folder", "")
    raw_text = request.json.get("raw_text", "")
    gibberish_text = request.json.get("gibberish_text", "")
    start_time = time.time()
    logger.debug(f"ID: {identifier} | Applying radio effect to generated audio...")
    timeout = 10
    start = time.time()

    while not os.path.exists("./cache/" + folder + "/" + identifier + ".radio"):
        if time.time() - start > timeout:
            logger.debug(f"ID: {identifier} | Timed out waiting for the input file!")
            abort(408)
        time.sleep(0.05)
    processed = radio_effect(io.BytesIO(torch.load("./cache/" + folder + "/" + identifier + ".radio", weights_only=False)), raw_text, gibberish_text)
    export_audio = numpy_to_audiosegment(processed, 48000)
    output_bytes = io.BytesIO()
    export_audio.export(output_bytes, format="ogg")
    result = send_file(
        io.BytesIO(output_bytes.getvalue()), mimetype="audio/ogg"
    )
    result.headers["audio-length"] = export_audio.duration_seconds
    logger.info(f"ID: {identifier} | Radio effect generation time: {time.time() - start_time:.4f}s")
    return result

if __name__ == "__main__":
    from waitress import serve
    print("Warming up...")
    noise = radio_effect(
        "./radio_warmup.ogg",
        "The quick brown fox jumps over the lazy dog.",
        "T&e q#$ck $%#!@ fox ju### ov2r 32e $$zy *(g."
    ),
    del noise
    print("Serving Radio Effects on :5005")
    serve(app, host="0.0.0.0", port=5005, backlog=32, channel_timeout=8)