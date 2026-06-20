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
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
app = Flask(__name__)

print("Loading Wav2Vec2...")
# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model  = bundle.get_model().to(device)
labels = bundle.get_labels()  # ('-', 'A', 'B', ..., '|')
print("Loaded Wav2Vec2.")

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

def find_corrupted_spans(clean_text, corrupted_text, char_timestamps):
    clean_chars   = [(i, c) for i, c in enumerate(clean_text)   if c.isalpha() or c == " "]
    corrupt_chars = [(i, c) for i, c in enumerate(corrupted_text) if c.isalpha() or c == " " or not c.isspace()]

    # First pass: find which alpha indices are corrupted
    corrupted_alpha_indices = set()
    alpha_idx = 0
    for idx, (_, clean_char) in enumerate(clean_chars):
        if clean_char == " ":
            continue
        if idx < len(corrupt_chars) and not corrupt_chars[idx][1].isalpha():
            corrupted_alpha_indices.add(alpha_idx)
        alpha_idx += 1

    # Second pass: build spans
    corrupted_spans = []
    ts_idx = 0

    for idx, (_, clean_char) in enumerate(clean_chars):
        if idx >= len(corrupt_chars):
            break
        corrupt_char = corrupt_chars[idx][1]

        if clean_char == " ":
            # Check if the alpha char before or after this space is corrupted
            prev_alpha = ts_idx - 1
            next_alpha = ts_idx
            adjacent_corrupted = (
                prev_alpha in corrupted_alpha_indices or
                next_alpha in corrupted_alpha_indices
            )
            if not corrupt_char.isspace() and adjacent_corrupted:
                if ts_idx > 0 and ts_idx < len(char_timestamps):
                    corrupted_spans.append((
                        char_timestamps[ts_idx - 1]["end"],
                        char_timestamps[ts_idx]["start"],
                    ))
        else:
            if not corrupt_char.isalpha() and ts_idx < len(char_timestamps):
                corrupted_spans.append((
                    char_timestamps[ts_idx]["start"],
                    char_timestamps[ts_idx]["end"],
                ))
            ts_idx += 1

    return merge_spans(corrupted_spans, gap_threshold=0.05)


def merge_spans(spans, gap_threshold=0.05):
    if not spans:
        return []
    spans = sorted(spans)
    merged = [spans[0]]
    for start, end in spans[1:]:
        if start - merged[-1][1] <= gap_threshold:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged

def apply_corruption_static(speech_audio, corrupted_spans, static, sr,
                             duck_gain=0, static_gain=0.85,
                             pre_mask_ms=60,   # bleed into preceding audio
                             post_mask_ms=80,  # tail after character ends
                             fade_ms=25):
    total = len(speech_audio)
    duck_envelope = np.ones(total)

    pre  = int(pre_mask_ms  * sr / 1000)
    post = int(post_mask_ms * sr / 1000)
    fade = int(fade_ms      * sr / 1000)

    for start_sec, end_sec in corrupted_spans:
        s = max(0, int(start_sec * sr) - pre)
        e = min(total, int(end_sec * sr) + post)

        duck_envelope[s:e] = duck_gain

        # Fade in (speech → ducked)
        fade_in_len = min(fade, (e - s) // 2)
        duck_envelope[s:s+fade_in_len] = np.linspace(1.0, duck_gain, fade_in_len)

        # Fade out (ducked → speech) — longer so word endings stay buried
        fade_out_len = min(fade * 2, (e - s) // 2)
        duck_envelope[e-fade_out_len:e] = np.linspace(duck_gain, 1.0, fade_out_len)

    ducked_speech = speech_audio * duck_envelope
    static_mask   = build_static_mask(corrupted_spans, static, total, sr,
                                       pre_ms=pre_mask_ms, post_ms=post_mask_ms,
                                       fade_ms=fade_ms, static_gain=static_gain)
    return ducked_speech + static_mask


def build_static_mask(corrupted_spans, static, total_samples, sr,
                       pre_ms=40, post_ms=40, fade_ms=25, static_gain=0.8):
    pre  = int(pre_ms  * sr / 1000)
    post = int(post_ms * sr / 1000)
    fade = int(fade_ms * sr / 1000)

    mask_envelope = np.zeros(total_samples)

    for start_sec, end_sec in corrupted_spans:
        s = max(0, int(start_sec * sr) - pre)
        e = min(total_samples, int(end_sec * sr) + post)

        mask_envelope[s:e] = 1.0

        fade_in_len  = min(fade, (e - s) // 2)
        fade_out_len = min(fade * 2, (e - s) // 2)
        mask_envelope[s:s+fade_in_len]       *= np.linspace(0, 1, fade_in_len)
        mask_envelope[e-fade_out_len:e]       *= np.linspace(1, 0, fade_out_len)

    static_full = np.tile(static, int(np.ceil(total_samples / len(static))))[:total_samples]
    return static_full * mask_envelope * static_gain

def audiosegment_to_torchaudio(seg):
    samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
    samples /= float(2 ** (seg.sample_width * 8 - 1))  # normalize to [-1, 1]
    
    waveform = torch.from_numpy(samples).unsqueeze(0)  # (1, T)
    
    # stereo: interleaved → (2, T)
    if seg.channels == 2:
        waveform = waveform.reshape(2, -1, seg.channels).squeeze(-1)  # cleaner split
        waveform = waveform.view(seg.channels, -1)
    
    return waveform, seg.frame_rate


cyrillic_to_latin = {
    'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E',
    'Ё': 'E', 'Ж': 'Z', 'З': 'Z', 'И': 'I', 'Й': 'I', 'К': 'K',
    'Л': 'L', 'М': 'M', 'Н': 'N', 'О': 'O', 'П': 'P', 'Р': 'R',
    'С': 'S', 'Т': 'T', 'У': 'U', 'Ф': 'F', 'Х': 'H', 'Ц': 'T',
    'Ч': 'C', 'Ш': 'S', 'Щ': 'S', 'Ъ': '', 'Ы': 'Y', 'Ь': '',
    'Э': 'E', 'Ю': 'I', 'Я': 'A',
}

cyrillic_regex = re.compile(r'[а-яёА-ЯЁ]')


def has_cyrillic(text):
    return bool(cyrillic_regex.search(text))


def transliterate_cyrillic(text):
    result = []
    for c in text:
        upper = c.upper()
        if c in cyrillic_to_latin:
            result.append(cyrillic_to_latin[c])
        elif upper in cyrillic_to_latin:
            latin = cyrillic_to_latin[upper]
            if c.islower() and latin:
                result.append(latin[0].lower() + latin[1:])
            else:
                result.append(latin)
        else:
            result.append(c)
    return ''.join(result)


def radio_effect(audio_path, raw_text, gibberish_text):
    base_audio = pydub.AudioSegment.from_file(audio_path, format="ogg")
    waveform, sr = audiosegment_to_torchaudio(base_audio)
    if sr != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)
    waveform = waveform.to("cuda")
    with torch.inference_mode():
        emissions, _ = model(waveform)
        log_probs = torch.log_softmax(emissions, dim=-1)

        label2idx = {c: i for i, c in enumerate(labels)}
        BLANK_IDX = 0
        def tokenize(text, label2idx):
            tokens = []
            for c in text.upper().replace(" ", "|"):
                idx = label2idx.get(c)
                if idx is None or idx == BLANK_IDX:
                    continue
                tokens.append(idx)
            return tokens

        if has_cyrillic(raw_text):
            raw_text = transliterate_cyrillic(raw_text)
        if has_cyrillic(gibberish_text):
            gibberish_text = transliterate_cyrillic(gibberish_text)

        token_ids = tokenize(raw_text, label2idx)
        targets   = torch.tensor(token_ids, dtype=torch.int32).unsqueeze(0).to(device)
        alignments, scores = torchaudio.functional.forced_align(
            log_probs, targets, blank=0
        )
        char_spans = torchaudio.functional.merge_tokens(alignments[0], scores[0])
        FRAME_STRIDE = 320          # wav2vec2-base: 320 samples per frame @ 16kHz
        SAMPLE_RATE  = bundle.sample_rate   # 16000

        def frames_to_sec(f):
            return (f * FRAME_STRIDE) / SAMPLE_RATE

        char_timestamps = []
        for span in char_spans:
            char = labels[span.token]
            if char == "|":
                continue
            char_timestamps.append({
                "char":  char,
                "start": frames_to_sec(span.start),
                "end":   frames_to_sec(span.end),
                "score": span.score,
            })


        click_on = load_and_match("mic_click_on.wav", 48000)
        click_off = load_and_match("mic_click_off.wav", 48000)
        static = load_and_match("diffstatic.wav", 48000)
        samples, numpy_sr = audiosegment_to_numpy(base_audio)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        speech = librosa.resample(samples, orig_sr=numpy_sr, target_sr=8000)
        speech = librosa.resample(speech, orig_sr=8000, target_sr=48000)
        speech = normalize(speech, 0.5)
        speech = bandpass(speech, numpy_sr)
        speech = saturate(speech, 2)
        speech = normalize(speech, 0.7)

        # Find which time windows are "corrupted"
        corrupted_spans = find_corrupted_spans(raw_text, gibberish_text, char_timestamps)

        # Duck speech + inject static bursts
        speech = apply_corruption_static(speech, corrupted_spans, static, numpy_sr,
                                        duck_gain=0, static_gain=0.85)

        static = static[: len(speech)]
        speech = speech + static * 0.2
        output = np.concatenate([click_on, speech, static[: int(numpy_sr * 0.1)], click_off])

        output = normalize(output, 0.9)

        return output

def mask_string(text):
    words = text.split()
    length = len(words)
    percent = random.uniform(0.35, 0.65)
    num_to_replace = int(length * percent)
    indices = random.sample(range(length), num_to_replace)
    for i in indices:
        words[i] = '#' * len(words[i])
    return ' '.join(words)

@app.route("/radio-gibberish")
def radio_handler():
    identifier = request.json.get("identifier", "")
    folder = request.json.get("folder", "")
    raw_text = request.json.get("raw_text", "")
    gibberish_text = request.json.get("gibberish_text", "")
    start_time = time.time()
    logger.debug(f"ID: {identifier} | Applying radio gibberish effect to generated audio...")
    timeout = 10
    start = time.time()

    while not os.path.exists("./cache/" + folder + "/" + identifier + ".radio"):
        if time.time() - start > timeout:
            logger.debug(f"ID: {identifier} | Timed out waiting for the input file!")
            abort(408)
        time.sleep(0.05)
    processed = radio_effect(io.BytesIO(torch.load("./cache/" + folder + "/" + identifier + ".radio", weights_only=False)), raw_text, mask_string(raw_text))
    export_audio = numpy_to_audiosegment(processed, 48000)
    output_bytes = io.BytesIO()
    export_audio.export(output_bytes, format="ogg")
    result = send_file(
        io.BytesIO(output_bytes.getvalue()), mimetype="audio/ogg"
    )
    result.headers["audio-length"] = export_audio.duration_seconds
    logger.info(f"ID: {identifier} | Radio gibberish effect generation time: {time.time() - start_time:.4f}s")
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
    print("Serving Radio Gibberish Effects on :5006")
    serve(app, host="0.0.0.0", port=5006, backlog=32, channel_timeout=8)