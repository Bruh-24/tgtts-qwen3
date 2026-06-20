import asyncio
import io
import json
import logging
import os
import time
from typing import *

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from huggingface_hub import snapshot_download
from nanovllm_voxcpm import VoxCPM
from nanovllm_voxcpm.models.voxcpm2.config import LoRAConfig
from nanovllm_voxcpm.models.voxcpm2.server import AsyncVoxCPM2ServerPool
from pydub import AudioSegment, effects
from tqdm.asyncio import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("tts.service")

speaker_latents_dir = "./speaker_latents"


def load_voice_mapping(mapping_path: str) -> dict[str, dict[str, str]]:
    with open(mapping_path, "r", encoding="utf-8") as file:
        raw = json.load(file)

    if not isinstance(raw, dict):
        return {}

    result: dict[str, dict[str, str]] = {}
    for voice_name, entry in raw.items():
        if isinstance(entry, str):
            result[voice_name] = {"latent": entry}
        elif isinstance(entry, dict) and isinstance(entry.get("latent"), str):
            result[voice_name] = entry
    return result


def build_voice_list_response(voice_mapping: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    voices = []
    for name, entry in voice_mapping.items():
        voices.append({"name": name, **entry})
    voices.sort(key=lambda e: e["name"])
    return voices


# --- Model Infrastructure ---


class TGAsyncPool(AsyncVoxCPM2ServerPool):
    async def generate(
        self,
        target_text: str,
        prompt_latents: bytes | None = None,
        prompt_text: str = "",
        prompt_id: str | None = None,
        max_generate_length: int = 2000,
        temperature: float = 1.0,
        cfg_value: float = 2.0,
        ref_audio_latents: bytes | None = None,
    ):
        if prompt_id is not None:
            if prompt_id not in self._prompt_pool:
                # Note: Keeping the synchronous load here as it's usually fast,
                # but in high-scale it could be moved to an async executor.
                latent_path = os.path.join(speaker_latents_dir, f"{prompt_id}.speaker_latent")
                self._prompt_pool[prompt_id] = torch.load(
                    latent_path, weights_only=False
                )

            prompt_info = self._prompt_pool[prompt_id]
            prompt_latents = None
            prompt_text = ""
            ref_audio_latents = prompt_info["latents"]

        min_load_server_idx = np.argmin(self.servers_load)
        self.servers_load[min_load_server_idx] += 1
        server = self.servers[min_load_server_idx]
        try:
            async for data in server.generate(
                target_text,
                prompt_latents,
                prompt_text,
                max_generate_length,
                temperature,
                cfg_value,
                ref_audio_latents,
            ):
                yield data
        finally:
            self.servers_load[min_load_server_idx] -= 1


class VoxCPM2_tg_Async:
    @staticmethod
    async def from_pretrained(
        model: str,
        inference_timesteps: int = 6,
        max_num_batched_tokens: int = 16384,
        max_num_seqs: int = 512,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.7,
        enforce_eager: bool = False,
        devices: List[int] = [],
        lora_config: Any = None,
        **kwargs,
    ):
        if "~" in model:
            model_path = os.path.expanduser(model)
        else:
            model_path = (
                model if os.path.isdir(model) else snapshot_download(repo_id=model)
            )

        if len(devices) == 0:
            devices = [0]

        pool = TGAsyncPool(
            model_path=model_path,
            inference_timesteps=inference_timesteps,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            devices=devices,
            lora_config=lora_config,
            **kwargs,
        )
        await pool.wait_for_ready()
        return pool


# --- App Setup ---

app = FastAPI()
model = None  # Global model instance
voice_name_mapping = {}
use_voice_name_mapping = False


# Audio Utility Functions
def normalize_to_target(seg, target_dbfs=-20.0):
    change = target_dbfs - seg.dBFS
    return seg.apply_gain(change)


def cap_loudness(seg, max_dbfs=-1.0):
    if seg.max_dBFS > max_dbfs:
        return seg.apply_gain(max_dbfs - seg.max_dBFS)
    return seg


@app.on_event("startup")
async def startup_event():
    global model, voice_name_mapping, use_voice_name_mapping

    # Load voice mappings
    voice_name_mapping = load_voice_mapping("./voice_mapping.json")
    use_voice_name_mapping = len(voice_name_mapping) > 0
    if use_voice_name_mapping:
        logger.info("Loaded %d voice mappings", len(voice_name_mapping))

    logger.info("Loading VoxCPM2 into memory...")
    model = await VoxCPM2_tg_Async.from_pretrained(
        "openbmb/VoxCPM2",
        devices=[0],
        max_num_batched_tokens=8192,
        max_num_seqs=16,
        gpu_memory_utilization=0.7,
    )
    logger.info("Model Loaded.")


# --- API Endpoints ---


@app.get("/generate-tts")  # Support both for compatibility
async def text_to_speech(request: Request):
    # Parse json body
    body = await request.json()

    text = body.get("text", "")
    voice = body.get("voice", "")
    request_start_time = time.time()
    logger.debug(f"Endpoint: /generate-tts | Voice: {voice} | Text: {text[:50]}...")

    # Check for actual content
    if not any(character.isalnum() for character in text):
        stub = AudioSegment.empty().set_frame_rate(48000)
        buffer = io.BytesIO()
        stub.export(buffer, format="wav")
        return Response(content=buffer.getvalue(), media_type="audio/wav")

    # Sanitize text
    if text and text[-1] not in [".", "?", "!"]:
        text += ". "
    if text and text[0].isalpha() and not text[0].isupper():
        text = text[0].upper() + text[1:]

    # Inference (No Lock needed - Nano-vLLM handles it)
    gen_start = time.time()
    voice_entry = voice_name_mapping.get(voice)
    max_generate_length = 256
    chunks = []
    async for chunk in model.generate(
        target_text=text,
        prompt_id=voice_entry["latent"] if voice_entry else None,
        max_generate_length=max_generate_length,
    ):
        chunks.append(chunk)

    wav_data = np.concatenate(chunks, axis=0)
    tts_duration = time.time() - gen_start

    tts_databytes = io.BytesIO()
    sf.write(tts_databytes, wav_data, 48000, format="wav")
    gen_end = time.time()
    tts_duration = gen_end - gen_start
    normalise_start = time.time()
    rawsound = AudioSegment.from_file(io.BytesIO(tts_databytes.getvalue()), "wav")
    temp_databytes = io.BytesIO()
    normalizedsound = normalize_to_target(rawsound, -25)
    normalizedsound = cap_loudness(normalizedsound, max_dbfs=-5)
    normalizedsound = effects.normalize(rawsound)
    normalizedsound.export(temp_databytes, format="wav")
    normalised_end = time.time()
    normalize_duration = normalised_end - normalise_start
    audio_duration = len(normalizedsound) / 1000
    total_time = time.time() - request_start_time
    logmsg = f"Slow request detected. Total time: {total_time:.4f}s | Voice: {voice} | Text: {text} | Audio Duration: {audio_duration:.2f}s | TTS Time: {tts_duration:.4f}s | Normalization Time: {normalize_duration:.4f}s"
    if total_time > 4.0:
        logger.warning(logmsg)

    else:
        logger.info(logmsg)
    
       # torch.cuda.empty_cache()
    return Response(content=temp_databytes.getvalue(), media_type="audio/wav")


@app.get("/tts-voices")
async def voices_list():
    if use_voice_name_mapping:
        return JSONResponse(content=build_voice_list_response(voice_name_mapping))
    return JSONResponse(content=[])


@app.get("/health-check")
async def tts_health_check():
    return Response(content="OK: 1", status_code=200)


@app.get("/toggle-logging")
async def toggle_logging(level: str = None):
    if level:
        try:
            logger.setLevel(level.upper())
        except (ValueError, TypeError):
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": f"Invalid level: {level}"},
            )
    else:
        new_level = (
            logging.DEBUG
            if logger.getEffectiveLevel() == logging.INFO
            else logging.INFO
        )
        logger.setLevel(new_level)

    return JSONResponse(
        {
            "status": "success",
            "new_level": logging.getLevelName(logger.getEffectiveLevel()),
        }
    )


if __name__ == "__main__":
    import uvicorn

    # Use uvicorn instead of Waitress
    uvicorn.run(app, host="0.0.0.0", port=5003, log_level="info")
