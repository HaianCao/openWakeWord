import os
import sys
import uuid
import random
import wave
import numpy as np
from scipy import signal
from pathlib import Path
from typing import List, Union, Tuple, Optional
import torch
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import math
import psutil

# ─────────────────────────────────────────────────────────────────────────────
# Piper TTS backend (multiprocess, existing code — DO NOT MODIFY)
# ─────────────────────────────────────────────────────────────────────────────

# Worker function for multiprocessing - must be at module level for pickling
def _generate_samples_worker(args: Tuple) -> int:
    """
    Worker function to generate samples for a single model in a separate process.
    
    Args:
        args: Tuple of (model_path_str, texts, target_samples, output_dir, 
                       length_scales, noise_scales, noise_scale_ws, use_cuda)
    
    Returns:
        Number of samples successfully generated
    """
    (model_path_str, texts, target_samples, output_dir, 
     length_scales, noise_scales, noise_scale_ws, use_cuda) = args
    
    try:
        from piper import PiperVoice, SynthesisConfig
    except ImportError as e:
        print(f"Failed to import Piper: {e}")
        return 0
    
    model_path = Path(model_path_str)
    if not model_path.exists():
        print(f"Model not found at {model_path_str}")
        return 0
    
    try:
        voice = PiperVoice.load(model_path, use_cuda=use_cuda)
    except Exception as e:
        if use_cuda:
            try:
                voice = PiperVoice.load(model_path, use_cuda=False)
            except Exception as e2:
                print(f"Failed to load model {model_path_str}: {e2}")
                return 0
        else:
            print(f"Failed to load model {model_path_str}: {e}")
            return 0
    
    generated = 0
    TARGET_SAMPLE_RATE = 16000
    
    for _ in range(target_samples):
        t = random.choice(texts)
        l_scale = random.choice(length_scales)
        n_scale = random.choice(noise_scales)
        n_w_scale = random.choice(noise_scale_ws)
        
        syn_config = SynthesisConfig(
            length_scale=l_scale,
            noise_scale=n_scale,
            noise_w_scale=n_w_scale
        )
        
        unique_id = str(uuid.uuid4())
        wav_path = os.path.join(output_dir, f"{unique_id}.wav")
        
        try:
            audio_chunks = list(voice.synthesize(t, syn_config))
            if not audio_chunks:
                continue
            
            wav_file = wave.open(wav_path, "wb")
            with wav_file:
                wav_file.setframerate(TARGET_SAMPLE_RATE)
                wav_file.setsampwidth(audio_chunks[0].sample_width)
                wav_file.setnchannels(audio_chunks[0].sample_channels)
                
                for i_chunk, audio_chunk in enumerate(audio_chunks):
                    audio = audio_chunk.audio_float_array
                    orig_sr = audio_chunk.sample_rate
                    
                    if orig_sr != TARGET_SAMPLE_RATE:
                        num_samples = int(round(len(audio) * float(TARGET_SAMPLE_RATE) / orig_sr))
                        audio = signal.resample(audio, num_samples)
                    
                    if i_chunk > 0:
                        silence_int16_bytes = bytes(int(TARGET_SAMPLE_RATE * 0.0 * 2))
                        wav_file.writeframes(silence_int16_bytes)
                    
                    _MAX_WAV_VALUE = 32767.0
                    audio_int16 = np.clip(audio * _MAX_WAV_VALUE, -_MAX_WAV_VALUE, _MAX_WAV_VALUE).astype(np.int16)
                    wav_file.writeframes(audio_int16.tobytes())
            generated += 1
        except Exception as e:
            if os.path.exists(wav_path):
                os.remove(wav_path)
    
    return generated


def generate_multi_model_samples(
    text: Union[str, List[str]],
    max_samples: int,
    piper_models: List[str],
    piper_src_path: str,
    output_dir: str,
    length_scales: List[float] = [0.75, 1.0, 1.25],
    noise_scales: List[float] = [0.98],
    noise_scale_ws: List[float] = [0.98],
    num_workers: int = None
):
    """
    Generate synthetic TTS samples using multiple Piper ONNX models with multiprocessing.
    
    Args:
        text: Target text(s) to synthesize
        max_samples: Total number of samples to generate
        piper_models: List of paths to Piper ONNX models
        piper_src_path: Path to piper source (kept for compatibility)
        output_dir: Output directory for generated WAV files
        length_scales: List of length scale values for variation
        noise_scales: List of noise scale values for variation
        noise_scale_ws: List of noise width scale values for variation
        num_workers: Number of worker processes (default: min(CPU count, len(piper_models) * 2))
    """
    if isinstance(text, str):
        texts = [text]
    else:
        texts = text

    os.makedirs(output_dir, exist_ok=True)
    
    # Filter valid models
    valid_models = [m for m in piper_models if Path(m).exists()]
    if not valid_models:
        print("Error: No valid Piper models found!")
        return
    
    # Auto-detect GPU for Piper TTS
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("GPU detected - using CUDA for Piper TTS")
    else:
        print("No GPU detected - using CPU for Piper TTS")
    
    # Determine number of workers
    if num_workers is None:
        # Default: 2 workers per model, capped at 8 total to avoid memory issues
        # Each Piper model load uses ~1-2GB RAM
        num_workers = min(8, len(valid_models) * 2)
    # Respect user config, but cap at CPU count
    num_workers = max(1, min(num_workers, cpu_count()))
    
    # Memory check
    available_gb = psutil.virtual_memory().available / (1024**3)
    estimated_memory_gb = num_workers * 1.5
    if estimated_memory_gb > available_gb * 0.8:
        recommended = max(1, int(available_gb * 0.8 / 1.5))
        print(f"⚠️  Warning: {num_workers} workers may need ~{estimated_memory_gb:.1f}GB RAM")
        print(f"   Available: {available_gb:.1f}GB. Recommended: {recommended} workers")
        print(f"   Auto-limiting to {recommended} workers...")
        num_workers = recommended
    
    # Final cap at CPU count (Colab free = 2, Pro = more)
    num_workers = min(num_workers, cpu_count())
    
    print(f"Generating {max_samples} samples across {len(valid_models)} models using {num_workers} workers (CPU cores: {cpu_count()})...")
    
    # Distribute samples across models
    samples_per_model = max_samples // len(valid_models)
    remaining_samples = max_samples % len(valid_models)
    
    # Prepare work items: each model gets its own worker(s)
    work_items = []
    for i, model_path in enumerate(valid_models):
        target_samples = samples_per_model + (1 if i < remaining_samples else 0)
        if target_samples > 0:
            # Limit workers per model to avoid duplicate model loading in memory
            max_workers_per_model = min(4, num_workers // len(valid_models))
            workers_for_model = max(1, min(max_workers_per_model, num_workers // len(valid_models)))
            samples_per_worker = target_samples // workers_for_model
            extra = target_samples % workers_for_model
            
            for w in range(workers_for_model):
                worker_samples = samples_per_worker + (1 if w < extra else 0)
                if worker_samples > 0:
                    work_items.append((
                        model_path, texts, worker_samples, output_dir,
                        length_scales, noise_scales, noise_scale_ws, use_cuda
                    ))
    
    # If we have fewer work items than workers, adjust
    actual_workers = min(num_workers, len(work_items))
    
    if actual_workers < num_workers:
        print(f"Note: Adjusted to {actual_workers} workers (limited by samples per model)")
    
    # Run multiprocessing with single progress bar in main process
    total_generated = 0
    with Pool(processes=actual_workers) as pool:
        # Use imap_unordered for progress tracking as results complete
        with tqdm(total=max_samples, desc="Generating samples", unit="sample") as pbar:
            for result in pool.imap_unordered(_generate_samples_worker, work_items):
                total_generated += result
                pbar.update(result)
    
    print(f"Successfully generated {total_generated} samples in {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# VieNeu-TTS v3 Turbo backend (GPU-accelerated, single process)
# ─────────────────────────────────────────────────────────────────────────────

def generate_samples_vieneu(
    texts: List[str],
    max_samples: int,
    output_dir: str,
    vieneu_repo: str = "pnnbao-ump/VieNeu-TTS-v3-Turbo",
    vieneu_voices: Optional[List[str]] = None,
    vieneu_ref_audio: Optional[str] = None,
    vieneu_emotion: str = "natural",
    batch_size: int = 4,
    hf_token: Optional[str] = None,
    target_sr: int = 16000,
):
    """
    Generate synthetic TTS samples using VieNeu-TTS v3 Turbo.

    Runs entirely in the CURRENT process (no subprocess / multiprocessing).
    The model is loaded once on GPU (if available, else CPU via ONNX), then
    clips are synthesised while **randomly cycling through ALL available preset
    voices** to maximise speaker diversity — mirroring what Piper does with
    multiple model files.

    Args:
        texts:           Pool of texts to sample from randomly.
        max_samples:     Total number of WAV clips to generate.
        output_dir:      Directory to write generated WAV files.
        vieneu_repo:     HuggingFace repo ID for the VieNeu v3-Turbo backbone.
        vieneu_voices:   Explicit list of preset voice names to cycle through
                         (e.g. ["Tuyen", "Thu", "Nam"]).
                         • None / [] → use ALL available preset voices (auto).
                         • ["default"] → use the model default voice only.
        vieneu_ref_audio: Path to a reference WAV for zero-shot voice cloning.
                         When provided, OVERRIDES vieneu_voices for every clip.
        vieneu_emotion:  Emotion style ("natural", "happy", …).
        batch_size:      Clips per inner loop iteration. On a T4 GPU, 4–8 is
                         safe; increase for higher throughput.
        hf_token:        HuggingFace token for private/gated model access.
        target_sr:       Output sample rate (must be 16000 for openWakeWord).
    """
    try:
        from scipy.signal import resample_poly
        from math import gcd
        import scipy.io.wavfile as wav_io
        from vieneu import Vieneu
    except ImportError as e:
        print(f"[VieNeu] Cannot import required libraries: {e}")
        print("[VieNeu] Install with: pip install vieneu scipy")
        return

    os.makedirs(output_dir, exist_ok=True)

    # ── Determine device ──────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[VieNeu] Loading v3-Turbo on {device.upper()} …")

    # ── Load model (once) ─────────────────────────────────────────────────────
    try:
        init_kwargs: dict = dict(backbone_repo=vieneu_repo, device=device)
        if hf_token:
            init_kwargs["hf_token"] = hf_token
        tts = Vieneu(mode="v3turbo", **init_kwargs)
    except Exception as e:
        print(f"[VieNeu] Failed to load model: {e}")
        return

    # ── Build voice pool ──────────────────────────────────────────────────────
    # Priority: ref_audio (zero-shot clone) > explicit list > auto-all presets
    use_ref_audio = bool(vieneu_ref_audio and Path(vieneu_ref_audio).exists())

    if use_ref_audio:
        voice_pool = [{"ref_audio": vieneu_ref_audio}]
        print(f"[VieNeu] Zero-shot cloning from: {vieneu_ref_audio} (1 voice)")
    else:
        if vieneu_voices:
            # User supplied explicit list — validate names
            all_preset_names = {name for _, name in tts.list_preset_voices()}
            invalid = [v for v in vieneu_voices if v not in all_preset_names and v != "default"]
            if invalid:
                print(f"[VieNeu] Warning: unknown voice(s) {invalid}. "
                      f"Available: {sorted(all_preset_names)}")
            voice_names = [v for v in vieneu_voices if v != "default" and v in all_preset_names]
            if not voice_names:
                voice_names = []  # fall to default below
        else:
            # Auto-discover every preset the model ships with
            voice_names = [name for _, name in tts.list_preset_voices()]

        if voice_names:
            voice_pool = [{"voice": name} for name in voice_names]
        else:
            voice_pool = [{}]  # model built-in default (no kwarg needed)
            voice_names = ["<default>"]

        print(f"[VieNeu] Voice pool ({len(voice_pool)} voices): {voice_names}")

    print(f"[VieNeu] batch_size={batch_size} | emotion={vieneu_emotion} | "
          f"target_sr={target_sr}")

    # ── Resample helper ───────────────────────────────────────────────────────
    def _resample(audio: np.ndarray, orig_sr: int, tgt_sr: int) -> np.ndarray:
        if orig_sr == tgt_sr:
            return audio
        g = gcd(orig_sr, tgt_sr)
        return resample_poly(audio, tgt_sr // g, orig_sr // g).astype(np.float32)

    # ── Generate loop ─────────────────────────────────────────────────────────
    total_generated = 0
    VMAX = 32767.0

    with tqdm(total=max_samples, desc="[VieNeu] Generating", unit="clip") as pbar:
        while total_generated < max_samples:
            current_batch = min(batch_size, max_samples - total_generated)

            for _ in range(current_batch):
                if total_generated >= max_samples:
                    break

                t = random.choice(texts)
                # Randomly pick a voice from the pool for diversity
                voice_kwargs = random.choice(voice_pool)
                wav_path = os.path.join(output_dir, f"{uuid.uuid4()}.wav")

                try:
                    audio = tts.infer(
                        text=t,
                        emotion=vieneu_emotion,
                        apply_watermark=False,
                        **voice_kwargs,
                    )
                    if audio is None or len(audio) == 0:
                        continue

                    # VieNeu v3 outputs 48 kHz — resample to 16 kHz
                    audio_16k = _resample(audio, tts.sample_rate, target_sr)
                    audio_int16 = np.clip(
                        audio_16k * VMAX, -VMAX, VMAX
                    ).astype(np.int16)
                    wav_io.write(wav_path, target_sr, audio_int16)

                    total_generated += 1
                    pbar.update(1)
                    pbar.set_postfix(voice=str(voice_kwargs.get("voice", "clone")))
                except Exception as e:
                    if os.path.exists(wav_path):
                        os.remove(wav_path)
                    print(f"\n[VieNeu] Warning: failed '{t[:40]}…' "
                          f"(voice={voice_kwargs}): {e}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    try:
        tts.close()
    except Exception:
        pass

    print(f"[VieNeu] Done — {total_generated} clips saved to {output_dir}")
