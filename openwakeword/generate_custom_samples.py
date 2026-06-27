import os
import sys
import uuid
import random
import wave
import numpy as np
from scipy import signal
from pathlib import Path
from typing import List, Union, Tuple
import torch
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import math
import psutil

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
