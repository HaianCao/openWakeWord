import os
import sys
import uuid
import random
import wave
from pathlib import Path
from typing import List, Union

def generate_multi_model_samples(
    text: Union[str, List[str]],
    max_samples: int,
    piper_models: List[str],
    piper_src_path: str,
    output_dir: str,
    length_scales: List[float] = [0.75, 1.0, 1.25],
    noise_scales: List[float] = [0.98],
    noise_scale_ws: List[float] = [0.98]
):
    """
    Generate synthetic TTS samples using multiple Piper ONNX models.
    """
    # Import Piper dynamically
    if piper_src_path not in sys.path:
        sys.path.insert(0, os.path.abspath(piper_src_path))
    
    try:
        from piper import PiperVoice, SynthesisConfig
    except ImportError as e:
        raise ImportError(f"Failed to import Piper from {piper_src_path}. Please check the path.") from e

    if isinstance(text, str):
        texts = [text]
    else:
        texts = text

    os.makedirs(output_dir, exist_ok=True)
    
    samples_per_model = max_samples // len(piper_models)
    remaining_samples = max_samples % len(piper_models)

    total_generated = 0

    print(f"Generating {max_samples} samples across {len(piper_models)} models...")

    for i, model_path_str in enumerate(piper_models):
        model_path = Path(model_path_str)
        if not model_path.exists():
            print(f"Warning: Piper model not found at {model_path_str}. Skipping...")
            continue
            
        print(f"Loading model: {model_path_str}")
        try:
            # use_cuda=False by default to avoid issues, can be changed if GPU is supported
            voice = PiperVoice.load(model_path, use_cuda=False)
        except Exception as e:
            print(f"Failed to load model {model_path_str}: {e}")
            continue
            
        # Determine how many samples to generate for this model
        target_samples = samples_per_model + (1 if i < remaining_samples else 0)
        
        for _ in range(target_samples):
            # Randomly select parameters
            t = random.choice(texts)
            l_scale = random.choice(length_scales)
            n_scale = random.choice(noise_scales)
            n_w_scale = random.choice(noise_scale_ws)
            
            syn_config = SynthesisConfig(
                length_scale=l_scale,
                noise_scale=n_scale,
                noise_w_scale=n_w_scale
            )
            
            # Generate unique filename
            unique_id = str(uuid.uuid4())
            wav_path = os.path.join(output_dir, f"{unique_id}.wav")
            
            # Synthesize and write wav
            try:
                wav_file = wave.open(wav_path, "wb")
                with wav_file:
                    wav_params_set = False
                    for i_chunk, audio_chunk in enumerate(voice.synthesize(t, syn_config)):
                        if not wav_params_set:
                            wav_file.setframerate(audio_chunk.sample_rate)
                            wav_file.setsampwidth(audio_chunk.sample_width)
                            wav_file.setnchannels(audio_chunk.sample_channels)
                            wav_params_set = True
                        
                        # Add silence between sentences if any (not usually needed for single words)
                        # but keeping it safe
                        if i_chunk > 0:
                            silence_int16_bytes = bytes(int(voice.config.sample_rate * 0.0 * 2))
                            wav_file.writeframes(silence_int16_bytes)
                            
                        wav_file.writeframes(audio_chunk.audio_int16_bytes)
                total_generated += 1
            except Exception as e:
                print(f"Error synthesizing text '{t}' with model {model_path_str}: {e}")
                
    print(f"Successfully generated {total_generated} samples in {output_dir}")
