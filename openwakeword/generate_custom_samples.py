import os
import sys
import uuid
import random
import wave
import tempfile
import numpy as np
from scipy import signal
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
    noise_scale_ws: List[float] = [0.98],
    tts_engine: str = "piper",
    batch_size: int = 32
):
    """
    Generate synthetic TTS samples using multiple Piper ONNX models or VieNeu-TTS.
    """
    if isinstance(text, str):
        texts = [text]
    else:
        texts = text

    os.makedirs(output_dir, exist_ok=True)
    total_generated = 0
    TARGET_SAMPLE_RATE = 16000

    if tts_engine.lower() == "vieneu":
        print(f"Generating {max_samples} samples using VieNeu-TTS...")
        try:
            from vieneu import Vieneu
        except ImportError as e:
            raise ImportError("Failed to import vieneu. Please ensure it is installed via 'pip install vieneu'.") from e
        
        try:
            # Khởi tạo Vieneu. Nó sẽ tự động dùng chế độ v3turbo chạy siêu nhẹ trên CPU thông qua ONNX
            tts = Vieneu()  
            preset_voices = tts.list_preset_voices()
            if not preset_voices:
                raise ValueError("No preset voices found in VieNeu-TTS.")
        except Exception as e:
            print(f"Failed to load VieNeu-TTS: {e}")
            return
            
        BATCH_SIZE = batch_size
        import scipy.io.wavfile
        from tqdm import tqdm
        
        with tqdm(total=max_samples, desc=f"Batched Generation (BS={BATCH_SIZE})") as pbar:
            while total_generated < max_samples:
                current_batch_size = min(BATCH_SIZE, max_samples - total_generated)
                batch_texts = [random.choice(texts) for _ in range(current_batch_size)]
                label, voice_id = random.choice(preset_voices)
                
                try:
                    # Infer the entire batch
                    audio_arrays = tts.infer_batch(batch_texts, voice=voice_id)
                    
                    for audio in audio_arrays:
                        unique_id = str(uuid.uuid4())
                        wav_path = os.path.join(output_dir, f"{unique_id}.wav")
                        
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                            tmp_path = tmp_file.name
                        
                        tts.save(audio, tmp_path)
                        orig_sr, dat = scipy.io.wavfile.read(tmp_path)
                        
                        if orig_sr != TARGET_SAMPLE_RATE:
                            num_samples = int(round(len(dat) * float(TARGET_SAMPLE_RATE) / orig_sr))
                            dat = signal.resample(dat, num_samples)
                        
                        if dat.dtype != np.int16:
                            _MAX_WAV_VALUE = 32767.0
                            if dat.dtype in [np.float32, np.float64]:
                                dat = np.clip(dat * _MAX_WAV_VALUE, -_MAX_WAV_VALUE, _MAX_WAV_VALUE).astype(np.int16)
                            else:
                                dat = dat.astype(np.int16)
                        
                        scipy.io.wavfile.write(wav_path, TARGET_SAMPLE_RATE, dat)
                        os.remove(tmp_path)
                        total_generated += 1
                        pbar.update(1)
                        
                except Exception as e:
                    print(f"\nError synthesizing batch with VieNeu-TTS (voice={voice_id}): {e}")
                    if 'tmp_path' in locals() and os.path.exists(tmp_path):
                        os.remove(tmp_path)
                    
        print(f"Successfully generated {total_generated} samples using VieNeu-TTS in {output_dir}")
        return

    # Original Piper Logic
    try:
        from piper import PiperVoice, SynthesisConfig
    except ImportError as e:
        raise ImportError(f"Failed to import Piper. Please ensure it is installed.") from e

    samples_per_model = max_samples // len(piper_models)
    remaining_samples = max_samples % len(piper_models)

    print(f"Generating {max_samples} samples across {len(piper_models)} piper models...")

    for i, model_path_str in enumerate(piper_models):
        model_path = Path(model_path_str)
        if not model_path.exists():
            print(f"Warning: Piper model not found at {model_path_str}. Skipping...")
            continue
            
        print(f"Loading piper model: {model_path_str}")
        try:
            # use_cuda=False by default to avoid issues, can be changed if GPU is supported
            voice = PiperVoice.load(model_path, use_cuda=False)
        except Exception as e:
            print(f"Failed to load piper model {model_path_str}: {e}")
            continue
            
        target_samples = samples_per_model + (1 if i < remaining_samples else 0)
        
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
                    print(f"Warning: No audio chunks generated for text '{t}'")
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
                total_generated += 1
            except Exception as e:
                print(f"Error synthesizing text '{t}' with model {model_path_str}: {e}")
                if os.path.exists(wav_path):
                    os.remove(wav_path)
                
    print(f"Successfully generated {total_generated} samples in {output_dir}")
