"""Generate a simple test audio file."""
import numpy as np
import torch
import torchaudio

# Generate a simple sine wave
sample_rate = 16000
duration = 3  # seconds
frequency = 440  # Hz (A4 note)

t = np.linspace(0, duration, int(sample_rate * duration))
waveform = np.sin(2 * np.pi * frequency * t)

# Add some variation
waveform += 0.3 * np.sin(2 * np.pi * 880 * t)  # Add harmonic

# Convert to torch tensor and add channel dimension
waveform = torch.from_numpy(waveform).float().unsqueeze(0)

# Save as wav file
output_path = '../fixtures/audio/test_audio.wav'
torchaudio.save(output_path, waveform, sample_rate)

print(f"Created test audio: {output_path}")
print(f"Duration: {duration}s, Sample rate: {sample_rate}Hz, Frequency: {frequency}Hz")
