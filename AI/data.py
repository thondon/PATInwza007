import os
import numpy as np
import librosa
import librosa.display
import glob
import soundfile as sf
import matplotlib.pyplot as plt
import speech_recognition as sr
from pythainlp.tokenize import word_tokenize

# ===== Global Parameters =====
TARGET_DURATION = 160  # Duration to stretch/compress all audio to (seconds)
DATA_SERIAL = 'tmp'      #
SR_RATE = 2200         # Sampling rate
FPS = 10               # Frames per second (sampling for pitch)
HOP_LEN = SR_RATE // FPS  # Hop length based on FPS
FRAME_LEN = SR_RATE // FPS
FMIN = 80              # Min pitch (Hz)
FMAX = 1100            # Max pitch (Hz)

# ===== Ensure Directories Exist =====
def ensure_dirs():
    dirs = [f"./data/{DATA_SERIAL}", f"./pitch/{DATA_SERIAL}",
            f"./lax/{DATA_SERIAL}", f"./audio/{DATA_SERIAL}"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

# ===== Load Audio =====
def load_audio(wav_path):
    y, sr = librosa.load(wav_path, sr=SR_RATE)
    return y, sr

# ===== Stretch Audio =====
def stretch_audio(y, sr, target_duration):
    current_duration = librosa.get_duration(y=y, sr=sr)
    stretch_factor = current_duration / target_duration
    y_stretched = librosa.effects.time_stretch(y, rate=stretch_factor)
    return y_stretched

# ===== Extract Pitch =====
def extract_pitch(y, sr):
    pitches, voiced_flag, _ = librosa.pyin(
        y, fmin=FMIN, fmax=FMAX, sr=sr,
        hop_length=HOP_LEN, frame_length=FRAME_LEN
    )
    # Fill NaN with 0 for unvoiced frames
    pitches = np.nan_to_num(pitches)
    return pitches

# ===== Plot Pitch =====
def plot_pitch(pitches, sr, audio_name):
    times = librosa.frames_to_time(np.arange(len(pitches)), sr=sr, hop_length=HOP_LEN)
    plt.figure(figsize=(10, 4))
    plt.plot(times, pitches, color='blue')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(f'Pitch Track: {audio_name}')
    plt.tight_layout()
    path = f'./pitch/{DATA_SERIAL}/{audio_name}.jpg'
    plt.savefig(path)
    plt.close()

# ===== Save Audio =====
def save_audio(y, sr, audio_name):
    sf.write(f'./audio/{DATA_SERIAL}/{audio_name}.wav', y, sr)

# ===== Save Pitch Data =====
def save_pitch(pitches, audio_name):
    np.save(f'./data/{DATA_SERIAL}/{audio_name}.npy', pitches)

# ===== Speech to Text =====
def speech_to_text(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio, language="th-TH")
        syllables = word_tokenize(text, engine="newmm")
        return " ".join(syllables)
    except (sr.UnknownValueError, sr.RequestError):
        return ""

# ===== Save Text =====
def save_text(text, audio_name):
    with open(f'./lax/{DATA_SERIAL}/{audio_name}.txt', 'w', encoding='utf-8') as f:
        f.write(text)

# ===== Main Process =====
def process_audio(wav_path):
    ensure_dirs()
    
    audio_name = os.path.splitext(os.path.basename(wav_path))[0]
    y, sr = load_audio(wav_path)
    
    y_stretched = stretch_audio(y, sr, TARGET_DURATION)
    
    pitches = extract_pitch(y_stretched, sr)
    save_pitch(pitches, audio_name)
    plot_pitch(pitches, sr, audio_name)
    
    save_audio(y_stretched, sr, audio_name)
    
    # Extract text after stretching (or original file if you prefer)
    text = speech_to_text(wav_path)
    save_text(text, audio_name)
    print(f"Processed {audio_name} successfully.")

# ===== Processed =====

def process_all_audios(folder_path):
    ensure_dirs()

    wav_files = glob.glob(folder_path)
    print(f"Found {len(wav_files)} files.")

    for wav_path in wav_files:
        process_audio(wav_path)

folder_path = f"./_lake/{DATA_SERIAL}/train/*.wav"
process_all_audios(folder_path)
folder_path = f"./_lake/{DATA_SERIAL}/test/*.wav"
process_all_audios(folder_path)