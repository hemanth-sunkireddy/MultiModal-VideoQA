import os
import pandas as pd
import librosa
from natsort import natsorted  # ensures natural sorting order (1,2,10 instead of 1,10,2)

# ======================================================
# 🔧 Global Paths (Edit These Only)
# ======================================================
ROOT_DIR = "Data/Digital-Signal-Processing/"
AUDIO_CHUNKS_DIR = "Data/Digital-Signal-Processing/Audio-Chunks"
TRANSCRIBED_TEXT_DIR = "Data/Digital-Signal-Processing/Transcribed-Text"
FINAL_SRT_DIR = "Data/Digital-Signal-Processing/SRT-Files"
# ======================================================


# === Intermediate Paths (auto-derived) ===
CSV_TIMESTAMP_DIR = "Data/Digital-Signal-Processing/complete_timestamp"  # MFA alignment CSVs
# You can modify above if needed


# ======================================================
# Utility Functions
# ======================================================
def get_audio_length(audio_file):
    """Return duration (in seconds) of an audio file."""
    y, sr = librosa.load(audio_file, sr=None)
    return librosa.get_duration(y=y, sr=sr)


def convert_to_srt_time(seconds):
    """Convert seconds to SRT timestamp format hh:mm:ss,ms."""
    millisec = int((seconds - int(seconds)) * 1000)
    hours, remainder = divmod(int(seconds), 3600)
    minutes, sec = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{sec:02},{millisec:03}"


def merge_chunks_to_srt(csv_folder, txt_folder, audio_folder, srt_file):
    """Generate a single SRT file combining sentence-level timestamps from all chunks."""
    chunk_files = natsorted([f for f in os.listdir(csv_folder) if f.endswith(".csv")])
    total_offset = 0  # Offset adjustment for next audio chunk

    with open(srt_file, "w", encoding="utf-8") as f:
        subtitle_index = 1
        for chunk in chunk_files:
            csv_path = os.path.join(csv_folder, chunk)
            txt_path = os.path.join(txt_folder, chunk.replace(".csv", ".txt"))
            audio_path = os.path.join(audio_folder, chunk.replace(".csv", ".wav"))

            # Skip if missing any required file
            if not (os.path.exists(csv_path) and os.path.exists(txt_path) and os.path.exists(audio_path)):
                print(f"⚠️ Missing files for chunk {chunk}, skipping...")
                continue

            # Get current chunk length
            chunk_length = get_audio_length(audio_path)

            # Read CSV (MFA alignment results)
            df = pd.read_csv(csv_path)
            df = df[df["Type"] == "words"].reset_index(drop=True)

            # Read sentence-wise transcription
            with open(txt_path, "r", encoding="utf-8") as txt_file:
                sentences = txt_file.readlines()

            word_index = 0

            for sentence in sentences:
                words = sentence.strip().split()
                if word_index >= len(df):
                    break

                # Determine start & end timestamps for the sentence
                start_time = df.loc[word_index, 'Begin'] + total_offset
                end_time = df.loc[min(word_index + len(words) - 1, len(df) - 1), 'End'] + total_offset

                # Write to SRT
                f.write(f"{subtitle_index}\n")
                f.write(f"{convert_to_srt_time(start_time)} --> {convert_to_srt_time(end_time)}\n")
                f.write(f"{sentence.strip()}\n\n")

                subtitle_index += 1
                word_index += len(words)

            total_offset += chunk_length  # shift timestamps for next chunk

    print(f"✅ SRT file saved: {srt_file}")


# ======================================================
# Main Processing Loop
# ======================================================
def generate_srt_files():
    """Iterate through numbered folders and generate SRTs for each."""
    os.makedirs(FINAL_SRT_DIR, exist_ok=True)

    for subfolder in sorted(os.listdir(AUDIO_CHUNKS_DIR), key=lambda x: int(x)):
        subfolder_audio_path = os.path.join(AUDIO_CHUNKS_DIR, subfolder)
        if not os.path.isdir(subfolder_audio_path):
            continue

        csv_subfolder = os.path.join(CSV_TIMESTAMP_DIR, subfolder)
        txt_subfolder = os.path.join(TRANSCRIBED_TEXT_DIR, subfolder)
        audio_subfolder = subfolder_audio_path
        srt_output_path = os.path.join(FINAL_SRT_DIR, f"{subfolder}.srt")

        if not os.path.exists(csv_subfolder):
            print(f"⚠️ Missing CSV folder for {subfolder}, skipping...")
            continue

        merge_chunks_to_srt(csv_subfolder, txt_subfolder, audio_subfolder, srt_output_path)


# ======================================================
# Entry Point
# ======================================================
if __name__ == "__main__":
    generate_srt_files()
