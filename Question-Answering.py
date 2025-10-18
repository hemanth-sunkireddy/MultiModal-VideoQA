FOLDER_NAME = "Machine-Learning/"

VIDEO_DIR = FOLDER_NAME + "Videos"             # Folder containing input video files
AUDIO_DIR = FOLDER_NAME + "Audios"             # Folder to store extracted audio files
CHUNK_DIR = FOLDER_NAME + "Audio-Chunks"       # Folder to save audio chunks after VAD# Directory containing SRT files
srt_directory = FOLDER_NAME + "SRT-Files"
sentences_file = FOLDER_NAME + 'sentences.txt'
metadata_file = FOLDER_NAME + 'srt-embedding-metadata.tsv'
grouped_sentences_file = FOLDER_NAME + "grouped_sentences.pkl"
grouped_sent_to_metadata_file = FOLDER_NAME + "grouped_sent_to_metadata.pkl"
grouped_sentences_embeddings_file = FOLDER_NAME + "grouped-sentences-embeddings.idx"
final_video_stitched_output_srt_file = FOLDER_NAME + "stitched_output.srt"
final_video_file = FOLDER_NAME + "answer.mp4"

from summarizer import Summarizer

# Initialize summarizer
summarizer = Summarizer()

import ffmpeg
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import re
import os
from tqdm import tqdm
import csv
import pickle
from datetime import datetime
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

faiss_index = faiss.read_index(grouped_sentences_embeddings_file)

with open(grouped_sentences_file, "rb") as f:
    grouped_sentences = pickle.load(f)

with open(grouped_sent_to_metadata_file, "rb") as f:
    grouped_sent_to_metadata = pickle.load(f)

student_question = input("Enter your question: ")
question_embedding = model.encode(student_question, prompt_name="query")
if question_embedding.ndim == 1:
    question_embedding = np.expand_dims(question_embedding, axis=0)

distances, indices = faiss_index.search(question_embedding, 10)

related_results = []
for idx in indices[0]:
    grouped_sent = grouped_sentences[idx]
    meta = grouped_sent_to_metadata.get(grouped_sent, None)

    if meta:
        filename = meta["filename"]
        timestamp_range = meta["timestamp_range"]
        individual_timestamps = meta.get("individual_timestamps", [])
    else:
        filename = "Unknown"
        timestamp_range = "Unknown"
        individual_timestamps = []

    related_results.append(
        (filename, timestamp_range, grouped_sent, individual_timestamps)
    )

print("Question:", student_question)
print("\nTop Related Sentences with Metadata:")
for filename, timestamp, sent, indiv_ts in related_results:
    print(f"- [{filename} | {timestamp}] {sent}")
    if indiv_ts:
        print("  ↳ Individual timestamps:", indiv_ts)
    print()

# Group results by file
grouped_by_file = {}
for filename, timestamp, sent, _ in related_results:
    if filename not in grouped_by_file:
        grouped_by_file[filename] = []
    grouped_by_file[filename].append((timestamp, sent))

# Helper to parse timestamps
def parse_ts(ts):
    start, end = ts.split("->")
    fmt = "%H:%M:%S.%f"
    return datetime.strptime(start.strip(), fmt), datetime.strptime(end.strip(), fmt)

# Helper to clean and normalize sentences
def clean_sentences(sentences):
    cleaned = []
    seen = set()
    for s in sentences:
        s = s.strip()
        while s and s[-1] in ".!?":
            s = s[:-1].strip()
        if s and s not in seen:
            cleaned.append(s)
            seen.add(s)
    return cleaned

def extract_file_number(fname):
    match = re.search(r"(\d+)", fname)
    return int(match.group(1)) if match else float("inf")

# Sort by file and start timestamp
related_results_sorted = sorted(
    related_results,
    key=lambda x: (extract_file_number(x[0]), parse_ts(x[1])[0])
)

# Merge overlapping timestamps within same file
merged_results = []
for filename, ts_range, text, indiv_ts in related_results_sorted:
    sentences = clean_sentences(text.split('. '))

    if not merged_results:
        merged_results.append([
            filename, ts_range, '. '.join(sentences) + '.', indiv_ts
        ])
        continue

    prev = merged_results[-1]
    prev_start, prev_end = parse_ts(prev[1])
    curr_start, curr_end = parse_ts(ts_range)

    if filename == prev[0] and curr_start <= prev_end:
        new_start = prev_start.strftime("%H:%M:%S.%f")[:-3]
        new_end = max(prev_end, curr_end).strftime("%H:%M:%S.%f")[:-3]
        prev[1] = f"{new_start} -> {new_end}"

        prev_sentences = clean_sentences(prev[2].split('. '))
        combined_sentences = prev_sentences + sentences
        prev[2] = '. '.join(clean_sentences(combined_sentences)) + '.'

        # Combine timestamps
        prev[3].extend(indiv_ts)
    else:
        merged_results.append([filename, ts_range, '. '.join(sentences) + '.', indiv_ts])

# Print merged results
print("\nOrdered & Merged Top Related Sentences (duplicates removed):")
for i, (fname, ts, sent, indiv_ts) in enumerate(merged_results, 1):
    print(f"\nGroup {i}")
    print("Text:", sent)
    print("Metadata:", (fname, ts))
    if indiv_ts:
        print("Individual timestamps:", indiv_ts)

segment_list = []

for filename, ts_range, text, individual_timestamps in merged_results:
    distance = 0.0  # placeholder for similarity or ranking score
    segment_list.append({
        "filename": filename,
        "timestamp_range": ts_range,
        "text": text,
        "individual_timestamps": individual_timestamps,
        "distance": distance
    })

# Now segment_list is ready to use
for seg in segment_list:
    print(f"[{seg['filename']} | {seg['timestamp_range']}]")
    print(f"Text: {seg['text']}")
    print(f"Individual timestamps: {seg['individual_timestamps']}")
    print()


# -------------------
# 🧠 Summarization Helper
# -------------------
def summarize_and_save(ratio, output_filename):
    texts = [sent for _, _, sent, _ in merged_results]
    timestamps = [ts for _, ts, _, _ in merged_results]

    full_text = " ".join(texts)
    summary_text = summarizer(full_text, ratio=ratio)
    if isinstance(summary_text, list):
        summary_text = " ".join(summary_text)

    summary_sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', summary_text) if s.strip()]

    def find_timestamp(sentence):
        for text, ts in zip(texts, timestamps):
            if sentence[:20].lower() in text.lower() or sentence.lower() in text.lower():
                return ts
        return "Unknown"

    print(f"\nSummary (ratio={ratio}) with timestamps:\n")
    summary_rows = []
    for s in summary_sentences:
        ts = find_timestamp(s)
        print(f"[{ts}] {s}")
        filename = merged_results[0][0] if merged_results else "Unknown"
        summary_rows.append([filename, ts, s])

    with open(output_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "timestamp", "sentence"])
        writer.writerows(summary_rows)
    print(f"\n✅ Saved summary to: {output_filename}\n")

# -------------------
# 📄 Save long (actual grouped) version
# -------------------
with open("long-answer.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "timestamp", "sentence"])
    for filename, ts, sent, _ in merged_results:
        writer.writerow([filename, ts, sent])
print("✅ Saved grouped (long) version to long-answer.csv\n")

# -------------------
# 📝 Generate short and medium summaries
# -------------------
summarize_and_save(0.3, "short-answer.csv")   # short summary
summarize_and_save(0.7, "medium-answer.csv")  # medium summary