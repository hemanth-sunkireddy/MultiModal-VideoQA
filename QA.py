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
    """
    Safely parse timestamp ranges like:
    '00:03:14.890 --> 00:03:33.410'
    or '00:03:14.890->00:03:33.410-'
    """
    # Normalize all variants of separators
    ts = ts.replace("-->", "->").replace("—>", "->").replace("−>", "->").strip()

    # Guard: skip empty or malformed entries
    if not ts or "->" not in ts:
        print(f"⚠️ Skipping invalid timestamp: {ts}")
        return datetime.min, datetime.min

    # Split and clean both sides
    start, end = ts.split("->", 1)
    fmt = "%H:%M:%S.%f"

    # Remove any unwanted characters (like '-' or text)
    start = re.sub(r"[^0-9:.]", "", start.strip())
    end = re.sub(r"[^0-9:.]", "", end.strip())

    try:
        return datetime.strptime(start, fmt), datetime.strptime(end, fmt)
    except ValueError as e:
        print(f"⚠️ Failed to parse '{ts}' → {e}")
        # fallback to zero time to keep sorting consistent
        return datetime.strptime("00:00:00.000", fmt), datetime.strptime("00:00:00.000", fmt)

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

with open("long-answer.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "timestamp", "sentence"])
    for filename, ts, sent, _ in merged_results:
        writer.writerow([filename, ts, sent])
print("✅ Saved grouped (long) version to long-answer.csv\n")

# Build flat list in memory (sentence -> timestamp)
# ----------------------------
flat_sentences = []  # [(filename, sentence, timestamp)]

for filename, group_ts, group_text, indiv_ts_list in merged_results:
    # Split group text into individual sentences using full stop
    indiv_sents = [s.strip() for s in re.split(r'\.\s+', group_text) if s.strip()]
    indiv_ts_list = indiv_ts_list or []
    
    # Map each sentence to its corresponding individual timestamp
    for i, s in enumerate(indiv_sents):
        if s.endswith('.'):
            s = s[:-1].strip()
        ts = indiv_ts_list[i] if i < len(indiv_ts_list) else group_ts
        flat_sentences.append((filename, s, ts))

print(f"✅ Built in-memory flat mapping with {len(flat_sentences)} sentences")

# ----------------------------
# Build flat dict: { sentence : (filename, timestamp) }
# ----------------------------
flat_sentence_map = {}

for filename, group_ts, group_text, indiv_ts_list in merged_results:
    indiv_sents = [s.strip() for s in re.split(r'\.\s+', group_text) if s.strip()]
    indiv_ts_list = indiv_ts_list or []

    for i, s in enumerate(indiv_sents):
        if s.endswith('.'):
            s = s[:-1].strip()
        ts = indiv_ts_list[i] if i < len(indiv_ts_list) else group_ts
        # Store sentence → (filename, timestamp)
        flat_sentence_map[s] = (filename, ts)

print(f"✅ Built flat mapping with {len(flat_sentence_map)} sentences (key-value dict)")
for key, value in flat_sentence_map.items():
    print(key, value)
    
# Combine everything for summarization
all_sentences = list(flat_sentence_map.keys())
full_text = ". ".join(all_sentences)
print(f"Full Text: {full_text}")

# ----------------------------
# Summarization + direct key-value timestamp lookup
# ----------------------------
def summarize_and_save_from_dict(ratio, output_filename):
    print(f"\nSummary (ratio={ratio}) with timestamps:\n")
    summary_text = summarizer(full_text, ratio=ratio)
    print(summary_text)
    # Split summary only on full stops
    summary_sentences = [s.strip() for s in summary_text.split('. ') if s.strip()]
    print(f"{summary_sentences} this are the")
    summary_rows = []

    for s in summary_sentences:
        matched_file, matched_ts = "Unknown", "Unknown"

        # ✅ Direct match or partial match lookup
        # Try exact sentence
        if s in flat_sentence_map:
            matched_file, matched_ts = flat_sentence_map[s]
        else:
            # Try partial substring match (start or inclusion)
            for orig_sentence, (fname, ts) in flat_sentence_map.items():
                if s[:25].lower() in orig_sentence.lower() or orig_sentence.lower() in s.lower():
                    matched_file, matched_ts = fname, ts
                    break

        print(f"[{matched_ts}] {s}.")
        summary_rows.append([matched_file, matched_ts, s])

    # Save summary CSV
    with open(output_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "timestamp", "sentence"])
        writer.writerows(summary_rows)

    print(f"\n✅ Saved summary to: {output_filename}\n")

# ----------------------------
# Generate short and medium summaries
# ----------------------------
summarize_and_save_from_dict(0.3, "short-answer.csv")
summarize_and_save_from_dict(0.7, "medium-answer.csv")
