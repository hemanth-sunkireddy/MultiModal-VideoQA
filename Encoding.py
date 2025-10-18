FOLDER_NAME = "Machine-Learning/"

VIDEO_DIR = FOLDER_NAME + "Videos"             # Folder containing input video files
AUDIO_DIR = FOLDER_NAME + "Audios"             # Folder to store extracted audio files
CHUNK_DIR = FOLDER_NAME + "Audio-Chunks"       # Folder to save audio chunks after VAD# Directory containing 
srt_directory = FOLDER_NAME + "SRT-Files"
sentences_file = FOLDER_NAME + 'sentences.txt'
metadata_file = FOLDER_NAME + 'srt-embedding-metadata.tsv'
grouped_sentences_file = FOLDER_NAME + "grouped_sentences.pkl"
grouped_sent_to_metadata_file = FOLDER_NAME + "grouped_sent_to_metadata.pkl"
grouped_sentences_embeddings_file = FOLDER_NAME + "grouped-sentences-embeddings.idx"
final_video_stitched_output_srt_file = FOLDER_NAME + "stitched_output.srt"
final_video_file = FOLDER_NAME + "answer.mp4"

import ffmpeg
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import re
import os
from tqdm import tqdm
import csv
import pickle


model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

metadata_list = []
with open(metadata_file, "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)
    for row in reader:
        filename, timestamp, sentence = row
        metadata_list.append((filename.strip(), timestamp.strip(), sentence.strip()))

group_size = 3
grouped_sentences = []
grouped_sent_to_metadata = {}

def extract_start_end(ts):
    start, end = ts.split("-->")
    return start.strip(), end.strip()

# Group by filename first
from itertools import groupby

for filename, file_group in groupby(metadata_list, key=lambda x: x[0]):
    file_group = list(file_group)
    for i in range(len(file_group) - group_size + 1):
        group = file_group[i:i+group_size]
        grouped_text = " ".join(sent for _, _, sent in group)
        first_start, _ = extract_start_end(group[0][1])
        _, last_end = extract_start_end(group[-1][1])
        timestamp_range = f"{first_start} --> {last_end}"
        individual_timestamps = [ts for _, ts, _ in group]
        grouped_sentences.append(grouped_text)
        grouped_sent_to_metadata[grouped_text] = {
            "filename": filename,
            "timestamp_range": timestamp_range,
            "individual_timestamps": individual_timestamps
        }

# Save pickles
with open(grouped_sentences_file, "wb") as f:
    pickle.dump(grouped_sentences, f)

with open(grouped_sent_to_metadata_file, "wb") as f:
    pickle.dump(grouped_sent_to_metadata, f)

first_key = next(iter(grouped_sent_to_metadata))
print("Grouped Sentence to Metadata First element:", first_key, "->", grouped_sent_to_metadata[first_key])

# Load grouped sentences
with open(grouped_sentences_file, "rb") as f:
    grouped_sentences = pickle.load(f)

# Load precomputed metadata mapping
with open(grouped_sent_to_metadata_file, "rb") as f:
    grouped_sent_to_metadata = pickle.load(f)
print("Encoding grouped sentences into embeddings...")

batch_size = 16  # you can adjust this based on memory
all_embeddings = []

for i in tqdm(range(0, len(grouped_sentences), batch_size), desc="Encoding batches"):
    batch = grouped_sentences[i:i+batch_size]
    batch_emb = model.encode(batch)  # encode the batch
    all_embeddings.append(batch_emb)

# Combine all batches into one array and convert to float32
grouped_embeddings = np.vstack(all_embeddings).astype('float32')

# Create FAISS index (Inner Product for cosine similarity)
embedding_dim = grouped_embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(embedding_dim)
faiss_index.add(grouped_embeddings)

# Save FAISS index
faiss.write_index(faiss_index, grouped_sentences_embeddings_file)

print("✅ FAISS index created and saved!")
