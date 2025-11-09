
from helpers.video_sticher import stitch_video_from_segments
import csv
import numpy as np
import os
import uuid
import requests
import json

# Load lecture sentences
def load_lecture_data(sentences_path='./data/sentences.txt', metadata_path='./data/srt-embedding-metadata.tsv'):
    with open(sentences_path, 'r') as file:
        lecture_sentences = file.readlines()
    lecture_sentences = [line.strip() for line in lecture_sentences if line.strip()]

    lecture_data = []
    with open(metadata_path, 'r', encoding='utf-8') as file:
        tsv_reader = csv.reader(file, delimiter='\t')
        for row in tsv_reader:
            if len(row) == 3:
                filename, timestamp, sentence = row
                lecture_data.append((filename.strip(), timestamp.strip(), sentence.strip()))
    return lecture_sentences, lecture_data

# def generate_video_answer(question: str, model, faiss_index, lecture_sentences, lecture_data):
#     # Clear previous stitched video and subtitle files
#     output_video = './data/output/stitched_output.mp4'
#     output_srt = './data/output/stitched_output.srt'

#     for file in [output_video, output_srt]:
#         if os.path.exists(file):
#             os.remove(file)

#     question_embedding = np.array(model.encode([question])).astype('float32')
#     # Search all sentences (max number can be total sentences in the index)
#     distances, indices = faiss_index.search(question_embedding, len(lecture_sentences))

#     # Define a distance threshold (lower means more similar)
#     distance_threshold = 0.7

#     related_sentences = []
#     related_results = []
#     for j in range(len(indices[0])):
#         i = indices[0][j]
#         distance = distances[0][j]
#         sentence = lecture_sentences[i]
        
#         # Check if the sentence is below the distance threshold and is not a question
#         if distance > 0 and distance <= distance_threshold and not sentence.strip().endswith('?'):
#             related_sentences.append((sentence, distance))
#             filename, timestamp, _ = lecture_data[i+1]
#             related_results.append((filename, timestamp, sentence, distance))

#     segments_info, sources = stitch_video_from_segments(related_results,pause_duration=1.0)

#     video_url = f"./data/output/stitched_output.mp4?cache_bust={uuid.uuid4()}"
#     # Read SRT content from the file
#     try:
#         with open('./data/output/stitched_output.srt', 'r', encoding='utf-8') as f:
#             srt_content = f.read()
#     except FileNotFoundError:
#         srt_content = "SRT file not found."
#     except Exception as e:
#         srt_content = f"An error occurred while reading the SRT file: {e}"

#     return {
#         "srtContent": srt_content,
#         "videoUrl": video_url,
#         "sources": sources,
#         "segments": segments_info
#     }

# Global cache to store the last question and its API response
_last_api_cache = {
    "question": None,
    "course": None,
    "response": None
}

def generate_video_answer(question: str, course: str, answer_length: str = "medium"):
    global _last_api_cache

    output_video = './data/output/stitched_output.mp4'
    output_srt = './data/output/stitched_output.srt'

    # Clear old stitched outputs
    for file in [output_video, output_srt]:
        if os.path.exists(file):
            os.remove(file)

    # ✅ Check if question matches the cached one
    if (
        _last_api_cache["question"] == question.strip().lower()
        and _last_api_cache["course"] == course.strip().lower()
        and _last_api_cache["response"] is not None
    ):
        print("⚡ Using cached API response")
        result = _last_api_cache["response"]
    else:
        print("🌐 Making new API request")
        api_url = "https://flask-ml-gcloud-696109823957.europe-west1.run.app/"
        payload = {"question": question, "course": course}

        try:
            response = requests.post(api_url, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
            response.raise_for_status()
            result = response.json()

            # ✅ Update cache with new data
            _last_api_cache = {
                "question": question.strip().lower(),
                "course": course.strip().lower(),
                "response": result
            }

        except requests.exceptions.RequestException as e:
            return {"error": f"API request failed: {e}"}
        except ValueError:
            return {"error": "Invalid JSON response from API."}

    # Directly use API output (e.g., long_answer)
    answer_segments = result.get(answer_length, [])
    print(answer_segments)

    segments_info, sources = stitch_video_from_segments(answer_segments, course,  pause_duration=0.01)

    video_url = f"./data/output/stitched_output.mp4?cache_bust={uuid.uuid4()}"

    try:
        with open(output_srt, 'r', encoding='utf-8') as f:
            srt_content = f.read()
    except FileNotFoundError:
        srt_content = "SRT file not found."
    except Exception as e:
        srt_content = f"An error occurred while reading the SRT file: {e}"

    return {
        "srtContent": srt_content,
        "videoUrl": video_url,
        "sources": sources,
        "segments": segments_info
    }
