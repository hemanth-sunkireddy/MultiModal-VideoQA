import os
import datetime
import csv
from moviepy.editor import VideoFileClip, concatenate_videoclips, ColorClip

# Set paths
VIDEO_DIR = "Data/Machine-Learning/Videos"  # folder containing video files like 1.mp4, 10.mp4 etc.
FINAL_VIDEO_FILE = "Data/Machine-Learning/Results/Q2/long-answer.mp4"
FINAL_SRT_FILE = "Data/Machine-Learning/Results/Q2/long-answer.srt"
INPUT_CSV_FILE = "Data/Machine-Learning/Results/Q2/long-answer.csv"

def format_srt_timestamp(seconds):
    td = datetime.timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    milliseconds = int((td.total_seconds() - total_seconds) * 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{int(seconds):02},{milliseconds:03}"


def create_continuous_srt(clips_info, output_filename=FINAL_SRT_FILE, transition_sec=0.01):
    srt_lines = []
    current_time = 0.0

    for idx, (duration, sentence, start, video_file) in enumerate(clips_info, start=1):
        start_time = format_srt_timestamp(current_time)
        end_time = format_srt_timestamp(current_time + duration)
        srt_lines.append(f"{idx}\n{start_time} --> {end_time}\n{sentence}\n")
        current_time += duration + transition_sec

    with open(output_filename, "w") as f:
        f.write("\n".join(srt_lines))


def parse_timestamp(timestamp_str):
    start, end = timestamp_str.split("--> ")
    return start.strip(), end.strip()


def load_segments_from_csv(csv_path):
    """Read CSV file with columns: filename,timestamp,sentence"""
    segments = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            filename, timestamp, sentence = row[0].strip(), row[1].strip(), row[2].strip()
            segments.append((filename, timestamp, sentence))
    return segments


def stitch_video_from_csv(csv_path, pause_duration=0.01):
    segment_list = load_segments_from_csv(csv_path)
    clips = []
    clips_info = []

    for idx, (filename, timestamp, sentence) in enumerate(segment_list):
        lecture_no = os.path.splitext(filename)[0]
        video_file = os.path.join(VIDEO_DIR, lecture_no + ".mp4")

        if not os.path.exists(video_file):
            print(f"Warning: Missing video file {video_file}, skipping.")
            continue

        start, end = parse_timestamp(timestamp)

        try:
            clip = VideoFileClip(video_file).subclip(start, end)

            # Add small black transition between clips
            if idx > 0:
                black_clip = ColorClip(size=clip.size, color=(0, 0, 0), duration=pause_duration)
                black_clip = black_clip.set_fps(clip.fps)
                clips.append(black_clip)

            clips.append(clip)
            clips_info.append((clip.duration, sentence, start, video_file))
        except Exception as e:
            print(f"Error processing segment ({filename}, {timestamp}): {e}")

    if not clips:
        print("No valid clips found.")
        return

    final_clip = concatenate_videoclips(clips, method="chain")

    final_clip.write_videofile(
        FINAL_VIDEO_FILE,
        codec="libx264",
        preset="ultrafast",
        threads=4,
        audio_codec="aac"
    )

    create_continuous_srt(clips_info, output_filename=FINAL_SRT_FILE, transition_sec=pause_duration)
    print(f"✅ Final stitched video saved as: {FINAL_VIDEO_FILE}")
    print(f"✅ Continuous SRT file saved as: {FINAL_SRT_FILE}")


# Example usage
if __name__ == "__main__":
    CSV_INPUT_FILE =  INPUT_CSV_FILE # your input CSV file
    stitch_video_from_csv(CSV_INPUT_FILE)
