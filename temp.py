import os
from pathlib import Path

# Folder containing your videos
folder_path = Path("Data/Videos-Digital-Signal-Processing")  # <-- change this

# File extensions to include
video_extensions = {'.mp4'}

# Collect all video files
files = [f for f in folder_path.iterdir() if f.suffix.lower() in video_extensions]

# Sort files by creation time
files.sort(key=lambda f: f.stat().st_ctime)

# Rename files as 1.mp4, 2.mp4, ...
for i, file in enumerate(files, start=1):
    new_name = f"{i}{file.suffix.lower()}"
    new_path = folder_path / new_name
    print(f"Renaming: {file.name} -> {new_name}")
    file.rename(new_path)

print("✅ Renaming completed!")
