import yt_dlp

ydl_opts = {
    # This format string selects the best video and audio streams
    # that are available in MP4 format.
    # The `ext=mp4` filter ensures that the output is an MP4 file.
    'format': 'bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
    'merge_output_format': 'mp4', # This is crucial for merging to MP4
}


video_url = f'https://www.youtube.com/playlist?list=PLvv3PyiCcNrFuT7CEIvIr4a4g4orascx3'
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    try:
        ydl.download([video_url])
        print("Download complete!")
    except Exception as e:
        print(f"An error occurred: {e}")
