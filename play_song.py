import yt_dlp
from pydub import AudioSegment
from pydub.playback import play
import os

# # Set the URL of the video you want to play
# url = "https://www.youtube.com/watch?v=KXw8CRapg7k"

# Download the audio file
# ydl_opts = {
#     'format': 'bestaudio/best',
#     'postprocessors': [{
#         'key': 'FFmpegExtractAudio',
#         'preferredcodec': 'mp3',
#         'preferredquality': '192',
#     }],
#     'outtmpl': 'audio.%(ext)s',  # Save with this filename
# }
# with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#     ydl.download([url])
# # Load and play the audio file
# audio = AudioSegment.from_file("audio.mp3")
# play(audio)


def play_(audio_file):
    # Load the audio file
    audio = AudioSegment.from_file(audio_file)

    # Slice the first 15 seconds (10 * 1000 milliseconds)
    first_15_seconds = audio[:10 * 1000]

    # Play the sliced audio
    play(first_15_seconds)