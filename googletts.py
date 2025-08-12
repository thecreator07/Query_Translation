from google.cloud import texttospeech
import subprocess
import sys
import os



def open_with_default_app(filepath):
    if os.name == "nt":       # Windows
        os.startfile(filepath)
    elif sys.platform == "darwin":  # macOS
        subprocess.call(["open", filepath])
    else:  # Linux / X11-based systems
        subprocess.call(["xdg-open", filepath])

# After writing output.mp3:

# Instantiate a client using service account credentials
client = texttospeech.TextToSpeechClient.from_service_account_file(
    'text-to-speech-467917-67fb31ad2152.json'
)

# Set the text input to be synthesized
synthesis_input = texttospeech.SynthesisInput(text="Hello, World! This is a test of Google Text-to-Speech API.")

# Build the voice request
voice = texttospeech.VoiceSelectionParams(
    language_code="en-US",
    ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
)

# Configure the audio output format
audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.MP3
)

# Perform the text-to-speech request
response = client.synthesize_speech(
    input=synthesis_input,
    voice=voice,
    audio_config=audio_config
)

# Write the output to a file
with open("output.mp3", "wb") as out:
    out.write(response.audio_content)
    print('Audio content written to file "output.mp3"')

# os.startfile("output.mp3")
# open_with_default_app("output.mp3")
subprocess.run(
    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-"],
    input=response.audio_content
)