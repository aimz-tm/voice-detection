import base64

with open("sample.wav", "rb") as f:
    audio_bytes = f.read()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

with open("audio.txt", "w") as f:
    f.write(audio_b64)

print("Base64 written to audio.txt")
