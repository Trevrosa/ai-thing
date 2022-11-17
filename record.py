import pyaudio
import wave
from pathlib import Path

chunk = 1024  # record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 2
fs = 44100  # record at 44100 samples per second

seconds = 1
classes = int(input("how many classes? "))

print("\n", end="")

for i in range(classes):

    filename = f"sounds/class{i}.wav"

    p = pyaudio.PyAudio()  # create an interface to PortAudio

    print(f"recording ({i})..", end="")

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # create list to store frames

    # store data in chunks for n seconds
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()

    p.terminate()

    Path("sounds").mkdir(exist_ok=True)

    # Save the recorded data as a WAV file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))

    print("done")
