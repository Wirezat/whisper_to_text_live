
import argparse
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import exit, platform

def parse_arguments():
    default_mic = "pulse" if 'linux' in platform else "default"

    parser = argparse.ArgumentParser(description="Speech recognition with Whisper and real-time transcription.")
    parser.add_argument("--model", default="turbo", choices=["tiny", "base", "small", "medium", "large", "turbo"],
                        help="Model to use for speech recognition.")
    parser.add_argument("--energy_threshold", type=int, default=1000,
                        help="Energy level for the microphone to detect speech.")
    parser.add_argument("--record_timeout", type=float, default=10,
                        help="Duration of each recording segment in seconds.")
    parser.add_argument("--phrase_timeout", type=float, default=2,
                        help="Silence duration before considering a new transcription line.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                        help="Device to run the model on (CPU or CUDA).")
    parser.add_argument("--default_microphone", default=default_mic, type=str,
                        help="Default microphone name. Use 'list' to show available microphones.")
    parser.add_argument("--log_folder", default=".", type=str,
                        help="Folder, where the output file should be generated")
    return parser.parse_args()

def make_file(path) -> str:
    # Ensure the log folder exists
    if not os.path.exists(path):
        print(f"The folder {path} does not exist. Creating it.")
        os.makedirs(path)

    # Create a filename with timestamp to ensure it is unique
    filename = f"log_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.txt"
    log_path = os.path.join(path, filename)  # Corrected from args.log_folder to path
    with open(log_path, 'a'):
        pass  # Just create the file (touch)
    return log_path

def main():
    # Parse arguments
    args = parse_arguments()

    # Create file in the specified folder
    log_path = make_file(args.log_folder)

    mic_name = args.default_microphone
    if mic_name == 'list':
        [print(f"Microphone with name \"{name}\" found") for index, name in enumerate(sr.Microphone.list_microphone_names())]
        exit(0)
    # Important for linux users. Goes through all microphone indexes, until match with given parameter is found, then assigns index and sample rate.
    # Skips, if run on a windows system, assigns only sample rate.
    mic_index = next((index for index, name in enumerate(sr.Microphone.list_microphone_names()) if mic_name in name), None) if 'linux' in platform else None
    source = sr.Microphone(sample_rate=16000, device_index=mic_index) if mic_index is not None else sr.Microphone(sample_rate=16000)

    #launch recorder, change its leveling to static, declared in --energy_threshold, then adjust for ambience noise based on microphone input.
    recorder = sr.Recognizer()
    recorder.energy_threshold, recorder.dynamic_energy_threshold = args.energy_threshold, False
    with source:
        recorder.adjust_for_ambient_noise(source)

    whisper_model = whisper.load_model(args.model, args.device)

    # initialize variables and datastructures.
    phrase_time = None
    data_queue = Queue()
    transcription = ['']

    def record_callback(recognizer: sr.Recognizer, audio: sr.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data_queue.put(audio.get_raw_data())

    # Create a background thread that will pass us raw audio bytes.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=args.record_timeout)

    # clear up the interface, list all given arguments and indicate the user that the program is ready
    os.system('cls' if os.name == 'nt' else 'clear')
    print("\n".join([f"{key}: {value}" for key, value in vars(args).items()]))
    print("Model loaded, start transcribing now.\n")

    while True:
        try:
            # Check if the timeout for the phrase has been exceeded and update the timestamp for the next check.
            # "set phrase_complete" to arbitrary data which makes it no longer false if condition is fulfilled.
            now = datetime.utcnow()
            phrase_complete = phrase_time and now - phrase_time > timedelta(seconds=args.phrase_timeout)
            phrase_time = now

            # Combine audio data from queue
            audio_data = b''.join(data_queue.queue)
            data_queue.queue.clear()

            # Convert raw audio bits to NumPy array
            audio_np = (np.frombuffer(audio_data, dtype=np.int16) / 32768.0).astype(np.float32)

            # NumPy array gets transcribed and return dictionary saved as <text>
            result = whisper_model.transcribe(audio_np)
            text = result['text'].strip()

            # text gets added to the transcription if phrase is completed
            if phrase_complete:
                transcription.append(text)
            else:
                transcription[-1] = text

            # print to terminal and log
            if text=="":
                sleep(0.25)
                continue

            print(text)
            with open(log_path, 'a') as file:
                file.write(f"{text}\n")
            transcription = [text]

            # sleep to not overly use CPU
            sleep(0.25)

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()