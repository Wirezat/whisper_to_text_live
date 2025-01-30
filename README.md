# whisper_to_text_live
Thanks to [davabase](https://github.com/davabase/whisper_real_time) for the inspiration and first chain of thought for this project!
***
## Installation:
Install all requirements with
````pip install -r requirements.txt````.

make sure to also install ffmpeg

***
## Usage:
use the program as 
`python3 .\transcoder.py --parameters` although the parameters are more optional

parameters:
 - --model: default: turbo, choices: "tiny", "base", "small", "medium", "large", "turbo"
   - Model to use for speech recognition.
  - --energy_threshold: default: 1000, type: int
    - Energy level for the microphone to detect speech.
  - --record_timeout: default: 10, type: float
    - Duration of each recording segment in seconds.
  - --phrase_timeout: default: 2, type: float
    - Silence duration before considering a new transcription line.
  - --device: default: "cpu", choices: "cpu", "cuda"
    - Device to run the model on (CPU or CUDA).
  - --default_microphone: default: default_mic, type: str
    - Default microphone name. Use 'list' to show available microphones.
  - --log_folder: default: ".", type: str
    - Folder, where the output file should be generated.
