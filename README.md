# Vocal Mix Assistant

Vocal Mix Assistant is a local Streamlit prototype for reviewing vocal mixes. It combines reproducible audio measurements with optional, experimental AI-generated descriptions to help explain analysis results.

## What it does

- Loads audio for local analysis and compares a vocal with an instrumental when provided.
- Presents measurements related to loudness, dynamics, level balance, and frequency-band energy.
- Can capture audio routed from Logic Pro using a virtual audio device such as BlackHole.
- Offers optional MOSS-Music analysis on supported Apple Silicon Macs. AI descriptions are experimental and are kept distinct from measured values.

The analysis flow is designed to process audio locally. The optional MOSS model must be downloaded separately and runs on the user's machine. Do not commit audio recordings, generated session data, model weights, or personal listening material to this repository.

## Requirements

- Python 3.11 or newer (the app has been developed with Python 3.12).
- FFmpeg, available through the Python `imageio-ffmpeg` dependency.
- For optional MOSS-Music support: a compatible Apple Silicon Mac and the separately installed MLX/MOSS model resources. This optional feature is experimental.

## Run the app

Create and activate a virtual environment, install dependencies, then start Streamlit:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local address printed by Streamlit. Audio-device capture requires a configured input device and may need macOS microphone permission.

## MOSS-Music and third-party resources

MOSS-Music support is optional. The model weights are not included in this repository; review the model card and its license before downloading or using a model. The official OpenMOSS MOSS-Music model card identifies Apache-2.0 licensing. The MLX community conversion also identifies Apache-2.0. The MLX-Audio project identifies an MIT license. These licenses apply to those upstream projects and resources, not automatically to this application.

- [OpenMOSS MOSS-Music model card](https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Thinking)
- [MLX community MOSS-Music conversion](https://huggingface.co/mlx-community/MOSS-Music-8B-Thinking-8bit)
- [MLX-Audio project](https://github.com/Neywa-Labs/mlx-audio)

## License

No license has been assigned to this application yet. Unless the author adds a license, others do not receive permission to reuse, modify, or redistribute this project's code.
