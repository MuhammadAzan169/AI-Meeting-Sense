# AI-Meeting-Sense

Local tooling for meeting intelligence:

- **Attendance** via face recognition from a webcam.
- **Speaker diarization** from live mic or audio files (two variants: normal + “coach”).
- **Real-estate meeting summaries** (PDF + JSON) including optional RAG/FAISS knowledge-base grounding.

This repo is a collection of Python scripts (not a single packaged library). The primary entry points are:

- `Attendence.py`
- `Normal-Diarization.py`
- `Coach-Diarization.py`
- `Normal-RealState-Summary.py`
- `RealState-Summary.py`

## Requirements

- Windows (repo paths and font handling are Windows-friendly; other OSes may work).
- Python 3.10+ recommended.
- FFmpeg (recommended) for audio conversions used by `pydub`.

Some components are optional and only required for specific scripts:

- **Diarization**: `torch`, `torchaudio`, `speechbrain`, `transformers`, `pydub`, `sounddevice`, `scipy`, `sklearn`, `reportlab`.
- **RAG summaries**: `sentence-transformers`, `faiss-cpu`, `python-docx`.
- **Attendance**: `opencv-python`, `insightface`, `keyboard`, `pandas`.

## Quick start

### 1) Create a virtual environment

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2) Install dependencies

There is no pinned `requirements.txt` in this repo. Install per feature:

```powershell
# Core
pip install python-dotenv requests numpy pydantic reportlab

# RAG (optional)
pip install sentence-transformers faiss-cpu python-docx

# Diarization (optional)
pip install torch torchaudio speechbrain transformers pydub soundfile sounddevice scipy scikit-learn

# Attendance (optional)
pip install opencv-python insightface pandas keyboard
```

Notes:

- `torch/torchaudio` installation varies by CUDA/CPU. If you run into install issues, install from PyTorch’s official instructions.
- If `pydub` fails to load audio, install FFmpeg and ensure it’s on PATH.

### 3) Configure environment variables

Copy `.env.example` to `.env` and fill values:

```powershell
copy .env.example .env
```

## Configuration

### `meeting_config.ini`

`meeting_config.ini` contains default paths used by some summary scripts:

- `transcript_path`: default transcript JSON (example in repo: `conversation.json`)
- `summaries_folder`: output folder (default: `RealEstateMeetingRecords`)

### Voice samples

The diarization scripts use voice sample folders:

- Normal diarization defaults to `voices/` (or set `VOICE_SAMPLES_DIR`).
- Coach diarization defaults to `Coach Voice/`.

### Summary Service (optional)

`Normal-Summary.py` is a FastAPI service that can use Firebase to fetch user settings.

- Set `FIREBASE_CREDENTIALS_PATH` to your service-account JSON.
- Optional server settings can be configured via env vars (see `.env.example`).

## Usage

### Attendance (face recognition)

1) Add reference images under `Faces/<PersonName>/...` (jpg/png).
2) Run:

```powershell
python Attendence.py
```

The script:

- Opens the webcam (`VIDEO_SOURCE = 0` by default).
- Tracks who is present during sessions.
- Writes session records to `attendance_sessions.csv`.

Controls are implemented via the `keyboard` library (see the script output for prompts).

### Speaker diarization (Normal)

```powershell
python Normal-Diarization.py
```

What it does (high-level):

- Splits audio into segments, clusters speakers, and tries to match known speakers from `voices/`.
- Produces transcript artifacts and a **PDF** with a table layout.
- Writes outputs under `output/`.

Environment knobs:

- `VOICE_SAMPLES_DIR` (optional): override `voices/`.

### Speaker diarization (Coach)

```powershell
python Coach-Diarization.py
```

This variant is designed for “coach vs client” flows:

- Uses a single voice sample (from `Coach Voice/`) to label matching segments.
- Any other speaker becomes `Client`.

Environment knobs:

- `SIMILARITY_THRESHOLD` (optional): cosine similarity threshold for accepting a speaker match (default `0.5`).

### Real-estate meeting summary (Normal)

```powershell
python Normal-RealState-Summary.py
```

This script generates professional PDFs and structured JSON from a transcript JSON (and optional knowledge base).

Inputs:

- Transcript JSON (example: `conversation.json`)
- Optional knowledge base (Docx + FAISS index files)

Outputs:

- PDF reports and JSON saved under `RealEstateMeetingRecords/` (and/or per-meeting folders).

LLM providers:

- Supports `openai`, `openrouter`, or `custom` (see `.env.example`).

### Real-estate meeting summary (Extended)

```powershell
python RealState-Summary.py
```

This is a more feature-rich variant (multiple PDFs + additional JSON artifacts).

## Data formats

### Transcript JSON

The summarizers expect a JSON structure similar to:

```json
{
  "agenda": "...",
  "transcripts": [
    {
      "speaker": "Speaker 1",
      "transcript": "Text...",
      "start": 0.0,
      "end": 4.2
    }
  ],
  "summary_info": {}
}
```

Field names vary slightly between scripts (some use `speaker_id` / `speaker_name`). If you have schema mismatches, open the script you’re running and align your JSON keys.

## Outputs & folders

- `output/`: diarization outputs (JSON/PDF).
- `temp/`: intermediate audio chunks.
- `model_cache/`: cached ML models/checkpoints.
- `RealEstateMeetingRecords/`: generated meeting summaries.
- `voices/`, `Coach Voice/`: voice samples for speaker identification.
- `Faces/`: face images grouped by person name.

## Troubleshooting

- **FAISS not available**: install `faiss-cpu` (Windows supported via pip for many Python versions).
- **`sentence-transformers` missing**: install it only if using RAG features.
- **Audio decode errors**: install FFmpeg and ensure it’s in PATH.
- **`sounddevice` errors**: check microphone permissions and installed audio drivers.
- **Torch install issues**: follow PyTorch official install command for your CUDA/CPU.

## Running the Summary API (optional)

```powershell
python Normal-Summary.py
```

Defaults:

- Runs a FastAPI app (port is controlled by `SUMMARY_SERVICE_PORT`, default in code: `8014`).
- Optional HTTPS via `SSL_KEYFILE` / `SSL_CERTFILE`.

## Safety & privacy

This project may process voice, face images, and meeting transcripts.

- Avoid committing sensitive recordings/keys to git.
- Keep `.env` private.
