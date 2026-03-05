#!/usr/bin/env python3
"""
Standalone diarization script using voice samples for speaker identification.
Outputs JSON and PDF reports. All non‑known speakers are labelled "Client".
"""

import os
import sys
import json
import warnings
import numpy as np
import torch
import torchaudio
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import scipy.signal as signal
from fpdf import FPDF                      # for PDF generation

# ReportLab for professional PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER, TA_LEFT
# Whisper for transcription
from transformers import pipeline

# SpeechBrain imports
import speechbrain as sb
from speechbrain.inference.speaker import SpeakerRecognition

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# 1. Model loading (cached)
# ----------------------------------------------------------------------
_speaker_model = None

def load_speaker_model():
    global _speaker_model
    if _speaker_model is not None:
        return _speaker_model
    print("Loading ECAPA‑TDNN speaker model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _speaker_model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="model_cache/speechbrain_ecapa",
        run_opts={"device": device}
    )
    print(f"Model loaded on {device.upper()}")
    return _speaker_model


# ----------------------------------------------------------------------
# 2. Voice sample embedding extraction
# ----------------------------------------------------------------------
def load_voice_sample_embedding(model, sample_path):
    """Load a single voice sample and return its normalised embedding."""
    try:
        waveform, sr = torchaudio.load(sample_path)
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Resample to 16 kHz (model expects 16k)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            sr = 16000
        # Ensure at least 3 seconds of audio
        min_len = int(3.0 * sr)
        if waveform.shape[1] < min_len:
            pad = min_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        # Use a 3‑second chunk from the middle
        total = waveform.shape[1]
        start = (total - min_len) // 2
        chunk = waveform[:, start:start + min_len]

        with torch.no_grad():
            if torch.cuda.is_available():
                chunk = chunk.to('cuda')
            emb = model.encode_batch(chunk)
            if isinstance(emb, tuple):
                emb = emb[0]
            emb = emb.squeeze().cpu().numpy()
        # Normalise
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb
    except Exception as e:
        print(f"Error processing {sample_path}: {e}")
        return None


def load_all_voice_samples(folder):
    """
    Load all .wav files from folder, return dict {speaker_name: embedding}.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Voice samples folder not found: {folder}")
    model = load_speaker_model()
    samples = {}
    # Load all .wav files directly from the folder, use filename (without extension) as speaker name
    wav_files = list(folder.glob("*.wav"))
    if not wav_files:
        raise RuntimeError("No .wav voice samples found in folder.")
    for wav_file in wav_files:
        speaker_name = wav_file.stem  # filename without extension
        emb = load_voice_sample_embedding(model, wav_file)
        if emb is not None:
            samples[speaker_name] = emb
            print(f"Loaded voice sample for '{speaker_name}' from {wav_file}")
    if not samples:
        raise RuntimeError("No valid voice samples found in folder.")
    return samples


# ----------------------------------------------------------------------
# 3. Audio preprocessing and segmentation
# ----------------------------------------------------------------------
def preprocess_audio(waveform, sr, target_sr=16000):
    """Convert to mono, resample, apply high‑pass filter, normalise."""
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr
    audio_np = waveform.squeeze().numpy()
    # High‑pass filter to remove low‑frequency noise
    sos = signal.butter(4, 80, 'hp', fs=sr, output='sos')
    audio_np = signal.sosfilt(sos, audio_np)
    # Normalise RMS
    rms = np.sqrt(np.mean(audio_np**2))
    if rms > 0:
        audio_np = audio_np / rms * 0.1
    audio_np = np.clip(audio_np, -1.0, 1.0)
    return torch.tensor(audio_np).unsqueeze(0), sr


def segment_audio(wav_path, segment_duration=2.0, overlap=0.5):
    """
    Load audio, preprocess, and split into overlapping segments.
    Returns: list of segment tensors, list of (start, end) timestamps.
    """
    waveform, sr = torchaudio.load(wav_path)
    waveform, sr = preprocess_audio(waveform, sr)

    seg_len = int(segment_duration * sr)
    hop_len = int((segment_duration - overlap) * sr)
    total_len = waveform.shape[1]

    segments = []
    timestamps = []
    for start in range(0, total_len - seg_len, hop_len):
        end = start + seg_len
        segments.append(waveform[:, start:end])
        timestamps.append((start / sr, end / sr))
    print(f"Created {len(segments)} segments of {segment_duration}s (hop {segment_duration-overlap:.2f}s)")
    return segments, timestamps, sr


def extract_embeddings(model, segments):
    """Compute embeddings for each segment."""
    embeddings = []
    for i, seg in enumerate(segments):
        with torch.no_grad():
            if torch.cuda.is_available():
                seg = seg.to('cuda')
            emb = model.encode_batch(seg)
            if isinstance(emb, tuple):
                emb = emb[0]
            emb = emb.squeeze().cpu().numpy()
            embeddings.append(emb)
        if torch.cuda.is_available() and (i+1) % 10 == 0:
            torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.array(embeddings)


# ----------------------------------------------------------------------
# 4. Clustering and diarization
# ----------------------------------------------------------------------
def estimate_n_speakers(embeddings, max_speakers=8):
    """Use silhouette score to estimate best number of clusters (≥2)."""
    valid = embeddings[np.any(embeddings, axis=1)]  # remove zero vectors
    if len(valid) < 2:
        return 1
    best_k = 2
    best_score = -1
    for k in range(2, min(max_speakers, len(valid))):
        clustering = AgglomerativeClustering(n_clusters=k)
        labels = clustering.fit_predict(valid)
        if len(set(labels)) == 1:
            continue
        score = silhouette_score(valid, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def diarize_segments(embeddings, timestamps):
    """
    Cluster embeddings and return list of (cluster_id, start, end) for each segment.
    """
    valid_idx = [i for i, e in enumerate(embeddings) if np.any(e)]
    valid_embs = embeddings[valid_idx]
    valid_ts = [timestamps[i] for i in valid_idx]

    if len(valid_embs) < 2:
        # Only one speaker
        return [(0, ts[0][0], ts[0][1]) for ts in valid_ts] if valid_ts else []

    n_speakers = estimate_n_speakers(valid_embs)
    clustering = AgglomerativeClustering(n_clusters=n_speakers)
    labels = clustering.fit_predict(valid_embs)

    # Re‑map indices back to original order
    result = []
    for i, idx in enumerate(valid_idx):
        result.append((labels[i], valid_ts[i][0], valid_ts[i][1]))
    return result


# ----------------------------------------------------------------------
# 5. Matching clusters to known speakers
# ----------------------------------------------------------------------
def get_cluster_embedding(embeddings, diarization, cluster_id):
    """Average embeddings of all segments belonging to a given cluster."""
    embs = []
    for i, (spk, _, _) in enumerate(diarization):
        if spk == cluster_id and i < len(embeddings):
            emb = embeddings[i]
            if np.any(emb):
                embs.append(emb)
    if not embs:
        return None
    mean_emb = np.mean(embs, axis=0)
    return mean_emb / (np.linalg.norm(mean_emb) + 1e-8)


def assign_speakers(diarization, embeddings, known_samples, threshold=0.5):
    """
    For each unique cluster, find the best matching known speaker by cosine similarity.
    Returns dict {cluster_id: speaker_name}.
    """
    unique_clusters = set(spk for spk, _, _ in diarization)
    cluster_to_speaker = {}

    for cluster in unique_clusters:
        cluster_emb = get_cluster_embedding(embeddings, diarization, cluster)
        if cluster_emb is None:
            cluster_to_speaker[cluster] = "Unknown"
            continue

        best_name = "Unknown"
        best_sim = -1.0
        for name, known_emb in known_samples.items():
            sim = float(np.dot(cluster_emb, known_emb))  # cosine similarity (normalised)
            if sim > best_sim:
                best_sim = sim
                best_name = name

        if best_sim < threshold:
            best_name = "Unknown"
        cluster_to_speaker[cluster] = best_name
        print(f"Cluster {cluster} -> {best_name} (sim={best_sim:.3f})")
    return cluster_to_speaker


# ----------------------------------------------------------------------
# 6. Merging segments by speaker name
# ----------------------------------------------------------------------
def merge_by_speaker(segments, tolerance=0.5):
    """
    Merge consecutive segments belonging to the same speaker.
    segments: list of (speaker_name, start, end)
    Returns merged list with same format.
    """
    if not segments:
        return []
    segments.sort(key=lambda x: x[1])  # sort by start time
    merged = []
    cur_spk, cur_start, cur_end = segments[0]
    for spk, start, end in segments[1:]:
        if spk == cur_spk and start - cur_end <= tolerance:
            cur_end = end
        else:
            merged.append((cur_spk, cur_start, cur_end))
            cur_spk, cur_start, cur_end = spk, start, end
    merged.append((cur_spk, cur_start, cur_end))
    return merged


# ----------------------------------------------------------------------
# 7. PDF generation
# ----------------------------------------------------------------------
def generate_pdf(result, output_path):
    """Create a simple PDF report with a table of diarization segments."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Title
    pdf.cell(200, 10, txt="Diarization Report", ln=True, align='C')
    pdf.ln(10)

    # Table header
    pdf.set_font("Arial", style='B', size=10)
    col_width = [40, 40, 40, 40, 30]  # widths for each column
    headers = ["Speaker", "Start (s)", "End (s)", "Duration (s)", ""]
    for i, header in enumerate(headers):
        pdf.cell(col_width[i], 10, header, border=1)
    pdf.ln()

    # Table rows
    pdf.set_font("Arial", size=10)
    for seg in result:
        pdf.cell(col_width[0], 8, seg["speaker"], border=1)
        pdf.cell(col_width[1], 8, f"{seg['start']:.2f}", border=1)
        pdf.cell(col_width[2], 8, f"{seg['end']:.2f}", border=1)
        pdf.cell(col_width[3], 8, f"{seg['duration']:.2f}", border=1)
        pdf.cell(col_width[4], 8, "", border=1)  # empty cell
        pdf.ln()

    # Output
    pdf.output(output_path)
    print(f"PDF report saved to: {output_path}")


# ----------------------------------------------------------------------
# 8. Main pipeline
# ----------------------------------------------------------------------
def diarize_with_voice_samples(audio_path, voice_folder, output_json=None):
    """
    Run full diarization and speaker identification.
    Saves result as JSON and PDF.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # 1. Load known voice samples
    print(f"\nLoading voice samples from: {voice_folder}")
    known_samples = load_all_voice_samples(voice_folder)

    # 2. Load speaker model
    model = load_speaker_model()

    # 3. Segment audio and extract embeddings
    print(f"\nProcessing audio: {audio_path}")
    segments, timestamps, sr = segment_audio(audio_path)
    embeddings = extract_embeddings(model, segments)

    # 4. Diarize (cluster) without prior knowledge
    diar = diarize_segments(embeddings, timestamps)   # list of (cluster_id, start, end)

    # 5. Match clusters to known speakers
    cluster_map = assign_speakers(diar, embeddings, known_samples, threshold=0.5)

    # ----- Rename all "Unknown" to "Client" -----
    for cluster in cluster_map:
        if cluster_map[cluster] == "Unknown":
            cluster_map[cluster] = "Client"

    # Convert cluster-based diarization to speaker-name-based list
    diar_with_names = [(cluster_map[spk], start, end) for spk, start, end in diar]

    # 6. Merge consecutive segments with same speaker name
    merged = merge_by_speaker(diar_with_names, tolerance=0.5)

    # 7. Build final segment list with speaker names
    result = []
    for spk, start, end in merged:
        result.append({
            "speaker": spk,
            "start": round(start, 2),
            "end": round(end, 2),
            "duration": round(end - start, 2),
            "transcript": ""   # placeholder for future transcription
        })

    # 8. Save outputs in 'Coach diarization' folder
    output_dir = Path("Coach diarization")
    output_dir.mkdir(exist_ok=True)

    # JSON file
    json_name = f"{audio_path.stem}.diarized.json"
    json_path = output_dir / json_name
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Transcript text file (simple)
    transcript_name = f"{audio_path.stem}.transcript.txt"
    transcript_path = output_dir / transcript_name
    with open(transcript_path, "w", encoding="utf-8") as f:
        for seg in result:
            f.write(f"[{seg['start']:.2f}-{seg['end']:.2f}] {seg['speaker']}: {seg['transcript']}\n")

    # PDF report
    pdf_name = f"{audio_path.stem}.diarized.pdf"
    pdf_path = output_dir / pdf_name
    generate_pdf(result, pdf_path)

    print(f"\nDiarization complete. Found {len(set(c['speaker'] for c in result))} speakers.")
    print(f"JSON saved to: {json_path}")
    print(f"Transcript saved to: {transcript_path}")
    print(f"PDF saved to: {pdf_path}")
    return result


# ----------------------------------------------------------------------
# 9. Command‑line entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Coach Diarization Script (JSON + PDF output)")
    audio_file = input("Enter path to meeting audio file: ").strip()
    voice_folder = "Coach Voice"  # fixed folder for voice samples

    try:
        diarize_with_voice_samples(audio_file, voice_folder)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)