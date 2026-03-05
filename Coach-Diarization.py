# diarization.py
"""
live+audio ask from user no minute of meeting batch processing
WITH AUTOMATIC SPEAKER IDENTIFICATION FROM A SINGLE VOICE SAMPLE
Segments matching the sample are named after it; all others become "Client".
PDF generation with 4‑column table.
"""
import os
import json
import torch
import torchaudio
import numpy as np
from pydub import AudioSegment
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import speechbrain
from speechbrain.inference import SpeakerRecognition
from transformers import pipeline
import soundfile as sf
import sounddevice as sd
import queue
import scipy.signal as signal
import warnings
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
import textwrap
import platform
warnings.filterwarnings("ignore")

# ---------------------------------------------------
# 🆕 Configuration Module
# ---------------------------------------------------

class Config:
    """Centralized configuration for all paths and settings"""
    
    @staticmethod
    def get_base_dir():
        """Get the base directory where the script is running"""
        return os.path.dirname(os.path.abspath(__file__))
    
    @staticmethod
    def get_voice_samples_dir():
        """Get voice samples directory - configurable via environment variable or default"""
        env_dir = os.getenv("VOICE_SAMPLES_DIR")
        if env_dir and os.path.exists(env_dir):
            return env_dir
        
        base_dir = Config.get_base_dir()
        coach_voice_dir = os.path.join(base_dir, "Coach Voice")
        
        if not os.path.exists(coach_voice_dir):
            os.makedirs(coach_voice_dir, exist_ok=True)
            print(f"\U0001F4C1 Created Coach Voice directory: {coach_voice_dir}")
        
        return coach_voice_dir
    
    @staticmethod
    def get_output_dir():
        """Get output directory for generated files"""
        base_dir = Config.get_base_dir()
        output_dir = os.path.join(base_dir, "output")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        return output_dir
    
    @staticmethod
    def get_temp_dir():
        """Get temporary directory for intermediate files"""
        base_dir = Config.get_base_dir()
        temp_dir = os.path.join(base_dir, "temp")
        
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)
        
        return temp_dir
    
    @staticmethod
    def get_model_cache_dir():
        """Get directory for caching models"""
        base_dir = Config.get_base_dir()
        model_cache_dir = os.path.join(base_dir, "model_cache")
        
        if not os.path.exists(model_cache_dir):
            os.makedirs(model_cache_dir, exist_ok=True)
        
        return model_cache_dir

# ---------------------------------------------------
# 🆕 Similarity threshold for accepting a speaker match
# ---------------------------------------------------
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
# Cosine similarity above this value → known speaker, below → Client

# ---------------------------------------------------
# 🆕 PDF Generation Module with Table Layout
# ---------------------------------------------------

class MeetingTranscriptPDF:
    """Professional PDF generator for meeting transcripts with table layout"""
    
    def __init__(self, title="Meeting Transcript"):
        self.title = title
        self.styles = getSampleStyleSheet()
        self._register_fonts()
        self._define_custom_styles()
        
    def _register_fonts(self):
        """Register custom fonts for professional appearance - platform independent"""
        try:
            system = platform.system()
            
            font_paths = []
            if system == "Windows":
                windows_font_dir = os.getenv("WINDIR", "C:\\Windows")
                font_dir = os.path.join(windows_font_dir, "Fonts")
                font_paths.extend([
                    os.path.join(font_dir, "arial.ttf"),
                    os.path.join(font_dir, "arialbd.ttf"),
                    os.path.join(font_dir, "ariali.ttf"),
                    os.path.join(font_dir, "calibri.ttf"),
                ])
            elif system == "Darwin":  # macOS
                font_paths.extend([
                    "/System/Library/Fonts/Helvetica.ttc",
                    "/System/Library/Fonts/Arial.ttf",
                ])
            elif system == "Linux":
                font_paths.extend([
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                ])
            
            registered = False
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        if font_path.endswith('.ttc'):
                            pdfmetrics.registerFont(TTFont('Arial', font_path, subfontIndex=0))
                        else:
                            pdfmetrics.registerFont(TTFont('Arial', font_path))
                        
                        bold_path = font_path.replace('.ttf', 'bd.ttf').replace('.ttc', 'bd.ttf')
                        if os.path.exists(bold_path):
                            pdfmetrics.registerFont(TTFont('Arial-Bold', bold_path))
                        
                        registered = True
                        print(f"✅ Registered font: {font_path}")
                        break
                    except Exception as e:
                        print(f"⚠ Could not register font {font_path}: {e}")
                        continue
            
            if not registered:
                print("⚠ Using default fonts (custom fonts not available)")
                try:
                    pdfmetrics.registerFont(TTFont('Arial', 'Helvetica'))
                    pdfmetrics.registerFont(TTFont('Arial-Bold', 'Helvetica-Bold'))
                except:
                    pass
                    
        except Exception as e:
            print(f"⚠ Font registration warning: {e}")
    
    def _define_custom_styles(self):
        """Define professional color theme and styles"""
        self.dark_blue = colors.HexColor('#1E3A8A')
        self.light_blue = colors.HexColor('#60A5FA')
        self.white = colors.white
        
        self.styles.add(ParagraphStyle(
            name='MeetingTitle',
            parent=self.styles['Title'],
            fontSize=24,
            textColor=self.dark_blue,
            alignment=TA_CENTER,
            spaceAfter=12,
            fontName='Arial-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='MeetingSubtitle',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=self.dark_blue,
            alignment=TA_CENTER,
            spaceAfter=20,
            fontName='Arial'
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=self.dark_blue,
            alignment=TA_CENTER,
            spaceAfter=12,
            spaceBefore=20,
            fontName='Arial-Bold',
            backgroundColor=self.light_blue
        ))
        
        self.styles.add(ParagraphStyle(
            name='TableHeader',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.white,
            alignment=TA_CENTER,
            fontName='Arial-Bold',
            spaceAfter=6,
            spaceBefore=6
        ))
        
        self.styles.add(ParagraphStyle(
            name='TableCell',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.black,
            alignment=TA_LEFT,
            fontName='Arial',
            spaceAfter=4,
            spaceBefore=4
        ))
        
        self.styles.add(ParagraphStyle(
            name='TableSpeaker',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=self.dark_blue,
            alignment=TA_LEFT,
            fontName='Arial-Bold',
            spaceAfter=4,
            spaceBefore=4
        ))
        
        self.styles.add(ParagraphStyle(
            name='TableTimestamp',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#6B7280'),
            alignment=TA_CENTER,
            fontName='Arial',
            spaceAfter=4,
            spaceBefore=4
        ))
        
        self.styles.add(ParagraphStyle(
            name='SummaryText',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.black,
            alignment=TA_LEFT,
            fontName='Arial',
            leftIndent=20,
            spaceAfter=6
        ))
    
    def format_timestamp(self, seconds):
        """Convert seconds to MM:SS format"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def wrap_text_for_table(self, text, max_chars=80):
        """Wrap long text for table cells"""
        if len(text) <= max_chars:
            return text
        
        sentences = text.split('. ')
        wrapped_lines = []
        current_line = ""
        
        for sentence in sentences:
            if sentence:
                sentence = sentence.strip()
                if not sentence.endswith('.'):
                    sentence += '.'
                
                if len(current_line) + len(sentence) + 2 <= max_chars:
                    if current_line:
                        current_line += " " + sentence
                    else:
                        current_line = sentence
                else:
                    if current_line:
                        wrapped_lines.append(current_line)
                    current_line = sentence
        
        if current_line:
            wrapped_lines.append(current_line)
        
        return '\n'.join(wrapped_lines)
    
    def create_pdf(self, transcript_data, output_path=None, 
                   meeting_title=None, meeting_date=None, agenda_items=None):
        """
        Create professional PDF transcript with 4-column table layout
        """
        if meeting_title is None:
            meeting_title = self.title
        
        if meeting_date is None:
            meeting_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        
        if output_path is None:
            output_dir = Config.get_output_dir()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"meeting_transcript_{timestamp}.pdf")
        
        print(f"\n📄 Generating professional PDF transcript...")
        print(f"   Output: {output_path}")
        
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch
        )
        
        story = []
        
        # 1. Title Section
        story.append(Paragraph(meeting_title, self.styles['MeetingTitle']))
        story.append(Paragraph(f"Date & Time: {meeting_date}", self.styles['MeetingSubtitle']))
        story.append(Spacer(1, 15))
        
        # 2. Optional Agenda Section
        if agenda_items and len(agenda_items) > 0:
            story.append(Paragraph("Meeting Agenda", self.styles['SectionHeader']))
            story.append(Spacer(1, 8))
            
            for i, item in enumerate(agenda_items, 1):
                agenda_text = f"<bullet>&bull;</bullet> {item}"
                story.append(Paragraph(agenda_text, self.styles['SummaryText']))
            
            story.append(Spacer(1, 15))
        
        # 3. Transcript Table Header
        story.append(Paragraph("Meeting Transcript", self.styles['SectionHeader']))
        story.append(Spacer(1, 10))
        
        # 4. Create 4-column table data
        table_data = []
        
        header_row = [
            Paragraph("Speaker", self.styles['TableHeader']),
            Paragraph("Start Time", self.styles['TableHeader']),
            Paragraph("End Time", self.styles['TableHeader']),
            Paragraph("Transcript", self.styles['TableHeader'])
        ]
        table_data.append(header_row)
        
        for segment in transcript_data:
            speaker = segment.get('speaker', 'Unknown Speaker')
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)
            transcript = segment.get('transcript', '').strip()
            
            if not transcript:
                continue
            
            start_str = self.format_timestamp(start_time)
            end_str = self.format_timestamp(end_time)
            
            wrapped_transcript = self.wrap_text_for_table(transcript, max_chars=120)
            
            row = [
                Paragraph(speaker, self.styles['TableSpeaker']),
                Paragraph(start_str, self.styles['TableTimestamp']),
                Paragraph(end_str, self.styles['TableTimestamp']),
                Paragraph(wrapped_transcript, self.styles['TableCell'])
            ]
            table_data.append(row)
        
        # 5. Create the table
        col_widths = [1.2*inch, 0.8*inch, 0.8*inch, 4.2*inch]
        transcript_table = Table(table_data, colWidths=col_widths, repeatRows=1)
        
        transcript_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.dark_blue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Arial-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 0), (-1, 0), 8),
            
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (0, 1), (2, -1), 'CENTER'),
            ('ALIGN', (3, 1), (3, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('FONTNAME', (0, 1), (-1, -1), 'Arial'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),   # <-- FIXED: colours → colors
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
            
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
            
            ('ALIGNMENT', (0, 0), (-1, -1), 'CENTER'),
        ]))
        
        transcript_table.hAlign = 'CENTER'
        
        story.append(transcript_table)
        story.append(Spacer(1, 20))
        
        # 6. Summary Section
        story.append(PageBreak())
        story.append(Paragraph("Meeting Summary", self.styles['SectionHeader']))
        story.append(Spacer(1, 15))
        
        total_duration = sum(seg.get('duration', 0) for seg in transcript_data)
        speakers = set(seg.get('speaker', 'Unknown') for seg in transcript_data)
        
        summary_data = [
            ["Total Duration:", f"{total_duration:.1f} seconds ({total_duration/60:.1f} minutes)"],
            ["Number of Speakers:", str(len(speakers))],
            ["Number of Segments:", str(len(transcript_data))],
            ["Speakers Present:", ", ".join(sorted(speakers))],
        ]
        
        summary_table = Table(summary_data, colWidths=[2*inch, 4*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), self.light_blue),
            ('BACKGROUND', (1, 0), (1, -1), colors.HexColor('#F3F4F6')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Arial-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Arial'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),   # already correct
        ]))
        
        summary_table.hAlign = 'CENTER'
        
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # 7. Speaker Statistics
        if len(speakers) > 1:
            story.append(Paragraph("Speaker Statistics", self.styles['SectionHeader']))
            story.append(Spacer(1, 10))
            
            speaker_stats = {}
            for segment in transcript_data:
                speaker = segment['speaker']
                duration = segment['duration']
                speaker_stats[speaker] = speaker_stats.get(speaker, 0) + duration
            
            stats_data = [["Speaker", "Speaking Time (seconds)", "Percentage"]]
            for speaker, duration in sorted(speaker_stats.items(), key=lambda x: x[1], reverse=True):
                percentage = (duration / total_duration) * 100
                stats_data.append([
                    speaker,
                    f"{duration:.1f}s",
                    f"{percentage:.1f}%"
                ])
            
            stats_table = Table(stats_data, colWidths=[2*inch, 2*inch, 2*inch])
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), self.dark_blue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Arial-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Arial'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),   # <-- FIXED: colours → colors
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('ROWBACKGROUNDS', (1, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
            ]))
            
            stats_table.hAlign = 'CENTER'
            
            story.append(stats_table)
        
        try:
            doc.build(story)
            print(f"✅ PDF successfully generated: {output_path}")
            return output_path
        except Exception as e:
            print(f"❌ PDF generation failed: {e}")
            return None

# ---------------------------------------------------
# ⿠ LIVE STREAMING ADD-ON
# ---------------------------------------------------

q = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(indata.copy())

def stream_audio(duration_sec, sr=16000, blocksize=1024):
    """
    Streams microphone audio for <duration_sec> seconds and saves to live_stream.wav
    """
    print(f"\n🎤 Recording from microphone for {duration_sec} seconds…")
    print("👉 Start speaking now...\n")

    frames = []
    total_blocks = int(sr / blocksize * duration_sec)

    with sd.InputStream(callback=audio_callback, channels=1, samplerate=sr, blocksize=blocksize):
        for _ in range(total_blocks):
            frame = q.get()
            frames.append(frame)

    audio_np = np.concatenate(frames, axis=0)

    temp_dir = Config.get_temp_dir()
    output_path = os.path.join(temp_dir, "live_stream.wav")
    sf.write(output_path, audio_np, sr)

    print(f"\n✅ Live audio saved as {output_path}\n")
    return output_path

# ---------------------------------------------------
# 🆕 Audio preprocessing for domain robustness (SPEAKER ID ONLY)
# ---------------------------------------------------
def preprocess_audio_for_speaker_id(waveform, sr, target_sr=16000):
    """
    Normalize and preprocess audio for speaker identification only:
    - Resample to target sample rate
    - Normalize volume
    - Apply voice activity detection (VAD) to extract speech portions
    - High-pass filter to remove low-frequency noise
    """
    if waveform.dim() > 1 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr
    
    audio_np = waveform.squeeze().numpy()
    
    sos = signal.butter(4, 80, 'hp', fs=sr, output='sos')
    audio_np = signal.sosfilt(sos, audio_np)
    
    rms = np.sqrt(np.mean(audio_np**2))
    if rms > 0:
        audio_np = audio_np / rms * 0.1
    
    audio_np = np.clip(audio_np, -1.0, 1.0)
    
    return torch.tensor(audio_np).unsqueeze(0), sr

# ---------------------------------------------------
# ⿡ Convert input audio to WAV
# ---------------------------------------------------
def convert_to_wav(audio_path):
    if not audio_path.endswith(".wav"):
        print("Converting input to WAV …")
        sound = AudioSegment.from_file(audio_path)
        
        temp_dir = Config.get_temp_dir()
        wav_filename = os.path.splitext(os.path.basename(audio_path))[0] + ".wav"
        wav_path = os.path.join(temp_dir, wav_filename)
        
        sound = sound.set_channels(1).set_frame_rate(16000)
        sound.export(wav_path, format="wav")
        return wav_path
    return audio_path

# ---------------------------------------------------
# ⿢ Load SpeechBrain ECAPA model with fallback
# ---------------------------------------------------
def load_embedding_model():
    print("Loading embedding model …")
    
    model_cache_dir = Config.get_model_cache_dir()
    
    model_sources = [
        "speechbrain/spkrec-ecapa-voxceleb",
        "speechbrain/spkrec-ecapa-voxceleb-v2.0",
        "speechbrain/spkrec-ecapa-voxceleb-v2",
    ]
    
    for source in model_sources:
        try:
            print(f"Trying to load model from: {source}")
            save_dir = os.path.join(model_cache_dir, source.replace('/', '_'))
            model = SpeakerRecognition.from_hparams(
                source=source,
                savedir=save_dir
            )
            print(f"✅ Successfully loaded model: {source}")
            return model
        except Exception as e:
            print(f"⚠ Failed to load {source}: {str(e)[:100]}")
            continue
    
    print("\n⚠ All online models failed. Trying local model...")
    try:
        local_model_path = os.path.join(model_cache_dir, "local_ecapa_model")
        if not os.path.exists(local_model_path):
            os.makedirs(local_model_path, exist_ok=True)
            print(f"Created local directory: {local_model_path}")
            print("NOTE: For better results, manually download a pretrained model")
        
        model = SpeakerRecognition.from_hparams(
            source=local_model_path,
            savedir=local_model_path
        )
        print("✅ Loaded local model")
        return model
    except Exception as e:
        print(f"❌ Could not load any speaker recognition model: {e}")
        print("\n💡 TROUBLESHOOTING:")
        print("1. Check your internet connection")
        print("2. Try: pip install --upgrade speechbrain huggingface_hub")
        print("3. Or manually download a model from HuggingFace")
        raise

# ---------------------------------------------------
# 🆕 Get AudioSamples directory path
# ---------------------------------------------------
def get_samples_directory():
    """Get voice samples directory using Config class"""
    return Config.get_voice_samples_dir()

# ---------------------------------------------------
# 🆕 SIMPLIFIED voice sample loading to avoid compute_features error
# ---------------------------------------------------
def load_voice_samples(model, samples_dir=None):
    """
    Load all voice samples with SIMPLIFIED preprocessing to avoid compute_features error
    """
    if samples_dir is None:
        samples_dir = get_samples_directory()
    
    print(f"\n📁 Loading voice samples from '{samples_dir}/' …")
    
    if not os.path.exists(samples_dir):
        print(f"⚠ Warning: '{samples_dir}/' directory not found. No voice samples loaded.")
        return {}
    
    voice_embeddings = {}
    
    for filename in os.listdir(samples_dir):
        if filename.lower().endswith(('.mp3', '.wav', '.m4a', '.flac')):
            speaker_name = os.path.splitext(filename)[0]
            sample_path = os.path.join(samples_dir, filename)
            
            try:
                wav_path = convert_to_wav(sample_path)
                waveform, sr = torchaudio.load(wav_path)
                
                if waveform.dim() > 1 and waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                
                if sr != 16000:
                    waveform = torchaudio.functional.resample(waveform, sr, 16000)
                    sr = 16000
                
                min_duration = 3.0
                min_samples = int(min_duration * sr)
                
                if waveform.shape[1] < min_samples:
                    print(f"  ⚠ {speaker_name}: audio too short (needs at least {min_duration}s)")
                    continue
                
                start_sample = max(0, (waveform.shape[1] - min_samples) // 2)
                end_sample = start_sample + min_samples
                chunk = waveform[:, start_sample:end_sample]
                
                with torch.no_grad():
                    emb = model.encode_batch(chunk)
                    if isinstance(emb, tuple):
                        emb = emb[0]
                    emb = emb.squeeze().cpu().numpy()
                    voice_embeddings[speaker_name] = emb
                    print(f"  ✅ Loaded: {speaker_name}")
                    
            except Exception as e:
                print(f"  ❌ Failed to load {filename}: {str(e)[:100]}...")
    
    print(f"\n📊 Total voice samples loaded: {len(voice_embeddings)}")
    return voice_embeddings

# ---------------------------------------------------
# 🆕 Enhanced speaker identification with score normalization
# ---------------------------------------------------
def identify_speaker(segment_embedding, voice_embeddings):
    """
    Compare segment embedding with all voice sample embeddings.
    Returns (best_name, best_score) for the closest sample.
    """
    if not voice_embeddings:
        return None, 0.0
    
    scores = {}
    
    for speaker_name, sample_embedding in voice_embeddings.items():
        seg_norm = segment_embedding / (np.linalg.norm(segment_embedding) + 1e-8)
        sample_norm = sample_embedding / (np.linalg.norm(sample_embedding) + 1e-8)
        
        score = np.dot(seg_norm, sample_norm)
        scores[speaker_name] = score
    
    best_match = max(scores, key=scores.get)
    best_score = scores[best_match]
    
    # Print all scores for debugging
    print(f"    🔍 Similarity scores:")
    for name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        marker = "👉" if name == best_match else "  "
        print(f"    {marker} {name}: {score:.4f}")
    
    return best_match, best_score

# ---------------------------------------------------
# ⿣ Enhanced audio segmentation with preprocessing (SPEAKER ID ONLY)
# ---------------------------------------------------
def segment_audio_for_speaker_id(wav_path, segment_duration=3.0, overlap=1.0):
    """
    Segment audio with preprocessing FOR SPEAKER IDENTIFICATION ONLY
    Using longer segments and less overlap for better accuracy
    """
    print("Segmenting audio for speaker identification…")
    waveform, sr = torchaudio.load(wav_path)
    
    waveform, sr = preprocess_audio_for_speaker_id(waveform, sr)

    segment_len = int(segment_duration * sr)
    hop_len = int((segment_duration - overlap) * sr)
    total_len = waveform.shape[1]

    segments, timestamps = [], []
    for start in range(0, total_len - segment_len, hop_len):
        end = start + segment_len
        segments.append(waveform[:, start:end])
        timestamps.append((start / sr, end / sr))

    print(f"Total segments created for speaker ID: {len(segments)}")
    return segments, timestamps, sr

# ---------------------------------------------------
# ⿤ Extract embeddings using ECAPA (WITH ERROR HANDLING)
# ---------------------------------------------------
def extract_embeddings(model, segments):
    print("Computing embeddings …")
    embeddings = []
    successful_segments = 0
    
    for i, segment in enumerate(segments):
        try:
            with torch.no_grad():
                emb = model.encode_batch(segment)
                if isinstance(emb, tuple):
                    emb = emb[0]
                emb = emb.squeeze().cpu().numpy()
                embeddings.append(emb)
                successful_segments += 1
        except Exception as e:
            print(f"  ⚠ Failed to extract embedding for segment {i}: {str(e)[:100]}...")
            if len(embeddings) > 0:
                embeddings.append(np.zeros_like(embeddings[0]))
            else:
                embeddings.append(np.zeros(192))
    
    embeddings = np.array(embeddings)
    print(f"Computed embeddings for {successful_segments}/{len(segments)} segments")
    return embeddings

# ---------------------------------------------------
# ⿥ Estimate number of speakers
# ---------------------------------------------------
def estimate_speakers(embeddings, max_speakers=8):
    valid_embeddings = []
    for emb in embeddings:
        if np.any(emb):
            valid_embeddings.append(emb)
    
    if len(valid_embeddings) < 2:
        print("⚠ Not enough valid embeddings for speaker estimation, defaulting to 1 speaker")
        return 1
    
    valid_embeddings = np.array(valid_embeddings)
    
    print("Estimating number of speakers …")
    best_k = 1
    best_score = -1

    for k in range(1, min(max_speakers, len(valid_embeddings))):
        clustering = AgglomerativeClustering(n_clusters=k)
        labels = clustering.fit_predict(valid_embeddings)
        if len(set(labels)) == 1:
            continue
        score = silhouette_score(valid_embeddings, labels)
        if score > best_score:
            best_score = score
            best_k = k

    print(f"Estimated number of speakers: {best_k}")
    return best_k

# ---------------------------------------------------
# ⿦ Clustering + diarization
# ---------------------------------------------------
def diarize(embeddings, timestamps):
    valid_indices = [i for i, emb in enumerate(embeddings) if np.any(emb)]
    valid_embeddings = embeddings[valid_indices]
    valid_timestamps = [timestamps[i] for i in valid_indices]
    
    if len(valid_embeddings) == 0:
        print("❌ No valid embeddings available for diarization!")
        return []
    
    if len(valid_embeddings) < 2:
        print("⚠ Only one valid segment, assigning to single speaker")
        return [(0, valid_timestamps[0][0], valid_timestamps[0][1])]
    
    num_speakers = estimate_speakers(valid_embeddings)
    clustering = AgglomerativeClustering(n_clusters=num_speakers)
    labels = clustering.fit_predict(valid_embeddings)

    diarization = []
    for i, (start, end) in enumerate(valid_timestamps):
        diarization.append((labels[i], start, end))

    return diarization

# ---------------------------------------------------
# ⿧ Merge consecutive segments of same speaker
# ---------------------------------------------------
def merge_segments(diarization, tolerance=1.0):
    if not diarization:
        return []
    
    diarization.sort(key=lambda x: x[1])
    merged = []
    current_spk, start, end = diarization[0]

    for spk, s, e in diarization[1:]:
        if spk == current_spk and s - end <= tolerance:
            end = e
        else:
            merged.append((current_spk, start, end))
            current_spk, start, end = spk, s, e

    merged.append((current_spk, start, end))
    return merged

# ---------------------------------------------------
# ⿨ Whisper Transcription (USING UNPROCESSED AUDIO)
# ---------------------------------------------------
def load_whisper_model():
    print("Loading Whisper model…")
    try:
        return pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3",
            device=0 if torch.cuda.is_available() else "cpu"
        )
    except:
        print("⚠ Falling back to whisper-medium...")
        return pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-medium",
            device=0 if torch.cuda.is_available() else "cpu"
        )

def transcribe_segment(model, audio_path, start, end, sr):
    """
    Transcribe using ORIGINAL UNPROCESSED audio for best quality
    """
    waveform, original_sr = torchaudio.load(audio_path)
    
    if original_sr != sr:
        start_original = int(start * original_sr)
        end_original = int(end * original_sr)
    else:
        start_original = int(start * sr)
        end_original = int(end * sr)
    
    end_original = min(end_original, waveform.shape[1])
    
    segment = waveform[:, start_original:end_original]
    
    if segment.shape[1] == 0:
        return "[No audio in this segment]"
    
    audio_np = segment.squeeze().numpy()
    
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=0)
    
    temp_dir = Config.get_temp_dir()
    temp_file = os.path.join(temp_dir, "temp_segment.wav")
    
    sf.write(temp_file, audio_np, original_sr, subtype='PCM_16')
    
    try:
        result = model(temp_file, generate_kwargs={"task": "translate"})
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return result["text"]
    except Exception as e:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        print(f"Transcription error: {e}")
        return "[Transcription failed]"

# ---------------------------------------------------
# 🆕 Enhanced cluster embedding with normalization
# ---------------------------------------------------
def get_cluster_embedding(embeddings, diarization, speaker_id):
    """
    Get normalized average embedding for all segments of a speaker.
    """
    speaker_embeddings = []
    for i, (spk, start, end) in enumerate(diarization):
        if spk == speaker_id:
            speaker_embeddings.append(embeddings[i])
    
    if not speaker_embeddings:
        return None
    
    mean_emb = np.mean(speaker_embeddings, axis=0)
    normalized_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
    
    return normalized_emb

# ---------------------------------------------------
# ⿩ Main diarization + transcription WITH AUTO IDENTIFICATION AND PDF
# ---------------------------------------------------
def main(audio_path, samples_dir=None):
    if samples_dir is None:
        samples_dir = get_samples_directory()
        
    wav_path = convert_to_wav(audio_path)
    
    try:
        model_spk = load_embedding_model()
    except Exception as e:
        print(f"❌ Failed to load speaker recognition model: {e}")
        print("Please check your internet connection or install required packages:")
        print("pip install --upgrade speechbrain huggingface_hub")
        return
    
    voice_embeddings = load_voice_samples(model_spk, samples_dir)
    
    try:
        whisper_model = load_whisper_model()
    except Exception as e:
        print(f"❌ Failed to load Whisper model: {e}")
        print("Please check your internet connection or try a smaller model")
        return

    segments, timestamps, processed_sr = segment_audio_for_speaker_id(wav_path)
    
    if len(segments) == 0:
        print("❌ No audio segments could be created. Please check your audio file.")
        return
        
    embeddings = extract_embeddings(model_spk, segments)
    diarization = diarize(embeddings, timestamps)
    
    if len(diarization) == 0:
        print("❌ No diarization results. Processing failed.")
        return
        
    merged = merge_segments(diarization)

    print("\n🎧 Final diarization + transcription:\n")
    results = []

    speaker_map = {}
    
    # Filter out segments shorter than 2.5 seconds
    filtered_segments = []
    for spk, start, end in merged:
        duration = end - start
        if duration >= 2.5:
            filtered_segments.append((spk, start, end))
        else:
            print(f"⚠ Skipping short segment ({duration:.2f}s): Speaker {spk} from {start:.2f}s to {end:.2f}s")

    print(f"\n📊 After filtering: {len(filtered_segments)} segments (removed {len(merged) - len(filtered_segments)} short segments)\n")

    # Process each segment
    for spk, start, end in filtered_segments:
        if spk not in speaker_map:
            cluster_embedding = get_cluster_embedding(embeddings, diarization, spk)
            
            if cluster_embedding is not None and voice_embeddings:
                print(f"\n🔍 Identifying Speaker {spk} (first appears at {start:.2f}s):")
                identified_name, score = identify_speaker(cluster_embedding, voice_embeddings)
                
                # Apply threshold: if score >= SIMILARITY_THRESHOLD, assign known name; else "Client"
                if score >= SIMILARITY_THRESHOLD:
                    speaker_map[spk] = identified_name
                    print(f"✅ Assigned as: {identified_name} (score: {score:.4f})")
                else:
                    speaker_map[spk] = "Client"
                    print(f"🔄 Assigned as: Client (score {score:.4f} below threshold {SIMILARITY_THRESHOLD})")
            else:
                # No voice samples available – fallback to generic names
                speaker_map[spk] = f"Speaker_{spk+1}"
                print(f"🆕 New speaker detected: Speaker_{spk+1}")
        
        name = speaker_map[spk]
        
        text = transcribe_segment(whisper_model, wav_path, start, end, processed_sr)
        duration = round(end - start, 2)

        print(f"\n{name} ({start:.2f}s - {end:.2f}s | {duration}s): {text}")

        results.append({
            "speaker": name,
            "start": round(start, 2),
            "end": round(end, 2),
            "duration": duration,
            "transcript": text
        })

    # Save JSON transcript
    output_dir = Config.get_output_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"conversation_{timestamp}.json")
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n✅ JSON transcript saved to {json_path}")
    
    # Generate PDF
    if results:
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        meeting_title = f"Meeting: {base_name.replace('_', ' ').title()}"
        
        pdf_generator = MeetingTranscriptPDF(title=meeting_title)
        
        pdf_path = pdf_generator.create_pdf(
            transcript_data=results,
            output_path=None,
            meeting_title=meeting_title,
            meeting_date=datetime.now().strftime("%B %d, %Y at %I:%M %p"),
            agenda_items=None
        )
        
        if pdf_path:
            print(f"\n📊 Transcript Summary:")
            print(f"   • Total Speakers: {len(set(item['speaker'] for item in results))}")
            print(f"   • Total Segments: {len(results)}")
            total_duration = sum(item['duration'] for item in results)
            print(f"   • Total Duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
            print(f"   • Output Files:")
            print(f"     - {json_path}")
            print(f"     - {pdf_path}")
            
            print(f"\n📁 All output files are saved in: {output_dir}")

# ---------------------------------------------------
# 🔚 Program entry point (FILE or MIC mode)
# ---------------------------------------------------
if __name__ == "__main__":
    import sys

    print("\n" + "="*50)
    print("🎙️  LIVE MEETING DIARIZATION WITH SINGLE SPEAKER ID")
    print("="*50 + "\n")
    print(f"🔧 Similarity threshold: {SIMILARITY_THRESHOLD} (set via env SIMILARITY_THRESHOLD)")
    
    Config.get_output_dir()
    Config.get_temp_dir()
    Config.get_model_cache_dir()
    
    print("Choose input mode:")
    print("1. Process audio file")
    print("2. Live microphone streaming\n")

    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        audio_file = input("\n📁 Enter the path to your audio file: ").strip()
        
        if not os.path.exists(audio_file):
            print(f"❌ Error: File '{audio_file}' not found!")
            sys.exit(1)
            
        try:
            print("\n🔧 Settings: Auto-speaker identification (single sample) | Automatic PDF generation")
            print("📊 Output: 4-column table layout (Speaker, Start, End, Transcript)")
            print("-" * 50)
            main(audio_file)
        except KeyboardInterrupt:
            print("\n\n⚠ Process interrupted by user")
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            import traceback
            traceback.print_exc()
            print("\n💡 TROUBLESHOOTING TIPS:")
            print("1. Make sure all dependencies are installed:")
            print("   pip install -r requirements.txt")
            print("2. Check your internet connection for model downloads")
            print("3. Ensure your audio file is in a supported format (WAV, MP3, M4A, FLAC)")

    elif choice == "2":
        try:
            duration_input = input("Enter meeting duration in seconds (default 600): ").strip()
            try:
                duration_sec = int(duration_input) if duration_input else 600
            except ValueError:
                print("Invalid input, defaulting to 600 seconds (10 minutes).")
                duration_sec = 600

            live_audio = stream_audio(duration_sec=duration_sec)
            
            try:
                print("\n🔧 Settings: Auto-speaker identification (single sample) | Automatic PDF generation")
                print("📊 Output: 4-column table layout (Speaker, Start, End, Transcript)")
                print("-" * 50)
                main(live_audio)
            except KeyboardInterrupt:
                print("\n\n⚠ Process interrupted by user")
            except Exception as e:
                print(f"\n❌ An error occurred: {e}")
                import traceback
                traceback.print_exc()
        except KeyboardInterrupt:
            print("\n\n⚠ Recording interrupted by user")

    else:
        print("❌ Invalid choice. Please run again and select 1 or 2.")