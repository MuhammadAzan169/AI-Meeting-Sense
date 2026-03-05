# summary_service.py
"""
Meeting Summary Service - FastAPI Service
Generates professional PDF summaries for meetings using OpenRouter API
Retrieves meeting details and user API keys from Firestore
Runs on port 8004
"""

import json
import os
import re
from datetime import datetime
from typing import Optional, List, Dict, Any
import requests
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.pdfgen import canvas
import traceback
import firebase_admin
from firebase_admin import credentials, firestore
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import uvicorn

# ===================== PYDANTIC MODELS =====================

class LLMProvider(str, Enum):
    """Supported LLM providers (simplified for OpenRouter)"""
    OPENROUTER = "openrouter"

class LLMConfig(BaseModel):
    """LLM Configuration Model - optimized for OpenRouter"""
    api_key: str = Field(..., description="API key for OpenRouter")
    model: str = Field(default="openai/gpt-4o-mini", description="Model name to use (OpenRouter format)")
    provider: LLMProvider = Field(default=LLMProvider.OPENROUTER, description="LLM provider (always openrouter)")
    base_url: str = Field(default="https://openrouter.ai/api/v1", description="OpenRouter API endpoint")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="Temperature for generation")
    max_tokens: int = Field(default=4000, ge=1, le=32000, description="Maximum tokens to generate")
    timeout: Optional[int] = Field(default=None, ge=1, description="Timeout in seconds for API calls")
    
    @field_validator('provider', mode='before')
    @classmethod
    def force_openrouter(cls, v):
        """Force provider to be openrouter, ignoring any other value"""
        return LLMProvider.OPENROUTER

class TranscriptSegment(BaseModel):
    """Transcript segment model"""
    speaker: str
    transcript: str
    segment_id: Optional[int] = None
    start: Optional[float] = 0.0
    end: Optional[float] = 0.0
    duration: Optional[float] = 0.0

class TranscriptData(BaseModel):
    """Complete transcript data model"""
    transcripts: List[TranscriptSegment]
    agenda: str
    metadata: Dict[str, Any] = {}
    summary_info: Dict[str, Any]

class AgendaAnalysis(BaseModel):
    """Agenda analysis results"""
    agenda_words: List[str]
    relevant_words: List[str]
    relevance_percentage: float

class ActionItemStatus(str, Enum):
    """Action item status"""
    PENDING = "pending"
    COMPLETED = "completed"
    DELAYED = "delayed"
    UNABLE_TO_COMPLETE = "unable_to_complete"

class ActionItemPriority(str, Enum):
    """Action item priority"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ActionItem(BaseModel):
    """Action item model"""
    taskId: str
    task: str
    responsiblePerson: str
    responsibleEmail: Optional[str] = None
    deadline: str
    priority: ActionItemPriority = ActionItemPriority.MEDIUM
    status: ActionItemStatus = ActionItemStatus.PENDING
    remarks: Optional[str] = None
    completionDescription: Optional[str] = None
    unableToCompleteReason: Optional[str] = None
    createdAt: str = ""
    updatedAt: str = ""

class ActionItemsData(BaseModel):
    """Action items data for a meeting"""
    meetingId: str
    meetingTitle: str
    meetingDate: str
    userId: str
    actionItems: List[ActionItem]
    createdAt: str
    updatedAt: str
    totalItems: int = 0
    completedItems: int = 0
    pendingItems: int = 0
    delayedItems: int = 0
    unableToCompleteItems: int = 0

class MeetingAnalysis(BaseModel):
    """Meeting analysis results"""
    content: str
    meeting_id: str
    transcript_data: TranscriptData
    agenda_analysis: AgendaAnalysis

# ============================================================
# FastAPI Setup
# ============================================================

app = FastAPI(title="Summary Service (OpenRouter)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Pydantic Models for FastAPI
# ============================================================

class GenerateSummaryRequest(BaseModel):
    meetingFolderPath: str
    userId: str
    meetingId: str

class GetActionItemsRequest(BaseModel):
    meetingId: str
    userId: str

class UpdateActionItemRequest(BaseModel):
    meetingId: str
    userId: str
    taskId: str
    task: Optional[str] = None
    responsiblePerson: Optional[str] = None
    responsibleEmail: Optional[str] = None
    deadline: Optional[str] = None
    priority: Optional[str] = None
    status: Optional[str] = None
    remarks: Optional[str] = None
    completionDescription: Optional[str] = None
    unableToCompleteReason: Optional[str] = None

class CreateActionItemRequest(BaseModel):
    meetingId: str
    userId: str
    task: str
    responsiblePerson: str
    responsibleEmail: Optional[str] = None
    deadline: str
    priority: str
    status: Optional[str] = "pending"

class DeleteActionItemRequest(BaseModel):
    meetingId: str
    userId: str
    taskId: str

class SendActionItemsEmailRequest(BaseModel):
    meetingId: str
    userId: str

# ============================================================
# Firebase Initialization
# ============================================================

db = None
try:
    if not firebase_admin._apps:
        firebase_creds_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "config/firebase-service-account.json")
        if not os.path.exists(firebase_creds_path):
            firebase_creds_path = "backend/config/firebase-service-account.json"
        cred = credentials.Certificate(firebase_creds_path)
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("✅ Firebase initialized successfully")
except Exception as e:
    print(f"❌ Firebase initialization error: {e}")

# ============================================================
# Summary Service Class
# ============================================================

class MeetingSummaryService:
    def __init__(self, meeting_folder_path, user_id, meeting_id):
        """
        Initialize the summary service for a specific meeting
        
        Args:
            meeting_folder_path: Path to the folder containing conversation.json
            user_id: Firebase user ID to retrieve API key and settings
            meeting_id: Meeting ID to retrieve meeting details from Firestore
        """
        self.meeting_folder_path = Path(meeting_folder_path)
        self.user_id = user_id
        self.meeting_id = meeting_id
        self.transcript_file = self.meeting_folder_path / "conversation.json"
        self.company_name = "Meeting Sense"
        self.company_tagline = "Intelligent Meeting Documentation"
        
        # Use global db
        self.db = db
        
        # Load meeting details and LLM configuration from Firestore
        self._load_meeting_details()
        self._setup_llm_config()
        
        # Setup PDF styles
        self._setup_balanced_styles()

    def _load_meeting_details(self):
        """Load meeting details from Firestore"""
        try:
            if not self.db:
                print("⚠️ Firebase not initialized, using defaults")
                self.meeting_title = "Meeting Discussion"
                self.meeting_agenda = "General Discussion"
                return
            
            # Retrieve meeting document from Firestore
            meeting_ref = self.db.collection('meetings').document(self.meeting_id)
            meeting_doc = meeting_ref.get()
            
            if meeting_doc.exists:
                meeting_data = meeting_doc.to_dict()
                self.meeting_title = meeting_data.get('title', 'Meeting Discussion')
                self.meeting_agenda = meeting_data.get('agenda', 'General Discussion')
                self.meeting_date = meeting_data.get('date', datetime.now().strftime('%Y-%m-%d'))
                self.meeting_time = meeting_data.get('time', datetime.now().strftime('%H:%M'))
                self.meeting_type = meeting_data.get('meetingType', 'General')
                self.meeting_mode = meeting_data.get('meetingMode', 'Online')
                print(f"✅ Meeting details loaded: {self.meeting_title}")
            else:
                print(f"⚠️ Meeting {self.meeting_id} not found in Firestore, using defaults")
                self.meeting_title = "Meeting Discussion"
                self.meeting_agenda = "General Discussion"
                self.meeting_date = datetime.now().strftime('%Y-%m-%d')
                self.meeting_time = datetime.now().strftime('%H:%M')
                self.meeting_type = "General"
                self.meeting_mode = "Online"
        except Exception as e:
            print(f"❌ Error loading meeting details: {e}")
            self.meeting_title = "Meeting Discussion"
            self.meeting_agenda = "General Discussion"
            self.meeting_date = datetime.now().strftime('%Y-%m-%d')
            self.meeting_time = datetime.now().strftime('%H:%M')

    def _setup_llm_config(self):
        """Setup LLM configuration using Pydantic model from user settings (OpenRouter focused)"""
        try:
            if not self.db:
                raise RuntimeError("❌ Firebase not initialized")
            
            # Retrieve user settings document from Firestore
            user_settings_ref = self.db.collection('userSettings').document(self.user_id)
            user_settings_doc = user_settings_ref.get()
            
            if not user_settings_doc.exists:
                raise RuntimeError(f"❌ User settings not found for user {self.user_id}")
            
            user_settings = user_settings_doc.to_dict()
            
            # Get API key (now used for OpenRouter)
            api_key = user_settings.get('openaiApiKey') or user_settings.get('llmApiKey')
            if not api_key:
                raise RuntimeError("❌ API Key not found in user settings. Please configure it in Settings > API Configuration.")
            
            # Get model (OpenRouter format, e.g., "openai/gpt-4o-mini")
            model = user_settings.get('llmModel', 'openai/gpt-4o-mini')
            
            # Force provider to OpenRouter
            provider = LLMProvider.OPENROUTER
            
            # Base URL is fixed for OpenRouter, but allow override if needed
            base_url = user_settings.get('llmBaseUrl', 'https://openrouter.ai/api/v1')
            
            # Get other settings with defaults
            temperature = float(user_settings.get('llmTemperature', '0.2'))
            max_tokens = int(user_settings.get('llmMaxTokens', '4000'))
            timeout_str = user_settings.get('llmTimeout', '')
            timeout = int(timeout_str) if timeout_str else None
            
            # Create LLMConfig instance
            self.llm_config = LLMConfig(
                api_key=api_key,
                model=model,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout
            )
            
            print(f"✅ LLM Configuration (OpenRouter):")
            print(f"   Model: {self.llm_config.model}")
            print(f"   Base URL: {self.llm_config.base_url}")
            print(f"   Temperature: {self.llm_config.temperature}")
            print(f"   Max Tokens: {self.llm_config.max_tokens}")
            
        except Exception as e:
            print(f"❌ Error setting up LLM configuration: {e}")
            traceback.print_exc()
            raise RuntimeError(f"Failed to setup LLM configuration: {str(e)}")

    def _get_api_headers(self) -> Dict[str, str]:
        """Get headers for OpenRouter API"""
        return {
            "Authorization": f"Bearer {self.llm_config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/meeting-summarizer",  # Replace with your site
            "X-Title": "Meeting Summarizer"
        }

    def _setup_balanced_styles(self):
        """Setup balanced professional styles with dark blue/light blue theme"""
        self.styles = getSampleStyleSheet()

        # Professional but clean styles
        self.pro_styles = {}

        # Main Title Style - Dark Blue
        self.pro_styles['MainTitle'] = ParagraphStyle(
            name='MainTitle',
            parent=self.styles['Title'],
            fontSize=24,
            textColor=colors.HexColor("#1E3A8A"),  # Dark Blue
            spaceAfter=6,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
            leading=28
        )

        # Meeting Title Style - Medium Blue
        self.pro_styles['MeetingTitle'] = ParagraphStyle(
            name='MeetingTitle',
            parent=self.styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor("#2563EB"),  # Blue-600
            spaceAfter=10,
            alignment=TA_CENTER,
            fontName='Helvetica',
            leading=18
        )

        # Section Header - Dark Blue
        self.pro_styles['SectionHeader'] = ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=14,
            textColor=colors.HexColor("#1E3A8A"),  # Dark Blue
            spaceAfter=6,
            spaceBefore=12,
            fontName='Helvetica-Bold',
            alignment=TA_LEFT
        )

        # Regular Text
        self.pro_styles['Regular'] = ParagraphStyle(
            name='Regular',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.black,
            spaceAfter=4,
            fontName='Helvetica',
            leading=12,
            wordWrap='LTR'
        )

        # Key Point Style
        self.pro_styles['KeyPoint'] = ParagraphStyle(
            name='KeyPoint',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor("#333333"),
            leftIndent=15,
            spaceAfter=3,
            fontName='Helvetica',
            leading=12,
            wordWrap='LTR'
        )

        # First Page Title Style
        self.pro_styles['FirstPageTitle'] = ParagraphStyle(
            name='FirstPageTitle',
            parent=self.styles['Title'],
            fontSize=28,
            textColor=colors.HexColor("#1E3A8A"),  # Dark Blue
            spaceAfter=15,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
            leading=32
        )

    def create_professional_header_footer(self, canvas, doc):
        """Create professional header/footer with dark blue theme"""
        canvas.saveState()

        # Header - Dark Blue
        header_height = 0.7 * inch
        canvas.setFillColor(colors.HexColor("#1E3A8A"))  # Dark Blue
        canvas.rect(0, doc.pagesize[1] - header_height, doc.pagesize[0], header_height, fill=1, stroke=0)

        # Company name
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 12)
        canvas.drawString(0.7 * inch, doc.pagesize[1] - 0.45 * inch, self.company_name)
        canvas.setFont("Helvetica", 8)
        canvas.drawString(0.7 * inch, doc.pagesize[1] - 0.65 * inch, self.company_tagline)

        # Document title
        canvas.setFont("Helvetica-Bold", 11)
        canvas.drawRightString(doc.pagesize[0] - 0.7 * inch, doc.pagesize[1] - 0.45 * inch, "MEETING MINUTES")
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(doc.pagesize[0] - 0.7 * inch, doc.pagesize[1] - 0.65 * inch, "Professional Summary")

        # Footer - Blue-600
        footer_height = 0.5 * inch
        canvas.setFillColor(colors.HexColor("#2563EB"))  # Blue-600
        canvas.rect(0, 0, doc.pagesize[0], footer_height, fill=1, stroke=0)

        # Page number
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica", 9)
        canvas.drawCentredString(doc.pagesize[0] / 2, 0.25 * inch, f"Page {doc.page}")

        canvas.setFont("Helvetica", 8)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        footer_left = f"© {datetime.now().year} {self.company_name} | Generated: {timestamp}"
        canvas.drawString(0.7 * inch, 0.15 * inch, footer_left)
        canvas.drawRightString(doc.pagesize[0] - 0.7 * inch, 0.15 * inch, "CONFIDENTIAL")

        canvas.restoreState()

    def create_first_page_header_footer(self, canvas, doc):
        """Create special header/footer for first page"""
        canvas.saveState()
        
        # Simple footer only - Blue-600
        canvas.setFillColor(colors.HexColor("#2563EB"))  # Blue-600
        canvas.rect(0, 0, doc.pagesize[0], 0.4 * inch, fill=1, stroke=0)
        
        # Page number
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica", 9)
        canvas.drawCentredString(doc.pagesize[0] / 2, 0.15 * inch, f"Page {doc.page}")
        
        canvas.setFont("Helvetica", 8)
        canvas.drawString(0.7 * inch, 0.1 * inch, f"© {datetime.now().year} {self.company_name}")
        canvas.drawRightString(doc.pagesize[0] - 0.7 * inch, 0.1 * inch, "CONFIDENTIAL")
        
        canvas.restoreState()

    def load_transcript_data(self) -> Optional[TranscriptData]:
        """Load transcript data from conversation.json"""
        try:
            if not self.transcript_file.exists():
                print(f"❌ Transcript file not found: {self.transcript_file}")
                return None

            with open(self.transcript_file, 'r', encoding='utf-8') as f:
                transcripts = json.load(f)

            print(f"📝 Found {len(transcripts)} transcript segments")
            print(f"📋 Meeting: {self.meeting_title}")
            print(f"📋 Agenda: {self.meeting_agenda}")

            # Process transcripts
            enhanced_transcripts = []
            total_words = 0
            speakers = set()

            for i, segment in enumerate(transcripts):
                try:
                    speaker = segment.get('speaker', f"Speaker_{i+1}")
                    transcript_text = segment.get('transcript', segment.get('text', ''))

                    if transcript_text and len(transcript_text.strip()) > 3:
                        enhanced_transcripts.append(TranscriptSegment(
                            speaker=speaker,
                            transcript=transcript_text.strip(),
                            segment_id=i + 1,
                            start=segment.get('start', 0),
                            end=segment.get('end', 0),
                            duration=segment.get('duration', 0)
                        ))
                        total_words += len(transcript_text.split())
                        speakers.add(speaker)

                except Exception as e:
                    print(f"⚠️ Error processing segment {i}: {e}")
                    continue

            print(f"✅ Processed {len(enhanced_transcripts)} valid segments")

            # Calculate total duration
            total_seconds = sum(t.duration for t in enhanced_transcripts)
            minutes = int(total_seconds // 60)
            seconds = int(total_seconds % 60)
            if minutes > 0:
                duration_formatted = f"{minutes} minutes {seconds} seconds"
            else:
                duration_formatted = f"{seconds} seconds"

            return TranscriptData(
                transcripts=enhanced_transcripts,
                agenda=self.meeting_agenda,
                metadata={
                    'date': self.meeting_date,
                    'time': self.meeting_time,
                    'type': self.meeting_type,
                    'mode': self.meeting_mode,
                    'title': self.meeting_title
                },
                summary_info={
                    'total_segments': len(enhanced_transcripts),
                    'total_words': total_words,
                    'total_duration': total_seconds,
                    'duration_formatted': duration_formatted,
                    'unique_speakers': len(speakers),
                    'speakers': list(speakers)
                }
            )

        except Exception as e:
            print(f"❌ Error loading transcript: {e}")
            traceback.print_exc()
            return None

    def _call_llm(self, prompt: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> Optional[str]:
        """Make API call to OpenRouter"""
        
        if not self.llm_config.api_key:
            print("❌ API key not configured. Please check your user settings")
            return None
        
        headers = self._get_api_headers()
        endpoint = f"{self.llm_config.base_url}/chat/completions"
        
        payload = {
            "model": self.llm_config.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens or self.llm_config.max_tokens,
            "temperature": temperature or self.llm_config.temperature
        }
        
        try:
            print(f"🔗 Calling OpenRouter API...")
            print(f"   Endpoint: {endpoint}")
            print(f"   Model: {self.llm_config.model}")
            
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=self.llm_config.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            # OpenRouter returns the same format as OpenAI
            return result['choices'][0]['message']['content']
                
        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"   Status Code: {e.response.status_code}")
                print(f"   Response: {e.response.text[:500]}")
            return None
        except KeyError as e:
            print(f"❌ Unexpected API response format: {e}")
            print(f"   Response: {result if 'result' in locals() else 'No response'}")
            return None
        except Exception as e:
            print(f"❌ Error calling OpenRouter: {e}")
            traceback.print_exc()
            return None

    def _extract_agenda_analysis(self, transcripts: List[TranscriptSegment], agenda: str) -> AgendaAnalysis:
        """Extract agenda analysis from transcript"""
        all_text = " ".join([t.transcript for t in transcripts]).lower()
        agenda_words = set(re.findall(r'\b\w+\b', agenda.lower()))

        # Calculate relevance
        relevant_words = [word for word in agenda_words if word in all_text and len(word) > 3]
        relevance_percentage = (len(relevant_words) / len(agenda_words) * 100) if agenda_words else 0

        return AgendaAnalysis(
            agenda_words=list(agenda_words),
            relevant_words=relevant_words,
            relevance_percentage=relevance_percentage
        )

    def _format_transcript_for_analysis(self, transcripts: List[TranscriptSegment]) -> str:
        """Format transcript for analysis"""
        formatted = ""
        for segment in transcripts:
            formatted += f"{segment.speaker}: {segment.transcript}\n\n"
        return formatted

    def _generate_balanced_llm_analysis(self, transcript_data: TranscriptData, meeting_id: str, 
                                      agenda_analysis: AgendaAnalysis) -> Optional[str]:
        """Generate balanced analysis with OpenRouter"""

        formatted_transcript = self._format_transcript_for_analysis(transcript_data.transcripts)

        prompt = f"""Create a PROFESSIONAL but CONCISE meeting summary with balanced detail.

MEETING INFORMATION:
- Title: {transcript_data.metadata.get('title', 'Meeting Discussion')}
- Agenda: {transcript_data.agenda}
- Participants: {', '.join(transcript_data.summary_info['speakers'])}
- Duration: {transcript_data.summary_info['duration_formatted']}
- Total Words: {transcript_data.summary_info['total_words']}
- Agenda Relevance: {agenda_analysis.relevance_percentage:.1f}%

YOUR TASK: Provide a comprehensive yet concise analysis with these EXACT sections:

## 1. EXECUTIVE SUMMARY
[3-4 sentences summarizing the entire meeting, objectives, and key outcomes]

## 2. KEY DECISIONS & OUTCOMES
[List 3-5 most important decisions made, format as bullet points]

## 3. PARTICIPANT KEY POINTS
[For each participant, list their 2-3 key contributions]
### [Participant Name]
• [Key contribution 1]
• [Key contribution 2]
[Optional: • Key contribution 3]

## 4. ACTION ITEMS
[Create a table with these columns]
| ID | Task Description | Responsible Person | Deadline | Priority |
|---|---|---|---|---|
| A1 | [Clear task description] | [Name] | [Date] | [High/Medium/Low] |

## 5. AGENDA ANALYSIS
### Topics Covered
[List bullet points of agenda items that were discussed]

### Topics Not Covered  
[List bullet points of agenda items that were NOT discussed]

### Agenda Adherence Assessment
[Brief assessment of how well the meeting stuck to the agenda]

## 6. KEY METRICS & DATA POINTS
[List all important numbers, percentages, dates, and metrics mentioned]

## 7. NEXT STEPS & RECOMMENDATIONS
[List 3-5 actionable next steps]

IMPORTANT INSTRUCTIONS:
1. Be professional but concise
2. Use bullet points, not long paragraphs
3. Include specific names and details from the transcript
4. Format tables clearly
5. Keep each section focused and to the point
6. Maximum 1 page of content (excluding title)

MEETING TRANSCRIPT:
{formatted_transcript}

Now provide the balanced professional summary in the exact structure above."""

        print("🧠 Analyzing meeting with balanced prompt (OpenRouter)...")
        response = self._call_llm(prompt, max_tokens=4500)

        if not response:
            print("❌ AI analysis failed")
            return self._generate_fallback_analysis(transcript_data, meeting_id, agenda_analysis)

        return response

    def _generate_fallback_analysis(self, transcript_data: TranscriptData, meeting_id: str, 
                                  agenda_analysis: AgendaAnalysis) -> str:
        """Generate fallback analysis if LLM fails"""
        speakers = transcript_data.summary_info['speakers']

        analysis = f"""## 1. EXECUTIVE SUMMARY
The meeting titled "{transcript_data.metadata.get('title', 'Meeting Discussion')}" focused on {transcript_data.agenda}. 
The discussion included updates and planning. Key outcomes were noted during the session.

## 2. KEY DECISIONS & OUTCOMES
• Decisions were discussed and documented during the meeting.

## 3. PARTICIPANT KEY POINTS
"""

        for speaker in speakers[:5]:  # Limit to 5 speakers
            analysis += f"\n### {speaker}\n"
            analysis += f"• Contributed to key discussions\n"

        analysis += f"""
## 4. ACTION ITEMS
| ID | Task Description | Responsible Person | Deadline | Priority |
|---|---|---|---|---|
| A1 | Review meeting minutes | Team Lead | {datetime.now().strftime('%Y-%m-%d')} | High |

## 5. AGENDA ANALYSIS
### Topics Covered
• Discussion related to {transcript_data.agenda}

### Topics Not Covered  
• Follow-up items to be addressed separately

### Agenda Adherence Assessment
The meeting focused on key agenda items with {agenda_analysis.relevance_percentage:.1f}% relevance.

## 6. KEY METRICS & DATA POINTS
• Meeting duration: {transcript_data.summary_info['duration_formatted']}
• Participants: {len(speakers)}
• Word count: {transcript_data.summary_info['total_words']}

## 7. NEXT STEPS & RECOMMENDATIONS
• Review meeting transcript
• Follow up on discussed items
• Schedule next meeting if needed"""

        return analysis

    def _wrap_table_text(self, text: str, max_length: int = 80) -> str:
        """Wrap text for table cells to prevent overflow"""
        if len(text) <= max_length:
            return text
        
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            if len(' '.join(current_line + [word])) <= max_length:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines)

    def _create_first_page(self, story, transcript_data: TranscriptData, agenda_analysis: AgendaAnalysis):
        """Create clean, professional first page with dark blue theme"""
        
        # Add top margin
        story.append(Spacer(1, 0.8 * inch))
        
        # Main document title - Dark Blue
        story.append(Paragraph("MEETING MINUTES", self.pro_styles['FirstPageTitle']))
        story.append(Spacer(1, 0.2 * inch))
        
        # Company name and tagline
        story.append(Paragraph(self.company_name, ParagraphStyle(
            name='CompanyFirst',
            fontName='Helvetica-Bold',
            fontSize=14,
            textColor=colors.HexColor("#2563EB"),  # Blue-600
            alignment=TA_CENTER,
            spaceAfter=4
        )))
        story.append(Paragraph(self.company_tagline, ParagraphStyle(
            name='TaglineFirst',
            fontName='Helvetica',
            fontSize=10,
            textColor=colors.HexColor("#4B5563"),  # Gray-600
            alignment=TA_CENTER,
            spaceAfter=0.4 * inch
        )))
        
        # Meeting title in a dark blue box
        story.append(Table(
            [[Paragraph(transcript_data.metadata.get('title', 'Meeting Discussion'), ParagraphStyle(
                name='TitleFirst',
                fontName='Helvetica-Bold',
                fontSize=16,
                textColor=colors.white,
                alignment=TA_CENTER,
                leading=18
            ))]],
            colWidths=[6.5 * inch],
            style=TableStyle([
                ('BACKGROUND', (0, 0), (0, 0), colors.HexColor("#1E3A8A")),  # Dark Blue
                ('BOX', (0, 0), (0, 0), 1, colors.HexColor("#172554")),  # Darker Blue
                ('PADDING', (0, 0), (0, 0), 12),
                ('VALIGN', (0, 0), (0, 0), 'MIDDLE'),
            ])
        ))
        
        story.append(Spacer(1, 0.4 * inch))
        
        # Meeting details
        duration = transcript_data.summary_info['duration_formatted']
        participants = len(transcript_data.summary_info['speakers'])
        relevance = agenda_analysis.relevance_percentage
        
        # Determine relevance color
        if relevance > 70:
            relevance_color = colors.HexColor("#16A34A")  # Green-600
            relevance_status = "Excellent"
        elif relevance > 40:
            relevance_color = colors.HexColor("#F59E0B")  # Amber-500
            relevance_status = "Good"
        else:
            relevance_color = colors.HexColor("#DC2626")  # Red-600
            relevance_status = "Low"
        
        details_data = [
            ['Meeting Details', ''],
            ['Meeting Title:', transcript_data.metadata.get('title', 'Meeting Discussion')],
            ['Date:', transcript_data.metadata.get('date', datetime.now().strftime('%Y-%m-%d'))],
            ['Time:', transcript_data.metadata.get('time', datetime.now().strftime('%H:%M'))],
            ['Duration:', duration],
            ['Participants:', str(participants)],
            ['Agenda Relevance:', f"{relevance:.1f}% ({relevance_status})"],
        ]
        
        details_table = Table(details_data, colWidths=[2.0 * inch, 4.5 * inch])
        details_table.setStyle(TableStyle([
            # Header row
            ('BACKGROUND', (0, 0), (1, 0), colors.HexColor("#2563EB")),  # Blue-600
            ('TEXTCOLOR', (0, 0), (1, 0), colors.white),
            ('FONTNAME', (0, 0), (1, 0), "Helvetica-Bold"),
            ('FONTSIZE', (0, 0), (1, 0), 12),
            ('ALIGN', (0, 0), (1, 0), 'CENTER'),
            ('PADDING', (0, 0), (1, 0), 10),
            
            # Label cells
            ('BACKGROUND', (0, 1), (0, -1), colors.HexColor("#E0F2FE")),  # Light Blue
            ('FONTNAME', (0, 1), (0, -1), "Helvetica-Bold"),
            ('FONTSIZE', (0, 1), (0, -1), 10),
            ('ALIGN', (0, 1), (0, -1), 'RIGHT'),
            ('PADDING', (0, 1), (0, -1), 8),
            
            # Value cells
            ('FONTNAME', (1, 1), (1, -1), "Helvetica"),
            ('FONTSIZE', (1, 1), (1, -1), 10),
            ('ALIGN', (1, 1), (1, -1), 'LEFT'),
            ('PADDING', (1, 1), (1, -1), 8),
            ('LEFTPADDING', (1, 1), (1, -1), 12),
            
            # Grid
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#BAE6FD")),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            
            # Relevance row special styling
            ('TEXTCOLOR', (1, 6), (1, 6), relevance_color),
            ('FONTNAME', (1, 6), (1, 6), "Helvetica-Bold"),
        ]))
        
        story.append(details_table)
        story.append(Spacer(1, 0.4 * inch))
        
        # Participants list
        speakers = transcript_data.summary_info['speakers']
        speakers_text = ", ".join(speakers)
        story.append(Paragraph("Participants", ParagraphStyle(
            name='ParticipantsTitle',
            fontName='Helvetica-Bold',
            fontSize=11,
            textColor=colors.HexColor("#1E3A8A"),  # Dark Blue
            alignment=TA_LEFT,
            spaceAfter=6
        )))
        story.append(Paragraph(speakers_text, ParagraphStyle(
            name='ParticipantsList',
            fontName='Helvetica',
            fontSize=10,
            textColor=colors.HexColor("#333333"),
            alignment=TA_LEFT,
            spaceAfter=0.3 * inch
        )))
        
        # Confidential notice
        story.append(Table(
            [[Paragraph("CONFIDENTIAL MEETING SUMMARY", ParagraphStyle(
                name='ConfidentialFirst',
                fontName='Helvetica-Bold',
                fontSize=11,
                textColor=colors.HexColor("#DC2626"),  # Red-600
                alignment=TA_CENTER
            ))]],
            colWidths=[6.5 * inch],
            style=TableStyle([
                ('BACKGROUND', (0, 0), (0, 0), colors.HexColor("#FEE2E2")),  # Red-50
                ('BOX', (0, 0), (0, 0), 1, colors.HexColor("#DC2626")),  # Red-600
                ('PADDING', (0, 0), (0, 0), 10),
                ('VALIGN', (0, 0), (0, 0), 'MIDDLE'),
            ])
        ))
        
        story.append(Spacer(1, 0.1 * inch))
        story.append(Paragraph("For internal distribution only • Do not share without authorization", 
                             ParagraphStyle(
                                 name='NoticeFirst',
                                 fontName='Helvetica',
                                 fontSize=8,
                                 textColor=colors.HexColor("#666666"),
                                 alignment=TA_CENTER
                             )))
        
        story.append(PageBreak())

    def _parse_analysis_sections(self, analysis: str) -> Dict[str, str]:
        """Parse analysis into sections"""
        sections = {}
        current_section = None
        current_content = []

        lines = analysis.split('\n')

        for line in lines:
            line = line.strip()

            if line.startswith('## '):
                if current_section is not None:
                    sections[current_section] = '\n'.join(current_content).strip()
                    current_content = []
                current_section = line[3:].strip()
            elif current_section is not None:
                if line or current_content:
                    current_content.append(line)

        if current_section is not None and current_content:
            sections[current_section] = '\n'.join(current_content).strip()

        return sections

    def _create_balanced_action_items(self, story, content: str):
        """Create balanced action items table with proper text wrapping"""
        lines = content.split('\n')
        table_data = [['ID', 'Task Description', 'Responsible', 'Deadline', 'Priority']]

        in_table = False
        for line in lines:
            line = line.strip()
            if '|' in line and '---' in line:
                in_table = True
                continue
            elif in_table and '|' in line:
                cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                if len(cells) >= 5:
                    # Wrap text for description column
                    task_desc = self._wrap_table_text(cells[1], max_length=60)
                    responsible = self._wrap_table_text(cells[2], max_length=20)
                    
                    # Apply priority styling
                    priority = cells[4].lower()
                    if 'high' in priority:
                        priority_color = colors.HexColor("#D32F2F")
                        priority_text = "High"
                    elif 'medium' in priority or 'med' in priority:
                        priority_color = colors.HexColor("#FF9800")
                        priority_text = "Medium"
                    else:
                        priority_color = colors.HexColor("#388E3C")
                        priority_text = "Low"
                    
                    priority_cell = Paragraph(priority_text,
                        ParagraphStyle(
                            name='PriorityStyle',
                            fontName='Helvetica-Bold',
                            fontSize=9,
                            textColor=priority_color,
                            alignment=TA_CENTER
                        ))

                    table_data.append([
                        Paragraph(cells[0], ParagraphStyle(
                            name='TableID',
                            fontName='Helvetica-Bold',
                            fontSize=10,
                            alignment=TA_CENTER
                        )),
                        Paragraph(task_desc, ParagraphStyle(
                            name='TableTask',
                            fontName='Helvetica',
                            fontSize=10,
                            textColor=colors.black,
                            wordWrap='LTR',
                            leading=12
                        )),
                        Paragraph(responsible, ParagraphStyle(
                            name='TableResponsible',
                            fontName='Helvetica',
                            fontSize=10,
                            alignment=TA_CENTER,
                            wordWrap='LTR'
                        )),
                        Paragraph(cells[3], ParagraphStyle(
                            name='TableDeadline',
                            fontName='Helvetica',
                            fontSize=10,
                            alignment=TA_CENTER
                        )),
                        priority_cell
                    ])
            elif in_table and not line:
                break

        if len(table_data) > 1:
            # Calculate column widths
            col_widths = [0.5 * inch, 3.5 * inch, 1.2 * inch, 0.8 * inch, 0.7 * inch]
            
            # Create table
            table = Table(table_data, colWidths=col_widths, repeatRows=1)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), "Helvetica-Bold"),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#BAE6FD")),
                ('PADDING', (0, 0), (-1, -1), 6),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5FBFF")]),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('ALIGN', (0, 0), (0, -1), 'CENTER'),
                ('ALIGN', (2, 0), (2, -1), 'CENTER'),
                ('ALIGN', (3, 0), (3, -1), 'CENTER'),
                ('ALIGN', (4, 0), (4, -1), 'CENTER'),
                ('LEADING', (0, 0), (-1, -1), 12),
            ]))
            
            story.append(KeepTogether(table))
        else:
            # Fallback if no table found
            story.append(Paragraph(content, self.pro_styles['Regular']))

    def _add_balanced_text_content(self, story, content: str):
        """Add balanced text content to PDF with proper wrapping"""
        lines = content.split('\n')

        for line in lines:
            line = line.strip()
            if line:
                if line.startswith('• '):
                    # Wrap bullet points
                    bullet_text = line[2:]
                    if len(bullet_text) > 100:
                        bullet_text = self._wrap_table_text(bullet_text, max_length=100)
                    story.append(Paragraph(f"• {bullet_text}", self.pro_styles['KeyPoint']))
                elif line.startswith('### '):
                    story.append(Paragraph(line[4:], ParagraphStyle(
                        name='Subsection',
                        fontName='Helvetica-Bold',
                        fontSize=12,
                        textColor=colors.HexColor("#2563EB"),
                        spaceAfter=4
                    )))
                    story.append(Spacer(1, 5))
                elif ':' in line and len(line) < 120:
                    # Key-value pairs
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        story.append(Paragraph(f"<b>{parts[0]}:</b> {parts[1]}", self.pro_styles['Regular']))
                else:
                    # Wrap regular text
                    if len(line) > 120:
                        line = self._wrap_table_text(line, max_length=120)
                    story.append(Paragraph(line, self.pro_styles['Regular']))
                story.append(Spacer(1, 3))

    def _add_balanced_sections(self, story, sections: Dict[str, str], transcript_data: TranscriptData):
        """Add balanced sections to PDF"""

        # Define section order
        section_order = [
            "1. EXECUTIVE SUMMARY",
            "2. KEY DECISIONS & OUTCOMES",
            "3. PARTICIPANT KEY POINTS",
            "4. ACTION ITEMS",
            "5. AGENDA ANALYSIS",
            "6. KEY METRICS & DATA POINTS",
            "7. NEXT STEPS & RECOMMENDATIONS"
        ]

        for section_title in section_order:
            if section_title in sections:
                # Add section header
                story.append(Paragraph(section_title, self.pro_styles['SectionHeader']))
                story.append(Spacer(1, 8))

                # Process section content
                content = sections[section_title]

                # Special handling for different sections
                if section_title == "4. ACTION ITEMS":
                    self._create_balanced_action_items(story, content)
                else:
                    self._add_balanced_text_content(story, content)

                story.append(Spacer(1, 15))

    def create_pdf(self, analysis: str, transcript_data: TranscriptData, agenda_analysis: AgendaAnalysis) -> Optional[str]:
        """Create professional PDF"""
        output_filename = f"{transcript_data.metadata.get('title', 'Meeting').replace(' ', '_')}_Summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        output_path = self.meeting_folder_path / output_filename

        print(f"📊 Creating PDF: {output_filename}...")

        try:
            doc = SimpleDocTemplate(
                str(output_path),
                pagesize=A4,
                rightMargin=0.7 * inch,
                leftMargin=0.7 * inch,
                topMargin=1.0 * inch,
                bottomMargin=0.8 * inch
            )

            story = []

            # Add first page
            self._create_first_page(story, transcript_data, agenda_analysis)

            # Parse and add analysis sections
            sections = self._parse_analysis_sections(analysis)
            self._add_balanced_sections(story, sections, transcript_data)

            # Build PDF
            def on_first_page(canvas, doc):
                self.create_first_page_header_footer(canvas, doc)
            
            def on_later_pages(canvas, doc):
                self.create_professional_header_footer(canvas, doc)
            
            doc.build(story, onFirstPage=on_first_page, onLaterPages=on_later_pages)

            print(f"✅ PDF created: {output_path}")
            return str(output_path)

        except Exception as e:
            print(f"❌ Error creating PDF: {e}")
            traceback.print_exc()
            return None

    def _save_analysis_json(self, analysis: str, transcript_data: TranscriptData, agenda_analysis: AgendaAnalysis):
        """Save analysis text as JSON for knowledge base indexing"""
        try:
            summary_data = {
                "analysis": analysis,
                "metadata": {
                    "meeting_title": transcript_data.metadata.get('title', 'Meeting Discussion'),
                    "meeting_agenda": transcript_data.agenda,
                    "meeting_date": getattr(self, 'meeting_date', datetime.now().strftime('%Y-%m-%d')),
                    "meeting_time": getattr(self, 'meeting_time', datetime.now().strftime('%H:%M')),
                    "meeting_type": getattr(self, 'meeting_type', 'General'),
                    "meeting_mode": getattr(self, 'meeting_mode', 'Online'),
                    "participants": transcript_data.summary_info['speakers'],
                    "unique_speakers": transcript_data.summary_info['unique_speakers'],
                    "duration_formatted": transcript_data.summary_info['duration_formatted'],
                    "total_words": transcript_data.summary_info['total_words'],
                    "agenda_relevance_percentage": agenda_analysis.relevance_percentage,
                    "generated_at": datetime.now().isoformat(),
                    "meeting_id": self.meeting_id,
                    "user_id": self.user_id
                }
            }

            summary_json_path = self.meeting_folder_path / "summary.json"
            with open(summary_json_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)

            print(f"✅ Analysis saved as JSON: {summary_json_path}")

        except Exception as e:
            print(f"❌ Error saving analysis JSON: {e}")
            traceback.print_exc()

    def _extract_action_items_from_analysis(self, analysis: str, transcript_data: TranscriptData) -> List[ActionItem]:
        """Extract action items from the LLM analysis text"""
        action_items = []
        
        try:
            # Find the ACTION ITEMS section
            action_items_section = re.search(r'## 4\.\s*ACTION ITEMS([\s\S]*?)(?=##|$)', analysis, re.IGNORECASE)
            
            if not action_items_section:
                print("⚠️ No ACTION ITEMS section found in analysis")
                return action_items
            
            section_content = action_items_section.group(1)
            lines = section_content.split('\n')
            
            in_table = False
            item_counter = 0
            speakers = transcript_data.summary_info.get('speakers', [])
            
            for line in lines:
                line = line.strip()
                
                # Detect table header separator
                if '|' in line and '---' in line:
                    in_table = True
                    continue
                
                # Parse table rows
                if in_table and '|' in line:
                    cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                    
                    # Skip header row
                    if len(cells) >= 4 and not cells[0].lower().startswith('id'):
                        item_counter += 1
                        task_id = cells[0] if not cells[0].lower().startswith('a') else cells[0]
                        if not task_id.startswith('A'):
                            task_id = f"A{item_counter}"
                        
                        task = cells[1] if len(cells) > 1 else ''
                        responsible = cells[2] if len(cells) > 2 else ''
                        deadline = cells[3] if len(cells) > 3 else 'Not specified'
                        priority_str = cells[4].lower() if len(cells) > 4 else 'medium'
                        
                        # Normalize priority
                        if 'high' in priority_str:
                            priority = ActionItemPriority.HIGH
                        elif 'low' in priority_str:
                            priority = ActionItemPriority.LOW
                        else:
                            priority = ActionItemPriority.MEDIUM
                        
                        # Try to find responsible person's email from participants
                        responsible_email = None
                        if responsible and self.db:
                            try:
                                # Search in members collection using filter keyword
                                from google.cloud.firestore_v1.base_query import FieldFilter
                                members_ref = self.db.collection('members')
                                members_query = members_ref.where(filter=FieldFilter('userId', '==', self.user_id)).stream()
                                
                                for member_doc in members_query:
                                    member_data = member_doc.to_dict()
                                    member_name = member_data.get('name', '').lower()
                                    if responsible.lower() in member_name or member_name in responsible.lower():
                                        responsible_email = member_data.get('email')
                                        break
                            except Exception as e:
                                print(f"⚠️ Could not look up email for {responsible}: {e}")
                        
                        now = datetime.now().isoformat()
                        action_item = ActionItem(
                            taskId=task_id,
                            task=task,
                            responsiblePerson=responsible,
                            responsibleEmail=responsible_email,
                            deadline=deadline,
                            priority=priority,
                            status=ActionItemStatus.PENDING,
                            remarks=None,
                            createdAt=now,
                            updatedAt=now
                        )
                        action_items.append(action_item)
                
                # End of table
                elif in_table and not line:
                    break
            
            print(f"✅ Extracted {len(action_items)} action items from analysis")
            
        except Exception as e:
            print(f"❌ Error extracting action items: {e}")
            traceback.print_exc()
        
        return action_items

    def _save_action_items_json(self, action_items: List[ActionItem], transcript_data: TranscriptData):
        """Save action items as separate JSON file"""
        try:
            now = datetime.now().isoformat()
            
            # Calculate statistics
            completed = sum(1 for item in action_items if item.status == ActionItemStatus.COMPLETED)
            pending = sum(1 for item in action_items if item.status == ActionItemStatus.PENDING)
            delayed = sum(1 for item in action_items if item.status == ActionItemStatus.DELAYED)
            
            action_items_data = ActionItemsData(
                meetingId=self.meeting_id,
                meetingTitle=transcript_data.metadata.get('title', 'Meeting Discussion'),
                meetingDate=getattr(self, 'meeting_date', datetime.now().strftime('%Y-%m-%d')),
                userId=self.user_id,
                actionItems=action_items,
                createdAt=now,
                updatedAt=now,
                totalItems=len(action_items),
                completedItems=completed,
                pendingItems=pending,
                delayedItems=delayed
            )
            
            # Save to action_items.json in meeting folder
            action_items_path = self.meeting_folder_path / "action_items.json"
            with open(action_items_path, 'w', encoding='utf-8') as f:
                json.dump(action_items_data.model_dump(), f, ensure_ascii=False, indent=2)
            
            print(f"✅ Action items saved: {action_items_path}")
            print(f"   Total: {len(action_items)}, Pending: {pending}, Completed: {completed}, Delayed: {delayed}")
            
            # Also save to Firestore for real-time sync
            self._save_action_items_to_firestore(action_items)
            
            return action_items_path
            
        except Exception as e:
            print(f"❌ Error saving action items JSON: {e}")
            traceback.print_exc()
            return None
    
    def _save_action_items_to_firestore(self, action_items: List[ActionItem]):
        """Save action items to Firestore subcollection for real-time sync"""
        if not db:
            print("⚠️ Firestore not initialized, skipping Firestore save")
            return
        
        try:
            action_items_ref = db.collection('meetings').document(self.meeting_id).collection('actionItems')
            
            # Clear existing items first (to handle re-generation)
            existing_docs = action_items_ref.stream()
            batch = db.batch()
            for doc in existing_docs:
                batch.delete(doc.reference)
            batch.commit()
            
            # Add new items
            batch = db.batch()
            for item in action_items:
                doc_ref = action_items_ref.document()
                
                doc_data = {
                    'id': doc_ref.id,
                    'taskId': doc_ref.id,
                    'task': item.task,
                    'responsiblePerson': item.responsiblePerson or item.responsible or '',
                    'responsibleEmail': (item.responsibleEmail or '').lower(),
                    'deadline': item.deadline or '',
                    'priority': (item.priority or 'medium').lower(),
                    'status': (item.status or 'pending').lower(),
                    'remarks': item.remarks or None,
                    'completionDescription': item.completionDescription or None,
                    'unableToCompleteReason': item.unableToCompleteReason or None,
                    'sourceSpeaker': getattr(item, 'source_speaker', None),
                    'sourceSegment': getattr(item, 'source_segment', None),
                    'createdAt': firestore.SERVER_TIMESTAMP,
                    'updatedAt': firestore.SERVER_TIMESTAMP,
                }
                
                # Remove None values
                doc_data = {k: v for k, v in doc_data.items() if v is not None}
                
                batch.set(doc_ref, doc_data)
            
            batch.commit()
            print(f"✅ Action items saved to Firestore: {len(action_items)} items")
            
        except Exception as e:
            print(f"❌ Error saving action items to Firestore: {e}")
            traceback.print_exc()

    def generate_summary(self) -> Optional[str]:
        """Generate complete meeting summary"""
        print("🚀 Starting meeting summary generation...")
        print("="*60)

        # Load transcript data
        transcript_data = self.load_transcript_data()
        if not transcript_data:
            print("❌ Failed to load transcript data")
            return None

        if not transcript_data.transcripts:
            print("⚠️ No valid transcript content available")
            return None

        # Extract agenda analysis
        agenda_analysis = self._extract_agenda_analysis(transcript_data.transcripts, transcript_data.agenda)

        meeting_id = f"MEET_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"📅 Meeting ID: {meeting_id}")

        # Generate analysis using LLM
        analysis = self._generate_balanced_llm_analysis(transcript_data, meeting_id, agenda_analysis)

        if not analysis:
            print("❌ Failed to generate analysis")
            return None

        # Save analysis text as JSON for knowledge base indexing
        self._save_analysis_json(analysis, transcript_data, agenda_analysis)

        # Extract and save action items as separate JSON
        action_items = self._extract_action_items_from_analysis(analysis, transcript_data)
        if action_items:
            self._save_action_items_json(action_items, transcript_data)

        # Generate PDF
        pdf_path = self.create_pdf(analysis, transcript_data, agenda_analysis)
        if pdf_path:
            print(f"\n🎉 MEETING SUMMARY COMPLETED!")
            print(f"   📄 PDF Report: {pdf_path}")
            print(f"   📋 Meeting: {transcript_data.metadata.get('title', 'Meeting Discussion')}")
            print(f"   👥 Participants: {transcript_data.summary_info['unique_speakers']}")
            print(f"   ⏱️  Duration: {transcript_data.summary_info['duration_formatted']}")
            print(f"   📊 Agenda Relevance: {agenda_analysis.relevance_percentage:.1f}%")
            print(f"   🤖 LLM Provider: OpenRouter")
            print(f"   ✅ Action Items: {len(action_items)}")
            return pdf_path

        print("❌ Failed to generate meeting summary")
        return None


# ============================================================
# Background Task
# ============================================================

def generate_summary_background(meeting_folder_path: str, user_id: str, meeting_id: str):
    """Background task to generate summary"""
    try:
        print(f"\n🚀 Starting summary generation for meeting {meeting_id}")
        service = MeetingSummaryService(meeting_folder_path, user_id, meeting_id)
        result = service.generate_summary()
        if result:
            print(f"✅ Summary generated successfully: {result}")

            # Trigger Knowledge Base summary auto-ingestion now that summary is ready
            try:
                print("\n📄 Triggering Knowledge Base summary auto-ingestion...", flush=True)

                # Extract meeting info from the service
                meeting_title = getattr(service, 'meeting_title', 'Meeting Discussion')
                meeting_date = getattr(service, 'meeting_date', datetime.now().strftime('%Y-%m-%d'))

                kb_payload = {
                    "meetingId": meeting_id,
                    "userId": user_id,
                    "meetingTitle": meeting_title,
                    "meetingDate": meeting_date
                }
                print(f"   Payload: {kb_payload}", flush=True)

                kb_response = requests.post(
                    "https://localhost:8017/auto-ingest-summary",
                    json=kb_payload,
                    verify=False,
                    timeout=30
                )
                print(f"   Response status: {kb_response.status_code}", flush=True)

                if kb_response.status_code == 200:
                    print("✅ Knowledge Base summary auto-ingestion triggered successfully", flush=True)
                else:
                    print(f"⚠️ Knowledge Base summary service responded with status {kb_response.status_code}", flush=True)
                    print(f"   Response: {kb_response.text}", flush=True)
            except requests.exceptions.ConnectionError as e:
                print(f"⚠️ Knowledge Base service not running on port 8007: {e}", flush=True)
            except Exception as e:
                print(f"⚠️ Could not trigger Knowledge Base summary ingestion: {e}", flush=True)
                import traceback
                traceback.print_exc()

            # Trigger Knowledge Base action items auto-ingestion
            try:
                print("\n📋 Triggering Knowledge Base action items auto-ingestion...", flush=True)

                meeting_title = getattr(service, 'meeting_title', 'Meeting Discussion')
                meeting_date = getattr(service, 'meeting_date', datetime.now().strftime('%Y-%m-%d'))

                kb_action_items_payload = {
                    "meetingId": meeting_id,
                    "userId": user_id,
                    "meetingTitle": meeting_title,
                    "meetingDate": meeting_date
                }
                print(f"   Payload: {kb_action_items_payload}", flush=True)

                kb_ai_response = requests.post(
                    "https://localhost:8017/auto-ingest-action-items",
                    json=kb_action_items_payload,
                    verify=False,
                    timeout=30
                )
                print(f"   Response status: {kb_ai_response.status_code}", flush=True)

                if kb_ai_response.status_code == 200:
                    print("✅ Knowledge Base action items auto-ingestion triggered successfully", flush=True)
                else:
                    print(f"⚠️ Knowledge Base action items service responded with status {kb_ai_response.status_code}", flush=True)
                    print(f"   Response: {kb_ai_response.text}", flush=True)
            except requests.exceptions.ConnectionError as e:
                print(f"⚠️ Knowledge Base service not running on port 8007: {e}", flush=True)
            except Exception as e:
                print(f"⚠️ Could not trigger Knowledge Base action items ingestion: {e}", flush=True)
                import traceback
                traceback.print_exc()
        else:
            print("❌ Summary generation failed")
    except Exception as e:
        print(f"❌ Error in summary generation: {e}")
        traceback.print_exc()


# ============================================================
# FastAPI Endpoints
# ============================================================

@app.get("/")
async def root():
    return {"message": "Summary Service (OpenRouter)", "version": "1.0.0", "status": "ready"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "summary",
        "firebase": "connected" if db else "not connected",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/generate-summary")
async def generate_summary(request: GenerateSummaryRequest, background_tasks: BackgroundTasks):
    """
    Generate meeting summary endpoint
    """
    print("\n" + "="*60)
    print("📄 SUMMARY GENERATION REQUEST RECEIVED")
    print("="*60)
    print(f"📁 Meeting Folder: {request.meetingFolderPath}")
    print(f"👤 User ID:        {request.userId}")
    print(f"📋 Meeting ID:     {request.meetingId}")
    print(f"⏰ Timestamp:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Verify conversation.json exists
    conversation_path = Path(request.meetingFolderPath) / "conversation.json"
    if not conversation_path.exists():
        raise HTTPException(status_code=404, detail="conversation.json not found in meeting folder")
    
    # Add background task
    background_tasks.add_task(
        generate_summary_background,
        request.meetingFolderPath,
        request.userId,
        request.meetingId
    )
    
    return {
        "success": True,
        "message": "Summary generation started in background",
        "meeting_id": request.meetingId,
        "status": "processing",
        "timestamp": datetime.now().isoformat()
    }


# ============================================================
# Helper Functions for Action Items
# ============================================================

def get_meeting_folder_from_firestore(meeting_id: str, user_id: str) -> Optional[Path]:
    """Get meeting folder path from Firestore meeting document"""
    if not db:
        return None
    
    try:
        meeting_ref = db.collection('meetings').document(meeting_id)
        meeting_doc = meeting_ref.get()
        
        if not meeting_doc.exists:
            return None
        
        meeting_data = meeting_doc.to_dict()
        meeting_title = meeting_data.get('title', 'Untitled')
        meeting_date = meeting_data.get('date', datetime.now().strftime('%Y-%m-%d'))
        
        # Sanitize and format
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', meeting_title)
        safe_name = re.sub(r'[\s_]+', '_', safe_name).strip('_ ')
        
        # Format date
        if meeting_date and 'T' in meeting_date:
            dt = datetime.fromisoformat(meeting_date.replace('Z', '+00:00'))
            formatted_date = dt.strftime('%Y-%m-%d')
        else:
            formatted_date = meeting_date
        
        folder_name = f"{safe_name}_{formatted_date}"
        recordings_dir = Path(__file__).parent / "recordings"
        folder_path = recordings_dir / user_id / folder_name
        
        return folder_path
    except Exception as e:
        print(f"❌ Error getting meeting folder: {e}")
        return None


# ============================================================
# Action Items Endpoints
# ============================================================

@app.post("/get-action-items")
async def get_action_items(request: GetActionItemsRequest):
    """
    Get action items for a specific meeting from Firestore
    """
    try:
        if not db:
            raise HTTPException(status_code=500, detail="Database not initialized")
        
        # Get action items from Firestore subcollection
        action_items_ref = db.collection('meetings').document(request.meetingId).collection('actionItems')
        docs = action_items_ref.stream()
        
        action_items = []
        for doc in docs:
            data = doc.to_dict()
            action_items.append(data)
        
        return {
            "success": True,
            "action_items": action_items,
            "total": len(action_items)
        }
    
    except Exception as e:
        print(f"❌ Error getting action items: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update-action-item")
async def update_action_item(request: UpdateActionItemRequest):
    """
    Update a specific action item in Firestore
    """
    try:
        if not db:
            raise HTTPException(status_code=500, detail="Database not initialized")
        
        # Find the action item in Firestore
        action_items_ref = db.collection('meetings').document(request.meetingId).collection('actionItems')
        
        # Try to find by taskId or id
        doc = None
        # First try by taskId
        query = action_items_ref.where('taskId', '==', request.taskId).limit(1)
        docs = query.stream()
        for d in docs:
            doc = d
            break
        
        # If not found, try by id field
        if not doc:
            query = action_items_ref.where('id', '==', request.taskId).limit(1)
            docs = query.stream()
            for d in docs:
                doc = d
                break
        
        # If still not found, try document ID
        if not doc:
            doc_ref = action_items_ref.document(request.taskId)
            doc_snapshot = doc_ref.get()
            if doc_snapshot.exists:
                doc = doc_snapshot
        
        if not doc:
            raise HTTPException(status_code=404, detail="Action item not found")
        
        # Build update data
        update_data = {
            'updatedAt': firestore.SERVER_TIMESTAMP
        }
        
        if request.task is not None:
            update_data['task'] = request.task
        if request.responsiblePerson is not None:
            update_data['responsiblePerson'] = request.responsiblePerson
        if request.responsibleEmail is not None:
            update_data['responsibleEmail'] = request.responsibleEmail.lower()
        if request.deadline is not None:
            update_data['deadline'] = request.deadline
        if request.priority is not None:
            update_data['priority'] = request.priority.lower()
        if request.status is not None:
            update_data['status'] = request.status.lower()
        if request.remarks is not None:
            update_data['remarks'] = request.remarks
        if request.completionDescription is not None:
            update_data['completionDescription'] = request.completionDescription
        if request.unableToCompleteReason is not None:
            update_data['unableToCompleteReason'] = request.unableToCompleteReason
        
        # Update in Firestore
        doc.reference.update(update_data)
        
        return {
            "success": True,
            "message": "Action item updated successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error updating action item: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/create-action-item")
async def create_action_item(request: CreateActionItemRequest):
    """
    Create a new action item in Firestore
    """
    try:
        if not db:
            raise HTTPException(status_code=500, detail="Database not initialized")
        
        # Verify meeting exists
        meeting_ref = db.collection('meetings').document(request.meetingId)
        meeting_doc = meeting_ref.get()
        
        if not meeting_doc.exists:
            raise HTTPException(status_code=404, detail="Meeting not found")
        
        # Get existing action items to determine next ID
        action_items_ref = meeting_ref.collection('actionItems')
        existing_docs = action_items_ref.stream()
        
        # Find the highest manual task ID (M1, M2, etc.)
        manual_ids = []
        for doc in existing_docs:
            data = doc.to_dict()
            task_id = data.get('taskId', '')
            if task_id.startswith('M') and task_id[1:].isdigit():
                manual_ids.append(int(task_id[1:]))
        
        next_id = max(manual_ids) + 1 if manual_ids else 1
        new_task_id = f"M{next_id}"
        
        # Create new action item document
        doc_ref = action_items_ref.document()
        
        new_item = {
            'id': doc_ref.id,
            'taskId': new_task_id,
            'task': request.task,
            'responsiblePerson': request.responsiblePerson,
            'responsibleEmail': (request.responsibleEmail or '').lower(),
            'deadline': request.deadline,
            'priority': (request.priority or 'medium').lower(),
            'status': (request.status or 'pending').lower(),
            'remarks': None,
            'completionDescription': None,
            'unableToCompleteReason': None,
            'createdAt': firestore.SERVER_TIMESTAMP,
            'updatedAt': firestore.SERVER_TIMESTAMP
        }
        
        doc_ref.set(new_item)
        
        return {
            "success": True,
            "message": "Action item created successfully",
            "taskId": new_task_id,
            "id": doc_ref.id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error creating action item: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/delete-action-item")
async def delete_action_item(request: DeleteActionItemRequest):
    """
    Delete a specific action item from Firestore
    """
    try:
        if not db:
            raise HTTPException(status_code=500, detail="Database not initialized")
        
        # Find the action item in Firestore
        action_items_ref = db.collection('meetings').document(request.meetingId).collection('actionItems')
        
        # Try to find by taskId or id
        doc = None
        # First try by taskId
        query = action_items_ref.where('taskId', '==', request.taskId).limit(1)
        docs = query.stream()
        for d in docs:
            doc = d
            break
        
        # If not found, try by id field
        if not doc:
            query = action_items_ref.where('id', '==', request.taskId).limit(1)
            docs = query.stream()
            for d in docs:
                doc = d
                break
        
        # If still not found, try document ID
        if not doc:
            doc_ref = action_items_ref.document(request.taskId)
            doc_snapshot = doc_ref.get()
            if doc_snapshot.exists:
                doc = doc_snapshot
        
        if not doc:
            raise HTTPException(status_code=404, detail="Action item not found")
        
        # Delete from Firestore
        doc.reference.delete()
        
        return {
            "success": True,
            "message": "Action item deleted successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error deleting action item: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/send-action-items-email")
async def send_action_items_email(request: SendActionItemsEmailRequest):
    """
    Send action items via email to each responsible person.
    Dynamically resolves emails by matching responsible person names
    to meeting participants in the Firestore members collection.
    Each person receives only their own action items.
    Reads action items from Firestore (not local file).
    """
    try:
        if not db:
            raise HTTPException(status_code=500, detail="Database not initialized")
        
        # Get action items from Firestore subcollection
        action_items_ref = db.collection('meetings').document(request.meetingId).collection('actionItems')
        docs = action_items_ref.stream()
        
        action_items = []
        for doc in docs:
            data = doc.to_dict()
            action_items.append(data)
        
        if not action_items:
            raise HTTPException(status_code=404, detail="No action items found for this meeting")
        
        # Get meeting details from Firestore
        meeting_ref = db.collection('meetings').document(request.meetingId)
        meeting_doc = meeting_ref.get()
        
        if not meeting_doc.exists:
            raise HTTPException(status_code=404, detail="Meeting not found")
        
        meeting_data = meeting_doc.to_dict()
        meeting_title = meeting_data.get('title', 'Meeting')
        meeting_date = meeting_data.get('date', '')
        meeting_participants = meeting_data.get('participants', [])
        
        # Build action_items_data structure for compatibility with rest of the code
        action_items_data = {
            'meetingId': request.meetingId,
            'meetingTitle': meeting_title,
            'meetingDate': meeting_date,
            'actionItems': action_items,
            'totalItems': len(action_items)
        }
        
        # Get user's email settings
        user_settings_ref = db.collection('userSettings').document(request.userId)
        user_settings_doc = user_settings_ref.get()
        
        if not user_settings_doc.exists:
            raise HTTPException(status_code=400, detail="User settings not found. Please configure email settings.")
        
        user_settings = user_settings_doc.to_dict()
        sender_email = user_settings.get('mailSenderEmail')
        google_app_password = user_settings.get('googleAppPassword')
        
        if not sender_email:
            raise HTTPException(status_code=400, detail="Mail Sender Email not configured. Please set it in Settings.")
        
        if not google_app_password:
            raise HTTPException(status_code=400, detail="Google App Password not configured. Please set it in Settings.")
        
        # Get display name from Firestore users collection
        user_doc = db.collection('users').document(request.userId).get()
        display_name = user_doc.to_dict().get('name', '') if user_doc.exists else ''
        
        # Get meeting participants from Firestore
        meeting_ref = db.collection('meetings').document(request.meetingId)
        meeting_doc = meeting_ref.get()
        meeting_participants = []
        if meeting_doc.exists:
            meeting_participants = meeting_doc.to_dict().get('participants', [])
        
        # Build a lookup of member name → email from the members collection
        # Only consider members that are participants of this meeting
        name_to_email = {}
        try:
            from google.cloud.firestore_v1.base_query import FieldFilter
            members_ref = db.collection('members')
            members_query = members_ref.where(filter=FieldFilter('userId', '==', request.userId)).stream()
            
            for member_doc_snap in members_query:
                member_data = member_doc_snap.to_dict()
                member_email = member_data.get('email', '')
                member_name = member_data.get('name', '')
                # Only include members who are participants of this meeting
                if member_email and member_name and (not meeting_participants or member_email in meeting_participants):
                    name_to_email[member_name.lower().strip()] = {
                        'email': member_email,
                        'name': member_name
                    }
        except Exception as e:
            print(f"⚠️ Error loading members for email lookup: {e}")
        
        # Group action items by responsible person, resolving emails dynamically
        items_by_email = {}
        unresolved_items = []
        
        for item in action_items_data.get('actionItems', []):
            responsible_name = item.get('responsiblePerson', '').strip()
            # First try the stored email
            resolved_email = item.get('responsibleEmail')
            
            # If no stored email, try to resolve from members lookup
            if not resolved_email and responsible_name:
                # Try exact match (case-insensitive)
                lookup = name_to_email.get(responsible_name.lower().strip())
                if lookup:
                    resolved_email = lookup['email']
                else:
                    # Try partial match
                    for name_key, info in name_to_email.items():
                        if responsible_name.lower() in name_key or name_key in responsible_name.lower():
                            resolved_email = info['email']
                            break
            
            if resolved_email:
                if resolved_email not in items_by_email:
                    items_by_email[resolved_email] = {
                        'name': responsible_name or 'Team Member',
                        'items': []
                    }
                items_by_email[resolved_email]['items'].append(item)
                
                # Also update the action item's email in the JSON for future use
                if not item.get('responsibleEmail'):
                    item['responsibleEmail'] = resolved_email
            else:
                unresolved_items.append(responsible_name or 'Unknown')
        
        if not items_by_email:
            detail_msg = "No action items could be matched to participant emails."
            if unresolved_items:
                detail_msg += f" Unresolved persons: {', '.join(set(unresolved_items))}. Please ensure responsible persons match meeting participant names."
            return {
                "success": False,
                "message": detail_msg,
                "successfulRecipients": [],
                "failedRecipients": []
            }
        
        # Update action items with resolved emails in Firestore
        for item in action_items_data.get('actionItems', []):
            if item.get('responsibleEmail'):
                # Find and update the item in Firestore
                item_ref = action_items_ref.document(item.get('id'))
                item_ref.update({
                    'responsibleEmail': item.get('responsibleEmail'),
                    'updatedAt': firestore.SERVER_TIMESTAMP
                })
        
        meeting_title = action_items_data.get('meetingTitle', 'Meeting')
        meeting_date = action_items_data.get('meetingDate', '')
        
        successful_recipients = []
        failed_recipients = []
        
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        
        for recipient_email, data in items_by_email.items():
            try:
                person_name = data['name']
                person_items = data['items']
                
                # Build email body
                email_subject = f"Action Items from Meeting: {meeting_title}"
                
                # Format items list
                items_list = ""
                for i, item in enumerate(person_items, 1):
                    priority = item.get('priority', 'medium').upper()
                    deadline = item.get('deadline', 'Not specified')
                    status = item.get('status', 'pending').upper()
                    remarks = item.get('remarks', '')
                    
                    items_list += f"""
{i}. {item.get('task', 'No description')}
   • Priority: {priority}
   • Deadline: {deadline}
   • Status: {status}"""
                    if remarks:
                        items_list += f"\n   • Remarks: {remarks}"
                    items_list += "\n"
                
                email_body = f"""
Dear {person_name},

You have been assigned the following action items from the meeting:

📋 Meeting: {meeting_title}
📅 Date: {meeting_date}

📌 Your Action Items:
{items_list}

Please review and complete these tasks by their respective deadlines.

If you have any questions, please contact the meeting organizer.

Best regards,
{display_name if display_name else sender_email}
Organizer

—
Powered by MeetingSense
"""
                
                # Create and send email
                msg = MIMEMultipart()
                msg['From'] = sender_email
                msg['To'] = recipient_email
                msg['Subject'] = email_subject
                msg.attach(MIMEText(email_body, 'plain'))
                
                server = smtplib.SMTP('smtp.gmail.com', 587)
                server.starttls()
                server.login(sender_email, google_app_password)
                server.sendmail(sender_email, recipient_email, msg.as_string())
                server.quit()
                
                successful_recipients.append(recipient_email)
                print(f"✅ Email sent to {recipient_email} ({person_name}) with {len(person_items)} action items")
                
            except Exception as e:
                print(f"❌ Failed to send email to {recipient_email}: {e}")
                failed_recipients.append(recipient_email)
        
        result_msg = f"Action items sent to {len(successful_recipients)} out of {len(items_by_email)} recipients"
        if unresolved_items:
            result_msg += f". Could not resolve emails for: {', '.join(set(unresolved_items))}"
        
        return {
            "success": True,
            "message": result_msg,
            "successfulRecipients": successful_recipients,
            "failedRecipients": failed_recipients
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error sending action items email: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/meeting-participants/{meeting_id}/{user_id}")
async def get_meeting_participants(meeting_id: str, user_id: str):
    """
    Get list of meeting participants with their names and emails
    for use in action item responsible person dropdown
    """
    try:
        if not db:
            raise HTTPException(status_code=500, detail="Database not initialized")
        
        # Get meeting document to get participant emails
        meeting_ref = db.collection('meetings').document(meeting_id)
        meeting_doc = meeting_ref.get()
        
        if not meeting_doc.exists:
            return {"success": False, "participants": []}
        
        meeting_data = meeting_doc.to_dict()
        participant_emails = meeting_data.get('participants', [])
        
        # Get member details from members collection
        from google.cloud.firestore_v1.base_query import FieldFilter
        members_ref = db.collection('members')
        members_query = members_ref.where(filter=FieldFilter('userId', '==', user_id)).stream()
        
        participants = []
        for member_doc_snap in members_query:
            member_data = member_doc_snap.to_dict()
            member_email = member_data.get('email', '')
            if member_email in participant_emails:
                participants.append({
                    "name": member_data.get('name', ''),
                    "email": member_email,
                    "designation": member_data.get('designation', '')
                })
        
        return {
            "success": True,
            "participants": participants
        }
    
    except Exception as e:
        print(f"❌ Error getting meeting participants: {e}")
        traceback.print_exc()
        return {"success": False, "participants": []}


@app.get("/action-items-analytics/{meeting_id}/{user_id}")
async def get_action_items_analytics(meeting_id: str, user_id: str):
    """
    Get action items analytics for meeting analytics visualization.
    Also automatically marks pending items as 'delayed' if past their deadline.
    Reads action items from Firestore (not local file).
    """
    try:
        if not db:
            raise HTTPException(status_code=500, detail="Database not initialized")
        
        # Get action items from Firestore subcollection
        action_items_ref = db.collection('meetings').document(meeting_id).collection('actionItems')
        docs = action_items_ref.stream()
        
        items = []
        for doc in docs:
            data = doc.to_dict()
            items.append(data)
        
        if not items:
            return {
                "success": False,
                "message": "No action items found for this meeting"
            }
        
        # Auto-delay check: mark pending items past deadline as 'delayed'
        today = datetime.now().date()
        auto_delayed = False
        for item in items:
            if item.get('status') == 'pending' and item.get('deadline'):
                try:
                    deadline_str = item['deadline']
                    # Try multiple date formats
                    deadline_date = None
                    for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%B %d, %Y', '%b %d, %Y', '%d %B %Y', '%d %b %Y']:
                        try:
                            deadline_date = datetime.strptime(deadline_str, fmt).date()
                            break
                        except ValueError:
                            continue
                    
                    if deadline_date and deadline_date < today:
                        # Update in Firestore
                        item_ref = action_items_ref.document(item.get('id'))
                        item_ref.update({
                            'status': 'delayed',
                            'remarks': item.get('remarks') or 'Automatically delayed - deadline passed',
                            'updatedAt': firestore.SERVER_TIMESTAMP
                        })
                        # Update local item for response
                        item['status'] = 'delayed'
                        item['remarks'] = item.get('remarks') or 'Automatically delayed - deadline passed'
                        auto_delayed = True
                        print(f"⏰ Auto-delayed: {item.get('taskId')} (deadline: {deadline_str})")
                except Exception as e:
                    pass  # Skip items with unparseable deadlines
        
        # Overall statistics
        overall_stats = {
            "totalItems": len(items),
            "completedItems": sum(1 for i in items if i.get('status') == 'completed'),
            "pendingItems": sum(1 for i in items if i.get('status') == 'pending'),
            "delayedItems": sum(1 for i in items if i.get('status') == 'delayed'),
            "unableToCompleteItems": sum(1 for i in items if i.get('status') == 'unable_to_complete'),
            "highPriority": sum(1 for i in items if i.get('priority') == 'high'),
            "mediumPriority": sum(1 for i in items if i.get('priority') == 'medium'),
            "lowPriority": sum(1 for i in items if i.get('priority') == 'low')
        }
        
        # Per-person breakdown
        person_stats = {}
        for item in items:
            person = item.get('responsiblePerson', 'Unassigned')
            if person not in person_stats:
                person_stats[person] = {
                    "name": person,
                    "email": item.get('responsibleEmail'),
                    "totalTasks": 0,
                    "completed": 0,
                    "pending": 0,
                    "delayed": 0,
                    "unableToComplete": 0,
                    "deadlines": []
                }
            
            person_stats[person]["totalTasks"] += 1
            status = item.get('status', 'pending')
            if status == 'completed':
                person_stats[person]["completed"] += 1
            elif status == 'delayed':
                person_stats[person]["delayed"] += 1
            elif status == 'unable_to_complete':
                person_stats[person]["unableToComplete"] += 1
            else:
                person_stats[person]["pending"] += 1
            
            if item.get('deadline'):
                person_stats[person]["deadlines"].append({
                    "task": item.get('task', ''),
                    "deadline": item.get('deadline'),
                    "priority": item.get('priority', 'medium')
                })
        
        return {
            "success": True,
            "overall": overall_stats,
            "byPerson": list(person_stats.values()),
            "items": items
        }
    
    except Exception as e:
        print(f"❌ Error getting action items analytics: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Main Entry Point
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("📄 SUMMARY SERVICE v1.0 (OpenRouter)")
    print("="*60)
    print(f"📌 Port: 8004")
    print(f"📌 Firebase: {'Connected' if db else 'Not connected'}")
    print("="*60 + "\n")
    
    import os
    port = int(os.getenv("SUMMARY_SERVICE_PORT", "8014"))

    # SSL Configuration
    ssl_keyfile = os.getenv("SSL_KEYFILE", "../59.103.127.217-key.pem")
    ssl_certfile = os.getenv("SSL_CERTFILE", "../59.103.127.217.pem")

    # Check if SSL files exist and use HTTPS
    if os.path.exists(ssl_keyfile) and os.path.exists(ssl_certfile):
        print(f"[Summary Service] Starting with HTTPS on port {port}")
        uvicorn.run(app, host="0.0.0.0", port=port, ssl_keyfile=ssl_keyfile, ssl_certfile=ssl_certfile)
    else:
        print(f"[Summary Service] SSL files not found, starting with HTTP on port {port}")
        uvicorn.run(app, host="0.0.0.0", port=port)