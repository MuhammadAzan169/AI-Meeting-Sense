# rizwan.py - RAG-Powered Critical Evaluation Engine for Real Estate Sales
# Combined Best-of-Both: Enterprise Visual Design + Robust Parsing
# Output: 3 PDFs — coaching_summary.pdf, agent_summary.pdf, visualizations.pdf
#         + visualizations_backup.json for frontend dashboard reconstruction
# Brutally honest, zero-repetition coaching evaluation
import json
import os
import re
import traceback
import pickle
import math
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from enum import Enum

import requests
import configparser
import numpy as np

# RAG and Embeddings
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("sentence-transformers not available. Install: pip install sentence-transformers")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS not available. Install: pip install faiss-cpu")

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    print("python-docx not available. Install: pip install python-docx")

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether, Flowable, HRFlowable, CondPageBreak
)
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import Drawing, Rect, String, Circle, Line, Polygon
from reportlab.graphics import renderPDF
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, ConfigDict


# ════════════════════════════════════════════════════════
# PYDANTIC V2 MODELS
# ════════════════════════════════════════════════════════

class LLMProvider(str, Enum):
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    CUSTOM = "custom"

class LLMConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    provider: str = Field(..., description="LLM provider name")
    api_key: str = Field(..., description="API key for the LLM provider")
    model: str = Field(..., description="Model name to use")
    base_url: Optional[str] = Field(..., description="Base URL for API calls")
    temperature: float = Field(..., ge=0.0, le=2.0)
    max_tokens: int = Field(..., ge=1, le=32000)
    timeout: int = Field(..., ge=1)

    @field_validator('base_url', mode='before')
    @classmethod
    def set_base_url_based_on_provider(cls, v, info):
        if v:
            return v
        provider = info.data.get('provider', 'openai').lower().strip()
        if provider == "openai":
            return "https://api.openai.com/v1"
        elif provider == "openrouter":
            return "https://openrouter.ai/api/v1"
        return v

class TranscriptSegment(BaseModel):
    speaker_id: Optional[str] = None
    speaker_name: Optional[str] = None
    transcript: str
    segment_id: Optional[int] = None
    start: Optional[float] = 0.0
    end: Optional[float] = 0.0
    duration: Optional[float] = 0.0

class TranscriptData(BaseModel):
    transcripts: List[TranscriptSegment]
    agenda: str
    metadata: Dict[str, Any] = {}
    summary_info: Dict[str, Any]

class AgendaAnalysis(BaseModel):
    agenda_words: List[str]
    relevant_words: List[str]
    relevance_percentage: float

class MeetingAnalysis(BaseModel):
    content: str
    meeting_id: str
    transcript_data: TranscriptData
    agenda_analysis: AgendaAnalysis
    enhanced_analysis: Dict[str, Any] = {}

class ActionItem(BaseModel):
    task: str
    responsible: str
    deadline: str = Field(default="No due date was mentioned")
    priority: str = Field(default="Medium")
    status: str = Field(default="Pending")

    @field_validator('deadline', mode='before')
    @classmethod
    def validate_deadline(cls, v):
        if not v or v.strip() == "":
            return "No due date was mentioned"
        return v

class KnowledgeChunk(BaseModel):
    chunk_id: int
    text: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = {}

class CoachingEvaluation(BaseModel):
    executive_evaluation: str
    critical_analysis: str
    strengths: List[str]
    improvement_areas: List[str]
    coaching_recommendations: List[str]
    knowledge_references: List[str]
    performance_scores: Dict[str, Any] = {}


# ════════════════════════════════════════════════════════
# VISUALIZATION LOGGER — JSON BACKUP FOR FRONTEND
# ════════════════════════════════════════════════════════

class VisualizationLogger:
    """Centralized visualization data storage system.
    
    Captures raw data from every chart/graph generated during evaluation
    and exports to a structured JSON file optimized for frontend libraries
    (Chart.js, D3.js, ECharts, Recharts).
    
    Usage:
        logger = VisualizationLogger(report_type="real_estate_sales_coaching")
        logger.log_chart(
            chart_id="confidence_score_chart",
            title="Agent Confidence Score",
            chart_type="bar",
            labels=["Opening", "Discovery"],
            datasets=[{"label": "Score", "values": [7, 8]}]
        )
        logger.save_to_json("visualizations_backup.json")
    """

    def __init__(self, report_type: str = "real_estate_sales_coaching"):
        self._visualizations: List[Dict[str, Any]] = []
        self._report_type = report_type
        self._created_at = datetime.now().isoformat()
        self._metadata: Dict[str, Any] = {}

    def set_metadata(self, **kwargs):
        """Set report-level metadata (agent name, call ID, duration, etc.)."""
        self._metadata.update(kwargs)

    def log_chart(
        self,
        chart_id: str,
        title: str,
        chart_type: str,
        labels: Optional[List[str]] = None,
        datasets: Optional[List[Dict[str, Any]]] = None,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        """Log a single visualization's raw data for frontend reconstruction.
        
        Args:
            chart_id: Unique identifier (e.g., 'talk_time_donut').
            title: Display title for the chart.
            chart_type: One of 'bar', 'horizontalBar', 'line', 'doughnut', 'radar',
                        'funnel', 'gauge', 'scatter', 'stackedBar', 'heatmap',
                        'quadrant', 'scoreCard', 'table'.
            labels: Category / axis labels.
            datasets: List of dataset dicts, each with 'label', 'values',
                      and optional 'colors', 'borderColors'.
            description: Human-readable description of what the chart shows.
            metadata: Chart-specific metadata (units, thresholds, etc.).
            options: Frontend rendering hints (y_range, stacked, etc.).
        """
        entry: Dict[str, Any] = {
            "id": chart_id,
            "title": title,
            "type": chart_type,
            "description": description,
            "labels": labels or [],
            "datasets": datasets or [],
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }
        if options:
            entry["options"] = options
        self._visualizations.append(entry)

    def get_all(self) -> List[Dict[str, Any]]:
        """Return all logged visualizations."""
        return list(self._visualizations)

    def to_dict(self) -> Dict[str, Any]:
        """Build the complete JSON-serializable output dict."""
        return {
            "generated_at": self._created_at,
            "report_type": self._report_type,
            "metadata": self._metadata,
            "total_visualizations": len(self._visualizations),
            "visualizations": self._visualizations,
        }

    def save_to_json(self, filepath: str):
        """Write the full visualization backup to a JSON file."""
        payload = self.to_dict()
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
            print(f"    Saved visualization JSON: {filepath}")
        except Exception as e:
            print(f"    Failed to save visualization JSON: {e}")


# ════════════════════════════════════════════════════════
# CUSTOM FLOWABLES (from Enterprise v2.0)
# ════════════════════════════════════════════════════════

class ScoreBar(Flowable):
    """Horizontal color-coded score bar: green >= 75%, yellow >= 50%, red < 50%."""
    def __init__(self, label, score, max_score=10, width=400, height=22):
        super().__init__()
        self.label = label
        self.score = score
        self.max_score = max_score
        self.width = width
        self.height = height

    def wrap(self, availWidth, availHeight):
        return self.width, self.height + 4

    def draw(self):
        pct = self.score / self.max_score if self.max_score else 0
        bar_w = self.width * 0.55
        bar_x = self.width * 0.40
        # Background
        self.canv.setFillColor(colors.HexColor("#ECEFF1"))
        self.canv.roundRect(bar_x, 4, bar_w, self.height - 4, 3, fill=1, stroke=0)
        # Filled portion
        fill_color = "#43A047" if pct >= 0.75 else "#FF8F00" if pct >= 0.50 else "#E53935"
        self.canv.setFillColor(colors.HexColor(fill_color))
        self.canv.roundRect(bar_x, 4, bar_w * pct, self.height - 4, 3, fill=1, stroke=0)
        # Label
        self.canv.setFont("Helvetica", 8)
        self.canv.setFillColor(colors.HexColor("#424242"))
        self.canv.drawString(4, 8, self.label)
        # Score text
        self.canv.setFont("Helvetica-Bold", 9)
        self.canv.setFillColor(colors.HexColor(fill_color))
        self.canv.drawString(bar_x + bar_w + 6, 8, f"{self.score}/{self.max_score}")


class SectionDivider(Flowable):
    """Thin horizontal rule divider."""
    def __init__(self, width=460, color="#1565C0", thickness=1.5):
        super().__init__()
        self.width = width
        self.color = color
        self.thickness = thickness

    def wrap(self, availWidth, availHeight):
        return self.width, self.thickness + 6

    def draw(self):
        self.canv.setStrokeColor(colors.HexColor(self.color))
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, 3, self.width, 3)


class ShadedBox(Flowable):
    """Metric card with label + value in a shaded container."""
    def __init__(self, label, value, width=140, bg="#F5F7FA", border="#1565C0"):
        super().__init__()
        self.label = label
        self.value = value
        self.bw = width
        self.bg = bg
        self.border = border

    def wrap(self, availWidth, availHeight):
        return self.bw, 48

    def draw(self):
        self.canv.setFillColor(colors.HexColor(self.bg))
        self.canv.setStrokeColor(colors.HexColor(self.border))
        self.canv.roundRect(0, 0, self.bw, 44, 4, fill=1, stroke=1)
        self.canv.setFont("Helvetica-Bold", 14)
        self.canv.setFillColor(colors.HexColor(self.border))
        self.canv.drawCentredString(self.bw / 2, 24, str(self.value))
        self.canv.setFont("Helvetica", 7)
        self.canv.setFillColor(colors.HexColor("#757575"))
        self.canv.drawCentredString(self.bw / 2, 10, self.label)


class RadarChartFlowable(Flowable):
    """Spider/radar chart for multi-dimensional score visualization."""
    def __init__(self, scores_dict, width=280, height=280):
        super().__init__()
        self.scores = scores_dict
        self.chart_w = width
        self.chart_h = height

    def wrap(self, availWidth, availHeight):
        return self.chart_w, self.chart_h

    def draw(self):
        labels = list(self.scores.keys())
        values = list(self.scores.values())
        n = len(labels)
        if n < 3:
            return
        cx, cy = self.chart_w / 2, self.chart_h / 2
        r = min(cx, cy) - 30
        angles = [2 * math.pi * i / n - math.pi / 2 for i in range(n)]

        # Grid circles
        self.canv.setStrokeColor(colors.HexColor("#E0E0E0"))
        self.canv.setLineWidth(0.3)
        for frac in [0.25, 0.5, 0.75, 1.0]:
            self.canv.circle(cx, cy, r * frac, fill=0)

        # Axes
        for angle in angles:
            x2 = cx + r * math.cos(angle)
            y2 = cy + r * math.sin(angle)
            self.canv.line(cx, cy, x2, y2)

        # Data polygon
        max_val = 10
        pts = []
        for i, val in enumerate(values):
            frac = val / max_val
            x = cx + r * frac * math.cos(angles[i])
            y = cy + r * frac * math.sin(angles[i])
            pts.append((x, y))

        if pts:
            self.canv.setFillColor(colors.Color(0.08, 0.35, 0.69, alpha=0.15))
            self.canv.setStrokeColor(colors.HexColor("#1565C0"))
            self.canv.setLineWidth(1.5)
            path = self.canv.beginPath()
            path.moveTo(pts[0][0], pts[0][1])
            for px, py in pts[1:]:
                path.lineTo(px, py)
            path.close()
            self.canv.drawPath(path, fill=1, stroke=1)
            # Dots
            self.canv.setFillColor(colors.HexColor("#1565C0"))
            for px, py in pts:
                self.canv.circle(px, py, 3, fill=1, stroke=0)

        # Labels
        self.canv.setFont("Helvetica", 6.5)
        self.canv.setFillColor(colors.HexColor("#424242"))
        for i, label in enumerate(labels):
            lx = cx + (r + 18) * math.cos(angles[i])
            ly = cy + (r + 18) * math.sin(angles[i])
            short = label[:18] + "..." if len(label) > 18 else label
            self.canv.drawCentredString(lx, ly - 3, short)


class DonutChartFlowable(Flowable):
    """Donut chart showing percentage splits (e.g., talk-time distribution)."""
    def __init__(self, segments, width=220, height=220):
        """segments: list of (label, value, hex_color)"""
        super().__init__()
        self.segments = segments
        self.chart_w = width
        self.chart_h = height

    def wrap(self, availWidth, availHeight):
        return self.chart_w + 120, self.chart_h

    def draw(self):
        cx, cy = self.chart_w / 2, self.chart_h / 2
        r_outer = min(cx, cy) - 10
        r_inner = r_outer * 0.55
        total = sum(v for _, v, _ in self.segments) or 1
        start_angle = 90

        for label, value, hex_color in self.segments:
            sweep = (value / total) * 360
            self.canv.setFillColor(colors.HexColor(hex_color))
            self.canv.setStrokeColor(colors.white)
            self.canv.setLineWidth(2)
            # Draw wedge
            path = self.canv.beginPath()
            path.moveTo(cx, cy)
            path.arcTo(cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer,
                       start_angle, sweep)
            path.close()
            self.canv.drawPath(path, fill=1, stroke=1)
            start_angle += sweep

        # Inner circle (donut hole)
        self.canv.setFillColor(colors.white)
        self.canv.setStrokeColor(colors.white)
        self.canv.circle(cx, cy, r_inner, fill=1, stroke=0)

        # Center text
        self.canv.setFont("Helvetica-Bold", 9)
        self.canv.setFillColor(colors.HexColor("#424242"))
        self.canv.drawCentredString(cx, cy + 2, "Talk Time")
        self.canv.setFont("Helvetica", 7)
        self.canv.drawCentredString(cx, cy - 10, "Distribution")

        # Legend
        legend_x = self.chart_w + 10
        legend_y = self.chart_h - 30
        for label, value, hex_color in self.segments:
            pct = (value / total) * 100
            self.canv.setFillColor(colors.HexColor(hex_color))
            self.canv.rect(legend_x, legend_y, 10, 10, fill=1, stroke=0)
            self.canv.setFont("Helvetica", 7.5)
            self.canv.setFillColor(colors.HexColor("#424242"))
            self.canv.drawString(legend_x + 14, legend_y + 1, f"{label}: {pct:.0f}%")
            legend_y -= 16


class HBarChartFlowable(Flowable):
    """Horizontal bar chart with labels and color-coded bars."""
    def __init__(self, data, width=440, bar_height=18, spacing=6, max_val=10):
        """data: list of (label, value, hex_color)"""
        super().__init__()
        self.data = data
        self.chart_w = width
        self.bar_h = bar_height
        self.spacing = spacing
        self.max_val = max_val

    def wrap(self, availWidth, availHeight):
        h = len(self.data) * (self.bar_h + self.spacing) + 10
        return self.chart_w, h

    def draw(self):
        label_w = self.chart_w * 0.30
        bar_w = self.chart_w * 0.55
        bar_x = label_w + 10
        y = len(self.data) * (self.bar_h + self.spacing)

        for label, value, hex_color in self.data:
            y -= (self.bar_h + self.spacing)
            # Background bar
            self.canv.setFillColor(colors.HexColor("#ECEFF1"))
            self.canv.roundRect(bar_x, y, bar_w, self.bar_h, 3, fill=1, stroke=0)
            # Filled bar
            pct = min(value / self.max_val, 1.0) if self.max_val else 0
            self.canv.setFillColor(colors.HexColor(hex_color))
            self.canv.roundRect(bar_x, y, bar_w * pct, self.bar_h, 3, fill=1, stroke=0)
            # Label
            self.canv.setFont("Helvetica", 7.5)
            self.canv.setFillColor(colors.HexColor("#424242"))
            short = label[:28] + ".." if len(label) > 28 else label
            self.canv.drawRightString(label_w, y + 5, short)
            # Value
            self.canv.setFont("Helvetica-Bold", 8)
            self.canv.setFillColor(colors.HexColor(hex_color))
            self.canv.drawString(bar_x + bar_w + 6, y + 5, f"{value:.1f}")


class LineChartFlowable(Flowable):
    """Line chart for sentiment over time or skill trends."""
    def __init__(self, data_points, width=420, height=180, y_label="Score", x_labels=None,
                 line_color="#1565C0", fill_color=None, y_range=None):
        """data_points: list of numeric values. x_labels: list of str labels for x-axis."""
        super().__init__()
        self.data = data_points
        self.chart_w = width
        self.chart_h = height
        self.y_label = y_label
        self.x_labels = x_labels
        self.line_color = line_color
        self.fill_color = fill_color
        self.y_range = y_range

    def wrap(self, availWidth, availHeight):
        return self.chart_w, self.chart_h + 30

    def draw(self):
        if not self.data or len(self.data) < 2:
            return
        margin_l, margin_b, margin_t, margin_r = 40, 28, 10, 10
        plot_w = self.chart_w - margin_l - margin_r
        plot_h = self.chart_h - margin_b - margin_t
        ox, oy = margin_l, margin_b

        y_min = self.y_range[0] if self.y_range else min(self.data)
        y_max = self.y_range[1] if self.y_range else max(self.data)
        if y_max == y_min:
            y_max = y_min + 1

        # Grid lines
        self.canv.setStrokeColor(colors.HexColor("#E0E0E0"))
        self.canv.setLineWidth(0.3)
        for i in range(5):
            gy = oy + plot_h * i / 4
            self.canv.line(ox, gy, ox + plot_w, gy)
            val = y_min + (y_max - y_min) * i / 4
            self.canv.setFont("Helvetica", 6)
            self.canv.setFillColor(colors.HexColor("#9E9E9E"))
            self.canv.drawRightString(ox - 4, gy - 3, f"{val:.1f}")

        # Axes
        self.canv.setStrokeColor(colors.HexColor("#BDBDBD"))
        self.canv.setLineWidth(0.5)
        self.canv.line(ox, oy, ox + plot_w, oy)
        self.canv.line(ox, oy, ox, oy + plot_h)

        # Plot points
        n = len(self.data)
        pts = []
        for i, val in enumerate(self.data):
            x = ox + plot_w * i / (n - 1)
            y = oy + plot_h * (val - y_min) / (y_max - y_min)
            pts.append((x, y))

        # Fill area under curve
        if self.fill_color and pts:
            path = self.canv.beginPath()
            path.moveTo(pts[0][0], oy)
            for px, py in pts:
                path.lineTo(px, py)
            path.lineTo(pts[-1][0], oy)
            path.close()
            self.canv.setFillColor(colors.HexColor(self.fill_color))
            self.canv.drawPath(path, fill=1, stroke=0)

        # Line
        self.canv.setStrokeColor(colors.HexColor(self.line_color))
        self.canv.setLineWidth(1.5)
        path = self.canv.beginPath()
        path.moveTo(pts[0][0], pts[0][1])
        for px, py in pts[1:]:
            path.lineTo(px, py)
        self.canv.drawPath(path, fill=0, stroke=1)

        # Dots
        self.canv.setFillColor(colors.HexColor(self.line_color))
        for px, py in pts:
            self.canv.circle(px, py, 2.5, fill=1, stroke=0)

        # X labels
        if self.x_labels:
            self.canv.setFont("Helvetica", 5.5)
            self.canv.setFillColor(colors.HexColor("#757575"))
            for i, lbl in enumerate(self.x_labels[:n]):
                x = ox + plot_w * i / (n - 1)
                short = lbl[:10]
                self.canv.drawCentredString(x, oy - 12, short)

        # Y axis label
        self.canv.setFont("Helvetica", 6.5)
        self.canv.setFillColor(colors.HexColor("#757575"))
        self.canv.saveState()
        self.canv.translate(8, oy + plot_h / 2)
        self.canv.rotate(90)
        self.canv.drawCentredString(0, 0, self.y_label)
        self.canv.restoreState()


class HeatmapFlowable(Flowable):
    """Heatmap matrix: rows x columns with color-coded cells."""
    def __init__(self, row_labels, col_labels, data_matrix, width=420, cell_h=22):
        """data_matrix[row][col] = value (0-10 or 0-1). Colors interpolated."""
        super().__init__()
        self.rows = row_labels
        self.cols = col_labels
        self.data = data_matrix
        self.chart_w = width
        self.cell_h = cell_h

    def wrap(self, availWidth, availHeight):
        h = (len(self.rows) + 1) * self.cell_h + 10
        return self.chart_w, h

    def draw(self):
        n_rows = len(self.rows)
        n_cols = len(self.cols)
        label_w = 90
        cell_w = (self.chart_w - label_w) / max(n_cols, 1)
        y_start = n_rows * self.cell_h

        # Column headers
        self.canv.setFont("Helvetica-Bold", 6.5)
        self.canv.setFillColor(colors.HexColor("#424242"))
        for j, col in enumerate(self.cols):
            x = label_w + j * cell_w + cell_w / 2
            short = col[:12]
            self.canv.drawCentredString(x, y_start + 4, short)

        # Rows
        for i, row_label in enumerate(self.rows):
            y = y_start - (i + 1) * self.cell_h
            # Row label
            self.canv.setFont("Helvetica", 7)
            self.canv.setFillColor(colors.HexColor("#424242"))
            self.canv.drawRightString(label_w - 6, y + self.cell_h / 2 - 3, row_label[:14])
            # Cells
            for j in range(n_cols):
                val = self.data[i][j] if i < len(self.data) and j < len(self.data[i]) else 0
                # Color: red(0) -> yellow(5) -> green(10)
                if val >= 7:
                    hex_c = "#43A047"
                elif val >= 4:
                    hex_c = "#FF8F00"
                else:
                    hex_c = "#E53935"
                alpha = max(0.2, min(1.0, val / 10))
                x = label_w + j * cell_w
                self.canv.setFillColor(colors.HexColor(hex_c))
                self.canv.setStrokeColor(colors.white)
                self.canv.setLineWidth(1)
                self.canv.rect(x, y, cell_w, self.cell_h, fill=1, stroke=1)
                # Value text
                self.canv.setFont("Helvetica-Bold", 7)
                self.canv.setFillColor(colors.white)
                display = f"{val:.0f}" if val == int(val) else f"{val:.1f}"
                self.canv.drawCentredString(x + cell_w / 2, y + self.cell_h / 2 - 3, display)


class FunnelFlowable(Flowable):
    """Funnel visualization for stage-wise drop-off."""
    def __init__(self, stages, width=400, height=200):
        """stages: list of (label, score_0_10, hex_color)"""
        super().__init__()
        self.stages = stages
        self.chart_w = width
        self.chart_h = height

    def wrap(self, availWidth, availHeight):
        return self.chart_w, self.chart_h + 10

    def draw(self):
        n = len(self.stages)
        if not n:
            return
        step_h = self.chart_h / n
        max_w = self.chart_w * 0.8
        cx = self.chart_w / 2
        shrink = 0.15 / max(n, 1) * 3  # Adaptive shrink based on # of stages

        for i, (label, score, hex_color) in enumerate(self.stages):
            pct = score / 10
            top_w = max_w * (1 - i * shrink)
            bot_w = max_w * (1 - (i + 1) * shrink)
            y_top = self.chart_h - i * step_h
            y_bot = self.chart_h - (i + 1) * step_h
            gap = 2  # Small gap between stages

            # Trapezoid with rounded appearance
            base_color = colors.HexColor(hex_color)
            # Slightly lighter version for background
            self.canv.setFillColor(base_color)
            self.canv.setStrokeColor(colors.white)
            self.canv.setLineWidth(2)
            path = self.canv.beginPath()
            path.moveTo(cx - top_w / 2, y_top - gap)
            path.lineTo(cx + top_w / 2, y_top - gap)
            path.lineTo(cx + bot_w / 2, y_bot + gap)
            path.lineTo(cx - bot_w / 2, y_bot + gap)
            path.close()
            self.canv.drawPath(path, fill=1, stroke=1)

            # Label and score
            mid_y = (y_top + y_bot) / 2
            self.canv.setFont("Helvetica-Bold", 9)
            self.canv.setFillColor(colors.white)
            self.canv.drawCentredString(cx, mid_y + 3, f"{label}")
            # Score bar indicator
            bar_w = min(top_w, bot_w) * 0.5
            bar_h = 4
            bar_y = mid_y - 10
            # Background bar
            self.canv.setFillColor(colors.Color(1, 1, 1, 0.3))
            self.canv.rect(cx - bar_w / 2, bar_y, bar_w, bar_h, fill=1, stroke=0)
            # Filled bar
            self.canv.setFillColor(colors.Color(1, 1, 1, 0.8))
            self.canv.rect(cx - bar_w / 2, bar_y, bar_w * pct, bar_h, fill=1, stroke=0)


class GaugeFlowable(Flowable):
    """Half-circle gauge / speedometer for deal momentum."""
    def __init__(self, value, max_val=100, label="Deal Momentum", width=200, height=120):
        super().__init__()
        self.value = value
        self.max_val = max_val
        self.label = label
        self.gauge_w = width
        self.gauge_h = height

    def wrap(self, availWidth, availHeight):
        # Total height = arc area + space for value text + label text below
        return self.gauge_w, self.gauge_h + 45

    def draw(self):
        total_h = self.gauge_h + 45
        cx = self.gauge_w / 2
        # Position arc center so everything fits: 40px for text below + arc radius
        r = min(self.gauge_w / 2 - 20, self.gauge_h - 10)
        cy = 42  # Base of the arc — leaves room for value + label below

        # Background arc segments (red/yellow/green) — thick for visibility
        segments = [
            (180, 60, "#E53935"),   # 0-33%  Cold
            (120, 60, "#FF8F00"),   # 33-67% Warm
            (60, 60, "#43A047"),    # 67-100% Hot
        ]
        for start, sweep, hex_c in segments:
            self.canv.setStrokeColor(colors.HexColor(hex_c))
            self.canv.setLineWidth(16)
            self.canv.arc(cx - r, cy - r, cx + r, cy + r, start, sweep)

        # White separator arcs between segments
        for start_angle in [120, 60]:
            self.canv.setStrokeColor(colors.HexColor("#FFFFFF"))
            self.canv.setLineWidth(18)
            self.canv.arc(cx - r, cy - r, cx + r, cy + r, start_angle, 1)

        # Needle
        pct = min(self.value / self.max_val, 1.0) if self.max_val else 0
        angle_deg = 180 - pct * 180
        angle_rad = math.radians(angle_deg)
        needle_len = r - 22
        nx = cx + needle_len * math.cos(angle_rad)
        ny = cy + needle_len * math.sin(angle_rad)
        # Shadow
        self.canv.setStrokeColor(colors.HexColor("#BDBDBD"))
        self.canv.setLineWidth(3)
        self.canv.line(cx + 1, cy - 1, nx + 1, ny - 1)
        # Needle
        self.canv.setStrokeColor(colors.HexColor("#212121"))
        self.canv.setLineWidth(2.5)
        self.canv.line(cx, cy, nx, ny)
        # Center dot
        self.canv.setFillColor(colors.HexColor("#212121"))
        self.canv.circle(cx, cy, 5, fill=1, stroke=0)
        self.canv.setFillColor(colors.HexColor("#FFFFFF"))
        self.canv.circle(cx, cy, 2, fill=1, stroke=0)

        # Value text — placed BELOW the arc center with proper spacing
        val_color = "#43A047" if pct >= 0.67 else "#FF8F00" if pct >= 0.33 else "#E53935"
        self.canv.setFont("Helvetica-Bold", 14)
        self.canv.setFillColor(colors.HexColor(val_color))
        self.canv.drawCentredString(cx, cy - 18, f"{self.value}%")

        # Label — below value
        self.canv.setFont("Helvetica", 7)
        self.canv.setFillColor(colors.HexColor("#757575"))
        self.canv.drawCentredString(cx, cy - 30, self.label)

        # Zone labels at arc edges
        self.canv.setFont("Helvetica-Bold", 7)
        self.canv.setFillColor(colors.HexColor("#E53935"))
        self.canv.drawString(cx - r - 2, cy + 5, "Cold")
        self.canv.setFillColor(colors.HexColor("#FF8F00"))
        self.canv.drawCentredString(cx, cy + r + 6, "Warm")
        self.canv.setFillColor(colors.HexColor("#43A047"))
        self.canv.drawRightString(cx + r + 2, cy + 5, "Hot")


class QuadrantFlowable(Flowable):
    """2D quadrant chart (e.g., Trust vs Pressure)."""
    def __init__(self, points, width=300, height=300,
                 x_label="Trust-Building →", y_label="Pressure →"):
        """points: list of (label, x_val, y_val) where 0-10 scale."""
        super().__init__()
        self.points = points
        self.chart_w = width
        self.chart_h = height
        self.x_label = x_label
        self.y_label = y_label

    def wrap(self, availWidth, availHeight):
        return self.chart_w + 20, self.chart_h + 20

    def draw(self):
        margin = 35
        pw = self.chart_w - 2 * margin
        ph = self.chart_h - 2 * margin
        ox, oy = margin, margin

        # Quadrant backgrounds
        half_w, half_h = pw / 2, ph / 2
        quads = [
            (ox, oy + half_h, half_w, half_h, "#FFEBEE", "High Pressure\nLow Trust"),
            (ox + half_w, oy + half_h, half_w, half_h, "#FFF8E1", "High Pressure\nHigh Trust"),
            (ox, oy, half_w, half_h, "#E3F2FD", "Low Pressure\nLow Trust"),
            (ox + half_w, oy, half_w, half_h, "#E8F5E9", "Low Pressure\nHigh Trust"),
        ]
        for qx, qy, qw, qh, bg, qlabel in quads:
            self.canv.setFillColor(colors.HexColor(bg))
            self.canv.rect(qx, qy, qw, qh, fill=1, stroke=0)
            self.canv.setFont("Helvetica", 5.5)
            self.canv.setFillColor(colors.HexColor("#BDBDBD"))
            lines = qlabel.split('\n')
            for li, line in enumerate(lines):
                self.canv.drawCentredString(qx + qw / 2, qy + qh / 2 - li * 8 + 4, line)

        # Grid
        self.canv.setStrokeColor(colors.HexColor("#BDBDBD"))
        self.canv.setLineWidth(0.5)
        self.canv.setDash(2, 2)
        self.canv.line(ox + half_w, oy, ox + half_w, oy + ph)
        self.canv.line(ox, oy + half_h, ox + pw, oy + half_h)
        self.canv.setDash()

        # Border
        self.canv.setStrokeColor(colors.HexColor("#9E9E9E"))
        self.canv.setLineWidth(0.5)
        self.canv.rect(ox, oy, pw, ph, fill=0, stroke=1)

        # Points
        for label, xv, yv in self.points:
            px = ox + pw * min(xv / 10, 1.0)
            py = oy + ph * min(yv / 10, 1.0)
            self.canv.setFillColor(colors.HexColor("#1565C0"))
            self.canv.circle(px, py, 5, fill=1, stroke=0)
            self.canv.setFont("Helvetica-Bold", 6)
            self.canv.setFillColor(colors.HexColor("#0D47A1"))
            self.canv.drawString(px + 7, py - 2, label[:20])

        # Ideal zone marker
        ideal_x = ox + pw * 0.75
        ideal_y = oy + ph * 0.25
        self.canv.setStrokeColor(colors.HexColor("#43A047"))
        self.canv.setLineWidth(1)
        self.canv.setDash(3, 3)
        self.canv.circle(ideal_x, ideal_y, 18, fill=0, stroke=1)
        self.canv.setDash()
        self.canv.setFont("Helvetica", 5)
        self.canv.setFillColor(colors.HexColor("#2E7D32"))
        self.canv.drawCentredString(ideal_x, ideal_y - 24, "IDEAL ZONE")

        # Axis labels
        self.canv.setFont("Helvetica-Bold", 7)
        self.canv.setFillColor(colors.HexColor("#424242"))
        self.canv.drawCentredString(ox + pw / 2, oy - 18, self.x_label)
        self.canv.saveState()
        self.canv.translate(ox - 20, oy + ph / 2)
        self.canv.rotate(90)
        self.canv.drawCentredString(0, 0, self.y_label)
        self.canv.restoreState()


class StackedBarFlowable(Flowable):
    """Stacked horizontal bar chart for question quality distribution etc."""
    def __init__(self, categories, width=420, bar_height=24):
        """categories: list of (label, [(segment_label, value, hex_color), ...])"""
        super().__init__()
        self.categories = categories
        self.chart_w = width
        self.bar_h = bar_height

    def wrap(self, availWidth, availHeight):
        h = len(self.categories) * (self.bar_h + 12) + 30
        return self.chart_w, h

    def draw(self):
        label_w = 100
        bar_w = self.chart_w - label_w - 20
        y = len(self.categories) * (self.bar_h + 12)

        # Legend at top
        legend_x = label_w
        all_segments = set()
        for _, segs in self.categories:
            for sl, _, sc in segs:
                all_segments.add((sl, sc))
        self.canv.setFont("Helvetica", 6)
        lx = legend_x
        for sl, sc in sorted(all_segments):
            self.canv.setFillColor(colors.HexColor(sc))
            self.canv.rect(lx, y + 4, 8, 8, fill=1, stroke=0)
            self.canv.setFillColor(colors.HexColor("#424242"))
            self.canv.drawString(lx + 10, y + 4, sl)
            lx += len(sl) * 5 + 24

        for cat_label, segments in self.categories:
            y -= (self.bar_h + 12)
            total = sum(v for _, v, _ in segments) or 1

            # Label
            self.canv.setFont("Helvetica", 7.5)
            self.canv.setFillColor(colors.HexColor("#424242"))
            self.canv.drawRightString(label_w - 6, y + self.bar_h / 2 - 3, cat_label[:16])

            # Stacked bar
            x = label_w
            for seg_label, value, hex_color in segments:
                w = bar_w * (value / total)
                self.canv.setFillColor(colors.HexColor(hex_color))
                self.canv.setStrokeColor(colors.white)
                self.canv.setLineWidth(0.5)
                self.canv.rect(x, y, w, self.bar_h, fill=1, stroke=1)
                # Value inside bar
                if w > 25:
                    self.canv.setFont("Helvetica-Bold", 6)
                    self.canv.setFillColor(colors.white)
                    pct = (value / total) * 100
                    self.canv.drawCentredString(x + w / 2, y + self.bar_h / 2 - 3, f"{pct:.0f}%")
                x += w


# ════════════════════════════════════════════════════════
# MAIN SUMMARIZER CLASS
# ════════════════════════════════════════════════════════

class RealEstateSalesMeetingSummarizer:
    """Combined best-of-both RAG coaching evaluation engine."""

    def __init__(self, transcript_file: str = "conversation.json"):
        load_dotenv()
        self.transcript_file = transcript_file
        self.config = configparser.ConfigParser()
        self.config.read('meeting_config.ini')
        self.summaries_folder = self.config.get('Paths', 'summaries_folder', fallback='RealEstateMeetingRecords')
        os.makedirs(self.summaries_folder, exist_ok=True)

        self._load_environment_variables()
        self._setup_llm_config()
        self._setup_professional_styles()
        self._setup_rag_system()
        self.viz_logger = VisualizationLogger(report_type="real_estate_sales_coaching")

    # ── Environment & Config ──

    def _load_environment_variables(self):
        self.llm_provider = os.getenv("LLM_PROVIDER").strip().lower()
        self.llm_api_key = os.getenv("LLM_API_KEY")
        self.llm_model = os.getenv("LLM_MODEL")
        self.llm_base_url = os.getenv("LLM_BASE_URL")
        self.llm_temperature = float(os.getenv("LLM_TEMPERATURE"))
        self.llm_max_tokens = int(os.getenv("LLM_MAX_TOKENS"))
        self.llm_timeout = int(os.getenv("LLM_TIMEOUT"))

    def _setup_llm_config(self):
        self.llm_config = LLMConfig(
            provider=self.llm_provider,
            api_key=self.llm_api_key,
            model=self.llm_model,
            base_url=self.llm_base_url,
            temperature=self.llm_temperature,
            max_tokens=self.llm_max_tokens,
            timeout=self.llm_timeout,
        )

    # ── Professional Styles ──

    def _setup_professional_styles(self):
        """Enterprise color palette + compact paragraph style registry."""
        self.C = {
            'primary': colors.HexColor("#1565C0"),
            'primary_dark': colors.HexColor("#0D47A1"),
            'primary_light': colors.HexColor("#42A5F5"),
            'accent': colors.HexColor("#FF8F00"),
            'success': colors.HexColor("#2E7D32"),
            'success_bg': colors.HexColor("#E8F5E9"),
            'danger': colors.HexColor("#C62828"),
            'danger_bg': colors.HexColor("#FFEBEE"),
            'warning': colors.HexColor("#F57F17"),
            'warning_bg': colors.HexColor("#FFF8E1"),
            'dark': colors.HexColor("#263238"),
            'medium': colors.HexColor("#546E7A"),
            'light': colors.HexColor("#78909C"),
            'bg': colors.HexColor("#F5F7FA"),
            'white': colors.white,
            'border': colors.HexColor("#CFD8DC"),
        }

        base = getSampleStyleSheet()

        def ps(name, **kw):
            return ParagraphStyle(name, parent=base['Normal'], **kw)

        self.ps = {
            'cover_title': ps('cover_title', fontName='Helvetica-Bold', fontSize=26,
                              textColor=self.C['primary_dark'], alignment=TA_CENTER, spaceAfter=6),
            'cover_sub': ps('cover_sub', fontName='Helvetica', fontSize=12,
                            textColor=self.C['medium'], alignment=TA_CENTER, spaceAfter=4),
            'sec': ps('sec', fontName='Helvetica-Bold', fontSize=14,
                       textColor=self.C['primary_dark'], spaceBefore=14, spaceAfter=6),
            'subsec': ps('subsec', fontName='Helvetica-Bold', fontSize=11,
                          textColor=self.C['primary'], spaceBefore=8, spaceAfter=4),
            'body': ps('body', fontName='Helvetica', fontSize=9.5,
                        textColor=self.C['dark'], leading=13, spaceAfter=3),
            'body_bold': ps('body_bold', fontName='Helvetica-Bold', fontSize=9.5,
                             textColor=self.C['dark'], leading=13, spaceAfter=3),
            'bullet': ps('bullet', fontName='Helvetica', fontSize=9.5,
                          textColor=self.C['dark'], leftIndent=18, leading=13, spaceAfter=2),
            'bullet_success': ps('bullet_success', fontName='Helvetica', fontSize=9.5,
                                  textColor=self.C['success'], leftIndent=18, leading=13, spaceAfter=2),
            'bullet_danger': ps('bullet_danger', fontName='Helvetica', fontSize=9.5,
                                 textColor=self.C['danger'], leftIndent=18, leading=13, spaceAfter=2),
            'bullet_accent': ps('bullet_accent', fontName='Helvetica', fontSize=9.5,
                                 textColor=self.C['accent'], leftIndent=18, leading=13, spaceAfter=2),
            'th': ps('th', fontName='Helvetica-Bold', fontSize=9,
                      textColor=self.C['white'], alignment=TA_CENTER),
            'td': ps('td', fontName='Helvetica', fontSize=9,
                      textColor=self.C['dark'], leading=12),
            'td_c': ps('td_c', fontName='Helvetica', fontSize=9,
                        textColor=self.C['dark'], alignment=TA_CENTER, leading=12),
            'ts': ps('ts', fontName='Helvetica-Bold', fontSize=8.5,
                      textColor=self.C['primary'], spaceAfter=2),
            'quote': ps('quote', fontName='Helvetica-Oblique', fontSize=9,
                         textColor=self.C['medium'], leftIndent=18, spaceAfter=3),
            'score_big': ps('score_big', fontName='Helvetica-Bold', fontSize=28,
                             textColor=self.C['primary_dark'], alignment=TA_CENTER),
            'score_label': ps('score_label', fontName='Helvetica', fontSize=8,
                               textColor=self.C['light'], alignment=TA_CENTER),
            'tier': ps('tier', fontName='Helvetica-Bold', fontSize=16,
                        textColor=self.C['primary_dark'], alignment=TA_CENTER),
            'conf': ps('conf', fontName='Helvetica', fontSize=7,
                        textColor=self.C['light'], alignment=TA_CENTER),
        }

    # ── RAG System (from Final_backup.py — robust implementation) ──

    def _setup_rag_system(self):
        self.knowledge_base_path = "knowledge_base.docx"
        self.faiss_index_path = "knowledge_base.faiss"
        self.chunks_path = "knowledge_base_chunks.pkl"
        self.embedding_model_name = "all-MiniLM-L6-v2"
        self.embedding_model = None
        self.faiss_index = None
        self.knowledge_chunks: List[KnowledgeChunk] = []

        if not EMBEDDINGS_AVAILABLE or not FAISS_AVAILABLE:
            print("RAG components not available. System will work without knowledge base.")
            return

        try:
            print(f"Loading embedding model: {self.embedding_model_name}...")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
        except Exception as e:
            print(f"Failed to load embedding model: {e}")
            return

        # Try loading existing index (check both possible pkl filenames)
        if os.path.exists(self.faiss_index_path):
            pkl_path = None
            for candidate in [self.chunks_path, "knowledge_chunks.pkl"]:
                if os.path.exists(candidate):
                    pkl_path = candidate
                    break
            if pkl_path:
                try:
                    self.faiss_index = faiss.read_index(self.faiss_index_path)
                    with open(pkl_path, 'rb') as f:
                        self.knowledge_chunks = pickle.load(f)
                    print(f"Loaded {len(self.knowledge_chunks)} knowledge chunks from existing index.")
                    return
                except Exception as e:
                    print(f"Failed to load existing index: {e}. Will rebuild...")

        # Build new index from docx
        if not os.path.exists(self.knowledge_base_path) or not DOCX_AVAILABLE:
            print(f"Knowledge base not found: {self.knowledge_base_path}")
            return

        print(f"Building FAISS index from {self.knowledge_base_path}...")
        self._build_faiss_index()

    def _load_knowledge_base_text(self) -> str:
        """Load knowledge base from .docx — extracts paragraphs AND tables."""
        try:
            doc = Document(self.knowledge_base_path)
            full_text = []
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    full_text.append(text)
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        text = cell.text.strip()
                        if text:
                            full_text.append(text)
            content = "\n".join(full_text)
            print(f"Loaded {len(content)} characters from knowledge base.")
            return content
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
            return ""

    def _semantic_chunk(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """Chunk text semantically with overlap for better retrieval."""
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        chunks = []
        current_chunk = []
        current_length = 0

        for para in paragraphs:
            para_length = len(para)
            if para_length > chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sent in sentences:
                    if current_length + len(sent) > chunk_size and current_chunk:
                        chunks.append(' '.join(current_chunk))
                        overlap_text = ' '.join(current_chunk[-2:]) if len(current_chunk) > 1 else (current_chunk[-1] if current_chunk else "")
                        current_chunk = [overlap_text, sent] if overlap_text else [sent]
                        current_length = len(overlap_text) + len(sent)
                    else:
                        current_chunk.append(sent)
                        current_length += len(sent)
            else:
                if current_length + para_length > chunk_size and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    overlap_text = ' '.join(current_chunk[-2:]) if len(current_chunk) > 1 else (current_chunk[-1] if current_chunk else "")
                    current_chunk = [overlap_text, para] if overlap_text else [para]
                    current_length = len(overlap_text) + para_length
                else:
                    current_chunk.append(para)
                    current_length += para_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks

    def _build_faiss_index(self):
        """Build FAISS index with semantic chunking."""
        kb_content = self._load_knowledge_base_text()
        if not kb_content:
            return

        text_chunks = self._semantic_chunk(kb_content, chunk_size=500, overlap=100)
        print(f"Created {len(text_chunks)} semantic chunks.")

        embeddings = self.embedding_model.encode(text_chunks, show_progress_bar=False, convert_to_numpy=True)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))

        self.faiss_index = index
        self.knowledge_chunks = [
            KnowledgeChunk(chunk_id=i, text=chunk, embedding=emb.tolist())
            for i, (chunk, emb) in enumerate(zip(text_chunks, embeddings))
        ]

        faiss.write_index(index, self.faiss_index_path)
        with open(self.chunks_path, 'wb') as f:
            pickle.dump(self.knowledge_chunks, f)
        print(f"FAISS index built with {len(self.knowledge_chunks)} chunks.")

    def _retrieve_relevant_knowledge(self, query: str, top_k: int = 8) -> List[KnowledgeChunk]:
        """Retrieve most relevant knowledge chunks for a query."""
        if not self.faiss_index or not self.knowledge_chunks or not self.embedding_model:
            return []
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        distances, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
        retrieved = []
        for idx in indices[0]:
            if 0 <= idx < len(self.knowledge_chunks):
                retrieved.append(self.knowledge_chunks[idx])
        return retrieved

    # ── Transcript Loading ──

    def _load_transcript(self) -> Optional[TranscriptData]:
        path = Path(self.transcript_file)
        if not path.exists():
            print(f"Transcript file not found: {self.transcript_file}")
            return None

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        segments = []
        speakers = set()
        total_dur = 0.0

        for i, entry in enumerate(data):
            speaker = entry.get("speaker", "Unknown")
            speakers.add(speaker)
            seg = TranscriptSegment(
                speaker_id=speaker, speaker_name=speaker,
                transcript=entry.get("transcript", ""),
                segment_id=i,
                start=entry.get("start", 0.0),
                end=entry.get("end", 0.0),
                duration=entry.get("duration", 0.0),
            )
            segments.append(seg)
            total_dur += seg.duration

        return TranscriptData(
            transcripts=segments,
            agenda="Real Estate Listing Presentation & Coaching Evaluation",
            summary_info={
                "total_segments": len(segments),
                "unique_speakers": list(speakers),
                "total_duration_seconds": total_dur,
            },
        )

    def _format_transcript_for_analysis(self, data: TranscriptData) -> str:
        lines = []
        for seg in data.transcripts:
            start_m, start_s = int(seg.start // 60), int(seg.start % 60)
            end_m, end_s = int(seg.end // 60), int(seg.end % 60)
            ts = f"[{start_m}:{start_s:02d}-{end_m}:{end_s:02d}]"
            lines.append(f"{ts} {seg.speaker_name}: {seg.transcript}")
        return "\n".join(lines)

    @staticmethod
    def _convert_timestamps_to_mmss(text: str) -> str:
        """Convert any [X.Xs-Y.Ys] or [Xs-Ys] timestamp patterns to [M:SS-M:SS] format."""
        def _seconds_to_mmss(match):
            try:
                start_raw = match.group(1)
                end_raw = match.group(2)
                start_sec = float(start_raw.rstrip('s'))
                end_sec = float(end_raw.rstrip('s'))
                s_m, s_s = int(start_sec // 60), int(start_sec % 60)
                e_m, e_s = int(end_sec // 60), int(end_sec % 60)
                return f"[{s_m}:{s_s:02d}-{e_m}:{e_s:02d}]"
            except (ValueError, AttributeError):
                return match.group(0)
        # Match patterns like [123.5s-145.0s], [0.0s-21.5s], [123.5-145.0], [0.0s - 21.5s]
        text = re.sub(r'\[(\d+\.?\d*s?)\s*-\s*(\d+\.?\d*s?)\]', _seconds_to_mmss, text)
        return text

    # ── LLM Call ──

    def _call_llm(self, prompt: str) -> Optional[str]:
        url = f"{self.llm_config.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.llm_config.api_key}",
        }
        payload = {
            "model": self.llm_config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.llm_config.temperature,
            "max_tokens": self.llm_config.max_tokens,
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.llm_config.timeout)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"LLM call failed: {e}")
            traceback.print_exc()
            return None

    # ════════════════════════════════════════════════════════
    # UNIFIED LLM PROMPT — BRUTALLY HONEST EVALUATION
    # ════════════════════════════════════════════════════════

    def _generate_unified_llm_analysis(self, transcript_text: str, knowledge_context: str) -> Optional[str]:
        prompt = f"""You are a ruthless senior sales performance auditor and executive coach with 25+ years of real estate sales coaching. You do NOT coddle agents. Your job is to protect BOTH the client's financial interests AND the agent's career by delivering unvarnished truth. A mediocre agent who thinks they're good will lose deals and damage clients. Your honesty is an act of professional respect.

YOUR MANDATE: Deliver a BRUTALLY HONEST, forensic-level performance audit. If the agent performed well, acknowledge it with specific evidence — but if they failed, NAME the failure, QUOTE the exact moment, and explain the REVENUE IMPACT. Do NOT sugarcoat. Do NOT use filler phrases like "overall, the agent did a good job." Every sentence must earn its place with evidence or insight. Mediocrity is not acceptable — it costs clients money and agents their reputation.

KNOWLEDGE BASE CONTEXT (use to benchmark against industry best practices):
{knowledge_context if knowledge_context else "No knowledge base available — rely on expert judgment."}

FULL TRANSCRIPT WITH TIMESTAMPS:
{transcript_text}

═══════════════════════════════════════════════
PRODUCE EXACTLY 18 SECTIONS + ACTION ITEMS USING ## HEADERS
═══════════════════════════════════════════════

## 1. Executive Brief
Write a 4-6 sentence ruthless executive summary. State who the agent is, what the meeting was about, and deliver a VERDICT — not a compliment. Was this a competent presentation or a missed-opportunity failure? Include the single most impressive moment AND the single worst failure. End with a one-line prognosis: will this deal close, and at what cost? Be direct — the agent's income depends on hearing the truth.

## 2. Performance Score Dashboard
Rate EACH dimension on a strict 1-10 scale using PIPE-DELIMITED format:
Label|X|10|One-sentence justification with specific evidence|AI Confidence: High/Medium/Low

IMPORTANT: The 5th pipe field MUST contain an AI confidence indicator (High/Medium/Low) reflecting how much transcript evidence supports this score.

Required dimensions:
Communication Effectiveness|?|10|...|AI Confidence: ...
Client Engagement & Rapport|?|10|...|AI Confidence: ...
Objection Handling Skill|?|10|...|AI Confidence: ...
Needs Discovery Depth|?|10|...|AI Confidence: ...
Closing Ability & Deal Progression|?|10|...|AI Confidence: ...
Confidence & Authority|?|10|...|AI Confidence: ...
Value Proposition Clarity|?|10|...|AI Confidence: ...
Risk Management Awareness|?|10|...|AI Confidence: ...
Emotional Intelligence|?|10|...|AI Confidence: ...
Listening Quality|?|10|...|AI Confidence: ...

Then provide:
OVERALL|?|10|Weighted assessment considering all dimensions above|AI Confidence: ...

Be HARSH. A 7/10 means "competent but with clear gaps." An 8+ requires exceptional evidence. A 5 or below means "this actively hurt the deal." Do not inflate scores to be nice — that helps nobody.

## 3. Behavioral Breakdown
Analyze the agent's behavior in granular detail using ### subheadings:
### Communication Patterns
### Body Language Signals (from verbal cues)
### Questioning Technique
### Active Listening Indicators
### Persuasion & Influence Tactics

For EACH, cite specific timestamps and quotes. Identify PATTERNS, not just isolated moments. Be clinical — patterns reveal whether this agent can repeat success or is getting lucky.

## 4. Strengths (Evidence-Anchored)
List 3-5 genuine strengths. For EACH strength, use this exact structure:

STRENGTH: [Name of the strength in 3-5 words]
TIMESTAMP: [exact time range]
EVIDENCE: "Exact quote from transcript proving this strength"
IMPACT: Why this specific strength matters for deal progression and client trust

Do NOT pad this section. If the agent only has 2 real strengths, list 2. Do NOT combine multiple observations into one bullet — each strength must be a separate, clearly structured entry.

## 5. Missed Opportunities & Critical Failures
This is the MOST IMPORTANT section. The agent NEEDS to hear this. For each failure, use this exact structure:

FAILURE TITLE: [Clear 3-5 word name for this failure]
TIMESTAMP: [exact time range]
QUOTE: "What the agent actually said"
WHAT HAPPENED: Describe the failure in 1-2 sentences
WHAT A TOP AGENT WOULD DO: Specific alternative approach with example script
REVENUE IMPACT: How this failure could cost money or trust — be specific with dollar estimates where possible

Provide AT LEAST 3 failures. Be SPECIFIC and CONSTRUCTIVE. Vague criticisms like "could have been better" waste everyone's time.

## 6. Tactical Corrections & Coaching Playbook
This section combines tactical fixes for past mistakes AND coaching drills for future scenarios into ONE unified playbook. For EACH entry (provide 5-7 entries total), use this structure:

SCENARIO: Brief description of the situation or recurring pattern
CLIENT TRIGGER: What the client said/did that created the opening
RISK: What happens if the agent keeps doing this — quantify the cost
ASSERTIVE APPROACH: "Exact word-for-word script using direct, authoritative framing"
CONSULTATIVE APPROACH: "Exact word-for-word script using collaborative, question-based framing"
WHY IT WORKS: Sales psychology principle behind both approaches

Each entry must address a DIFFERENT skill gap. No repetition from other sections.

## 7. 7-Day Intensive Coaching Plan
Provide a comprehensive day-by-day coaching plan for the next 7 days. Each day must include a focus area, a specific drill or exercise, and a measurable outcome. This replaces both short-term focus and long-term roadmap — compress ALL development priorities into one aggressive 7-day sprint.

Format each day as:
Day X: [Focus Area] -- [Specific Drill/Exercise] -- [Measurable Outcome/Success Metric]

Cover all 7 days. Each day must target a DIFFERENT skill gap identified in this evaluation. Include:
- At least 2 role-play exercises across the 7 days
- At least 1 script-writing exercise
- At least 1 objection-handling drill
- At least 1 listening/discovery practice session
Make each drill SPECIFIC and ACTIONABLE — not vague advice like "practice more." Include exact scenarios, word counts, or rep counts where possible.

## 8. Decisions & Commitments Tracker
Split into two clearly labeled subsections:
Decisions Made:
- List every decision made during the meeting with context
Commitments Given:
- List every commitment/promise made by either party

## Action Items
Create a table of specific action items arising from the meeting:
| Task Description | Responsible Person | Deadline | Priority | Status |
|---|---|---|---|---|
| [Specific task] | [Name] | [Date if mentioned, otherwise "No due date mentioned"] | [High/Medium/Low] | Pending |

Include 4-6 concrete action items based on what was discussed and agreed.

## 9. Deal Intelligence Summary
Use PIPE-DELIMITED format for key metrics:
Deal Closing Probability|XX%|Justification
Client Motivation Level|Strong/Moderate/Weak|Evidence
Urgency Level|High/Medium/Low|Evidence
Price Sensitivity|High/Medium/Low|Evidence
Competition Risk|High/Medium/Low|Evidence
Trust Level Established|Strong/Moderate/Weak|Evidence
Objection Resolution Rate|XX%|Evidence

Then add:
Recommended Next Steps:
1. [Specific action]
2. [Specific action]
3. [Specific action]

## 10. Agent Tier Classification
Classify the agent using this tier system:
Agent Tier: Level X - [Title]

Tier Scale:
Level 1: Trainee -- Fundamental gaps in sales methodology; needs intensive coaching
Level 2: Developing -- Shows awareness of techniques but inconsistent execution
Level 3: Competent -- Solid fundamentals with specific areas for growth
Level 4: Advanced -- Strong performer with minor refinements needed
Level 5: Elite -- Exceptional across all dimensions; ready to mentor others

Promotion Criteria (Next Level):
- At least 3 specific, measurable criteria

## 11. Client Emotional Journey & Risk Map
Analyze the client's emotional state throughout the conversation AND provide actionable solutions for each risk identified. For EACH phase:

EARLY PHASE (first third of conversation):
Sentiment: [Neutral/Curious/Skeptical/Engaged/etc.]
Evidence: Quote specific client language
Timestamp: [range]
Risk Identified: What could go wrong based on this sentiment
Solution: Specific action the agent should take to address this risk

MID PHASE (middle third):
Sentiment: [state]
Evidence: How client language changed
Timestamp: [range]
Risk Identified: What danger signs appeared
Solution: What the agent should do differently next time at this stage

LATE PHASE (final third):
Sentiment: [state]
Evidence: Quote specific language shifts
Timestamp: [range]
Risk Identified: End-state risk assessment
Solution: Immediate corrective action to protect the deal

EMOTIONAL TURNING POINTS:
- Identify 2-3 moments where sentiment shifted. For each: TIMESTAMP, TRIGGER, DIRECTION (positive/negative), RECOMMENDED RESPONSE

TRUST ARC: Did trust increase, decrease, or remain flat? Cite evidence and state what the agent must do to strengthen trust going forward.

OVERALL SENTIMENT TRAJECTORY: [e.g., Neutral -> Curious -> Engaged -> Price-Anxious -> Cautiously Interested]

## 12. Communication Balance & Listening Evaluation
Evaluate the agent's listening and communication balance:

TALK RATIO:
Estimated Agent Talk Time: XX%
Estimated Client Talk Time: XX%
Verdict: [Balanced / Agent-dominated / Client-led] — explain WHY this ratio helps or hurts the deal
Recommendation: Specific adjustment for next meeting

INTERRUPTION ANALYSIS:
- Count instances where agent cut short client statements
- For each: TIMESTAMP, what happened, and what the agent should have done instead

REFLECTIVE LISTENING EVALUATION:
- How often did the agent paraphrase, summarize, or validate?
- Cite specific examples with timestamps
- Score: X/10
- Improvement Tip: One specific technique to improve this score

FOLLOW-UP QUESTION QUALITY:
- Rate depth: Surface-level / Moderate / Deep
- Cite best and worst examples with timestamps
- Recommendation: Specific question types the agent should add to their toolkit

CRITICAL LISTENING FAILURES:
- Moments where the agent missed or ignored important client signals
- For each: TIMESTAMP, what the client said, what the agent missed, and what a top agent would have caught

## 13. Five Core Discovery Pillars Assessment
Evaluate how thoroughly the agent explored the 5 essential discovery areas that every top agent MUST cover before any listing presentation can succeed:

Use PIPE-DELIMITED format:
Pillar|Status|Evidence|AI Confidence
Budget/Financial Capacity|Confirmed/Partially Explored/Not Addressed|Specific quote or timestamp|High/Medium/Low
Decision Authority|Confirmed/Partially Explored/Not Addressed|Evidence|High/Medium/Low
Timeline & Urgency|Confirmed/Partially Explored/Not Addressed|Evidence|High/Medium/Low
Emotional Motivations|Explored/Partially Explored/Not Addressed|Evidence|High/Medium/Low
Competitive Alternatives|Explored/Partially Explored/Not Addressed|Evidence|High/Medium/Low

DISCOVERY COMPLETENESS SCORE: X/5 pillars adequately explored

MISSING DISCOVERY QUESTIONS:
- List 3-5 specific questions the agent MUST ask next time, with the exact wording

## 14. Emotional Intelligence Assessment
Evaluate the agent's EQ during the interaction. For EACH dimension, provide the score on its OWN line, then the evidence as separate bullet points below:

EMPATHY SCORE: X/10
- Evidence point 1 with timestamp
- Evidence point 2 with timestamp

ADAPTABILITY SCORE: X/10
- Evidence point 1 with timestamp
- Evidence point 2 with timestamp

SOCIAL AWARENESS SCORE: X/10
- Evidence point 1 with timestamp
- Evidence point 2 with timestamp

EMOTIONAL REGULATION SCORE: X/10
- Evidence point 1 with timestamp
- Evidence point 2 with timestamp

OVERALL EQ SCORE: X/10
AI Confidence: High/Medium/Low

Do NOT overlap score text with evidence. Each score must be on its own line.

## 15. Ethics & Compliance Audit
Evaluate ethical and compliance dimensions:

PRESSURE TACTICS ASSESSMENT:
- Instances of high-pressure tactics with TIMESTAMP and language used
- Risk Level: None/Low/Medium/High

OVER-PROMISING CHECK:
- Claims or guarantees that may be unsupported with TIMESTAMP
- Risk Level: None/Low/Medium/High

REGULATORY LANGUAGE REVIEW:
- Statements creating legal/regulatory exposure
- Fair housing compliance indicators

TRANSPARENCY ASSESSMENT:
- Was the agent transparent about fees, processes, limitations?

OVERALL ETHICS RISK: None/Low/Medium/High
Justification: Brief assessment

ETHICS RECOMMENDATIONS:
- Specific actionable suggestions

## 16. Self-Awareness & Adaptability Profile
Evaluate whether the agent recognizes their own patterns and adapts in real-time. For EACH dimension, provide the findings then a specific improvement recommendation:

SIGNAL RECOGNITION:
- Did the agent recognize client hesitation or disengagement?
- For each signal: TIMESTAMP, description, whether agent noticed it
- Improvement Recommendation: Specific technique to improve signal detection

REAL-TIME ADAPTATION:
- Did the agent adjust approach mid-conversation?
- Cite specific pivot moments or missed adaptation opportunities
- Improvement Recommendation: Specific drill to build adaptive response skills

SELF-CORRECTION INSTANCES:
- Moments where the agent caught and corrected their own approach
- Improvement Recommendation: How to build self-correction into muscle memory

BLIND SPOTS:
- Recurring patterns the agent appears unaware of
- Improvement Recommendation: Specific exercise to expose and address each blind spot

SELF-AWARENESS SCORE: X/10
AI Confidence: High/Medium/Low

## 17. Client Engagement & Tone Analysis
Analyze the agent's tone, energy, and client engagement quality throughout the conversation. This section focuses EXCLUSIVELY on HOW the agent communicates — not WHAT they say.

TONE CONSISTENCY:
- Was the agent's tone professional, warm, authoritative, or inconsistent?
- Identify tone shifts with TIMESTAMPS and describe the trigger
- Score: X/10

ENERGY & ENTHUSIASM:
- Did the agent maintain appropriate energy? Too flat? Too aggressive?
- Cite specific moments where energy helped or hurt the interaction
- Score: X/10

CLIENT ENGAGEMENT QUALITY:
- How effectively did the agent keep the client involved and participating?
- Identify moments of peak engagement vs disengagement with TIMESTAMPS
- Score: X/10

RAPPORT-BUILDING LANGUAGE:
- Specific phrases, mirroring, name usage, personal references that built connection
- Missed opportunities for rapport (with exact moments)
- Score: X/10

OVERALL TONE & ENGAGEMENT SCORE: X/10
AI Confidence: High/Medium/Low

## 18. Negotiation & Persuasion Proficiency
Evaluate the agent's negotiation tactics, persuasion techniques, and deal-advancing behaviors. This section analyzes the agent's ability to move the client toward commitment.

PERSUASION TECHNIQUES USED:
- Identify specific persuasion frameworks (social proof, scarcity, authority, reciprocity, anchoring, framing)
- For each technique: TIMESTAMP, what was said, effectiveness rating (Effective/Partially Effective/Ineffective)

NEGOTIATION POSITIONING:
- Did the agent establish strong positioning or give away leverage?
- Specific moments where negotiation advantage was gained or lost
- Score: X/10

OBJECTION-TO-COMMITMENT CONVERSION:
- Track each objection raised and whether the agent converted it toward commitment
- For each: OBJECTION → AGENT RESPONSE → OUTCOME (Resolved/Partially Resolved/Unresolved)

VALUE FRAMING ABILITY:
- How well did the agent frame their value proposition relative to price/competition?
- Specific examples with TIMESTAMPS
- Score: X/10

CLOSING SIGNALS & TRIAL CLOSES:
- Did the agent use trial closes or test for commitment?
- Missed closing windows with TIMESTAMPS
- Score: X/10

OVERALL NEGOTIATION SCORE: X/10
AI Confidence: High/Medium/Low

═══════════════════════════════════════════════
ABSOLUTE RULES — VIOLATING ANY RULE INVALIDATES THE ENTIRE OUTPUT
═══════════════════════════════════════════════

1. ZERO REDUNDANCY: If a fact, quote, or observation appears in one section, it MUST NOT appear in any other section. Each section must contain 100% unique content. Sections 11-18 each cover a DIFFERENT analytical dimension — do not repeat findings across them.
2. EVIDENCE-ANCHORED: Every claim must reference a specific timestamp, quote, or observable behavior. No unsupported assertions. ALL TIMESTAMPS must be in M:SS or MM:SS format (minutes:seconds), e.g., [0:30-0:45], [5:15-6:00], [24:30-25:10]. NEVER use raw seconds format.
3. CLEAR SEPARATION: Sections 1-5 = DIAGNOSIS (what happened). Section 6 = COACHING (what to do differently). Sections 7-8 = DEVELOPMENT PLAN & COMMITMENTS. Section 9-10 = DEAL INTELLIGENCE & CLASSIFICATION. Sections 11-18 = ADVANCED INTELLIGENCE (each with a unique analytical lens). Never mix these purposes.
4. NO FILLER: Remove every sentence that doesn't add NEW information. If a section would be empty, write "No significant findings for this dimension."
5. PIPE FORMAT: Scores in Section 2, metrics in Sections 9 and 13 MUST use the pipe-delimited format specified.
6. HONEST AUTHORITY: Be direct and clinical. Do not soften language to spare feelings — that costs the agent money. Use "significant gap," "below benchmark," "critical development area," "requires immediate attention." Use "exceptional," "expert-level," "textbook execution" ONLY when genuinely earned with evidence.
7. NO EMOJIS, NO HASHTAGS: Professional language only. No markdown headers beyond ## and ###.
8. MINIMUM DEPTH: Section 5 must contain AT LEAST 3 failures. Section 6 must contain AT LEAST 5 playbook entries. Section 4 must contain AT LEAST 3 strengths.
9. SECTION 3 MUST include all 5 ### subheadings listed. Each subheading MUST have its own paragraph with timestamps.
10. SECTION 10 MUST include the full Tier Scale (all 5 levels) and at least 3 Promotion Criteria bullet points.
11. AI CONFIDENCE TRANSPARENCY: Every numerical score must include an AI confidence qualifier.
12. DUAL COACHING STYLES: Section 6 MUST provide both an assertive AND a consultative approach for each entry.
13. ETHICS AWARENESS: Section 15 must objectively assess ethical dimensions.
14. SCORE FORMATTING: In Sections 14, 16, 17, and 18, scores (X/10) must appear on their OWN line, separate from evidence text. Never put a score and evidence paragraph on the same line.
15. SECTION 7 MUST cover all 7 days with a different skill focus each day. Include at least 2 role-play drills across the week.
16. SECTIONS 17-18 must provide unique analysis not covered in earlier sections — focus on HOW the agent communicates (tone) and HOW they negotiate (persuasion).
"""
        return self._call_llm(prompt)

    # ════════════════════════════════════════════════════════
    # SECTION PARSING
    # ════════════════════════════════════════════════════════

    def _parse_unified_sections(self, raw_text: str) -> Dict[str, str]:
        sections = {}
        current_key = None
        current_lines = []

        for line in raw_text.split('\n'):
            # Match numbered headers like "## 1. Title" or unnumbered like "## Action Items"
            m = re.match(r'^##\s*(?:(\d+)\.\s*)?(.*)', line)
            if m and m.group(2).strip():
                header_text = m.group(2).strip()
                # Skip lines that look like table separators
                if header_text.startswith('|') or header_text.startswith('---'):
                    if current_key:
                        current_lines.append(line)
                    continue
                if current_key:
                    sections[current_key] = '\n'.join(current_lines).strip()
                num = m.group(1)
                current_key = f"{num}. {header_text}" if num else header_text
                current_lines = []
            elif current_key:
                current_lines.append(line)

        if current_key:
            sections[current_key] = '\n'.join(current_lines).strip()
        return sections

    # ════════════════════════════════════════════════════════
    # ORCHESTRATOR
    # ════════════════════════════════════════════════════════

    def generate_unified_coaching_evaluation(self):
        transcript_data = self._load_transcript()
        if not transcript_data:
            print("Failed to load transcript.")
            return

        # Extract agent and client names for anonymization
        speakers = transcript_data.summary_info.get('unique_speakers', [])
        if len(speakers) == 1:
            agent_name = speakers[0]
        elif len(speakers) > 1:
            speaker_counts = {}
            for seg in transcript_data.transcripts:
                if seg.speaker_name:
                    speaker_counts[seg.speaker_name] = speaker_counts.get(seg.speaker_name, 0) + 1
            agent_name = max(speaker_counts, key=speaker_counts.get) if speaker_counts else speakers[0]
        else:
            agent_name = "Agent"

        client_names = set()
        for seg in transcript_data.transcripts:
            if seg.speaker_name and seg.speaker_name != agent_name and seg.speaker_name != 'Unknown':
                client_names.add(seg.speaker_name)
        # Also scan agent's dialogue for names
        name_patterns = re.findall(
            r'(?:So|Hey|Now|And|,)\s+([A-Z][a-z]{2,})(?:[,\s])', 
            ' '.join(seg.transcript for seg in transcript_data.transcripts if seg.speaker_name == agent_name)
        )
        common_words = {'The', 'This', 'That', 'When', 'What', 'Where', 'How', 'Why', 'Now',
                        'Here', 'There', 'Well', 'Yes', 'Yeah', 'Cool', 'Right', 'Just',
                        'Let', 'Got', 'But', 'And', 'Because', 'Matter', 'Number', 'Level',
                        'Royal', 'Okay', 'Does', 'Both', 'First', 'Second', 'Third',
                        'Some', 'Like', 'Also', 'Most', 'Very', 'Sure', 'Homes', 'Home'}
        for name in name_patterns:
            if name not in common_words and name != agent_name:
                client_names.add(name)

        speaker_map = {agent_name: 'Agent'}
        for client in client_names:
            speaker_map[client] = 'Client'

        self._agent_name = agent_name  # Store for renderers

        transcript_text = self._format_transcript_for_analysis(transcript_data)

        # RAG retrieval — build a rich query from agenda + transcript preview
        print("Retrieving relevant knowledge from knowledge base...")
        rag_query = f"Sales coaching evaluation for: Real Estate Listing Presentation. {transcript_text[:500]}"
        relevant_chunks = self._retrieve_relevant_knowledge(rag_query, top_k=8)

        if relevant_chunks:
            print(f"Retrieved {len(relevant_chunks)} relevant knowledge chunks.")
            knowledge = "\n\n".join([
                f"[Knowledge Reference {i+1}]\n{chunk.text}"
                for i, chunk in enumerate(relevant_chunks)
            ])
        else:
            print("No knowledge base chunks retrieved — using general expertise.")
            knowledge = ""

        print("Calling LLM for brutally honest evaluation...")
        raw = self._generate_unified_llm_analysis(transcript_text, knowledge)
        if not raw:
            print("LLM returned no response.")
            return

        sections = self._parse_unified_sections(raw)
        if not sections:
            print("Failed to parse LLM response into sections.")
            return

        # Anonymize sections except Executive Brief
        for key, content in sections.items():
            if not key.startswith('1.'):
                for old_name, new_name in speaker_map.items():
                    content = content.replace(old_name, new_name)
                sections[key] = content

        print(f"Parsed {len(sections)} sections. Building 3 PDFs...")
        self._transcript_data = transcript_data  # Store for visualization section access

        # Build section number map for content lookup
        self._sec_num_map = {}
        for key in sections:
            m = re.match(r'^(\d+)\.?\s*', key)
            if m:
                self._sec_num_map[int(m.group(1))] = key
            else:
                self._sec_num_map[key] = key

        # PDF 1: Coaching Summary
        print("  Building PDF 1: coaching_summary.pdf...")
        self._create_coaching_summary_pdf(sections, transcript_data)

        # PDF 2: Agent Profile
        print("  Building PDF 2: agent_summary.pdf...")
        self._create_agent_profile_pdf(sections, transcript_data)

        # PDF 3: Visualizations
        print("  Building PDF 3: visualizations.pdf...")
        self._create_visualizations_pdf(sections, transcript_data)

        # Save visualization data as JSON for frontend
        json_path = os.path.join(self.summaries_folder, "visualizations_backup.json")
        self.viz_logger.set_metadata(
            agent=getattr(self, '_agent_name', 'Agent'),
            speakers=transcript_data.summary_info.get('unique_speakers', []),
            total_segments=transcript_data.summary_info.get('total_segments', 0),
            total_duration_seconds=transcript_data.summary_info.get('total_duration_seconds', 0),
            generated_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        )
        self.viz_logger.save_to_json(json_path)

        print("\n  All 3 PDFs + visualization JSON generated successfully!")

    # ════════════════════════════════════════════════════════
    # HELPER: Find section content by number or keyword
    # ════════════════════════════════════════════════════════

    def _find_section_content(self, sections: Dict[str, str], sec_num: int = 0, title_words: List[str] = None) -> str:
        """Find section content by number or keyword match."""
        sec_num_map = getattr(self, '_sec_num_map', {})
        if sec_num > 0 and sec_num in sec_num_map:
            return sections.get(sec_num_map[sec_num], '')
        if title_words:
            words = [w.lower() for w in title_words if len(w) > 3]
            for key, val in sections.items():
                if any(word in key.lower() for word in words):
                    return val
        return ''

    # ════════════════════════════════════════════════════════
    # HELPER: Add numbered section to story
    # ════════════════════════════════════════════════════════

    def _add_section_to_story(self, story, counter: int, title: str, renderer, content: str):
        """Add a numbered section header + rendered content to a story."""
        if not content or not content.strip():
            return counter
        counter += 1
        story.append(CondPageBreak(2.0 * inch))
        story.append(SectionDivider(width=460, color="#0D47A1", thickness=1.5))
        story.append(Spacer(1, 2))
        story.append(Paragraph(f"{counter}. {title}", self.ps['sec']))
        story.append(Spacer(1, 4))
        renderer(story, content)
        story.append(Spacer(1, 4))
        return counter

    # ════════════════════════════════════════════════════════
    # PDF 1: COACHING SUMMARY
    # ════════════════════════════════════════════════════════

    def _create_coaching_summary_pdf(self, sections: Dict[str, str], transcript_data: TranscriptData):
        pdf_path = os.path.join(self.summaries_folder, "coaching_summary.pdf")
        doc = SimpleDocTemplate(
            pdf_path, pagesize=A4,
            leftMargin=0.6 * inch, rightMargin=0.6 * inch,
            topMargin=0.7 * inch, bottomMargin=0.5 * inch,
            title="Coaching Summary Report",
            author="RAG Coaching Evaluation System",
            subject="Sales Performance Coaching Summary",
        )
        story = []
        self._build_cover_page(story, transcript_data, pdf_title="COACHING SUMMARY REPORT",
                               pdf_subtitle="Actionable Performance Coaching & Improvement Plan")
        story.append(PageBreak())

        counter = 0

        # 1. Executive Brief (LLM §1)
        content = self._find_section_content(sections, 1)
        counter = self._add_section_to_story(story, counter, 'EXECUTIVE BRIEF', self._render_executive_brief, content)

        # 2. Performance Score Dashboard (LLM §2)
        content = self._find_section_content(sections, 2)
        counter = self._add_section_to_story(story, counter, 'PERFORMANCE SCORE DASHBOARD', self._render_score_dashboard_text, content)

        # 3. Strengths & Areas for Improvement (LLM §4 + §5 combined)
        content_strengths = self._find_section_content(sections, 4)
        content_failures = self._find_section_content(sections, 5)
        if content_strengths or content_failures:
            counter += 1
            story.append(CondPageBreak(2.0 * inch))
            story.append(SectionDivider(width=460, color="#0D47A1", thickness=1.5))
            story.append(Spacer(1, 2))
            story.append(Paragraph(f"{counter}. STRENGTHS & AREAS FOR IMPROVEMENT", self.ps['sec']))
            story.append(Spacer(1, 4))
            if content_strengths:
                story.append(Paragraph("<b>A. Evidence-Anchored Strengths</b>", self.ps['subsec']))
                story.append(Spacer(1, 4))
                self._render_strengths(story, content_strengths)
                story.append(Spacer(1, 8))
            if content_failures:
                story.append(Paragraph("<b>B. Missed Opportunities & Critical Failures</b>", self.ps['subsec']))
                story.append(Spacer(1, 4))
                self._render_objections(story, content_failures)
            story.append(Spacer(1, 4))

        # 4. Client Insights (LLM §11 Client Emotional Journey + §12 Communication Balance as subs)
        content_sentiment = self._find_section_content(sections, 11)
        content_listening = self._find_section_content(sections, 12)
        if content_sentiment or content_listening:
            counter += 1
            story.append(CondPageBreak(2.0 * inch))
            story.append(SectionDivider(width=460, color="#0D47A1", thickness=1.5))
            story.append(Spacer(1, 2))
            story.append(Paragraph(f"{counter}. CLIENT INSIGHTS & COMMUNICATION INTELLIGENCE", self.ps['sec']))
            story.append(Spacer(1, 4))
            if content_sentiment:
                story.append(Paragraph("<b>A. Client Emotional Journey & Risk Map</b>", self.ps['subsec']))
                story.append(Spacer(1, 4))
                self._render_client_sentiment(story, content_sentiment)
                story.append(Spacer(1, 8))
            if content_listening:
                story.append(Paragraph("<b>B. Communication Balance & Listening Evaluation</b>", self.ps['subsec']))
                story.append(Spacer(1, 4))
                self._render_listening_intelligence(story, content_listening)
            story.append(Spacer(1, 4))

        # 5. Coaching Recommendations & Tactical Playbook (LLM §6)
        content = self._find_section_content(sections, 6)
        counter = self._add_section_to_story(story, counter, 'COACHING RECOMMENDATIONS & TACTICAL PLAYBOOK', self._render_tactical_playbook_combined, content)

        # 6. 7-Day Intensive Coaching Plan (LLM §7 — merged day-by-day plan with drills)
        content = self._find_section_content(sections, 7)
        counter = self._add_section_to_story(story, counter, '7-DAY INTENSIVE COACHING PLAN', self._render_7day_coaching_plan, content)

        # 7. Deal Intelligence Summary (LLM §9 as main, with Action Items + Decisions §8 as subs)
        content_deal = self._find_section_content(sections, 9)
        content_actions = self._find_section_content(sections, 0, ['Action', 'Items'])
        content_decisions = self._find_section_content(sections, 8)
        if content_deal or content_actions or content_decisions:
            counter += 1
            story.append(CondPageBreak(2.0 * inch))
            story.append(SectionDivider(width=460, color="#0D47A1", thickness=1.5))
            story.append(Spacer(1, 2))
            story.append(Paragraph(f"{counter}. DEAL INTELLIGENCE & ACTION PLAN", self.ps['sec']))
            story.append(Spacer(1, 4))
            if content_deal:
                self._render_deal_intelligence_text(story, content_deal)
                story.append(Spacer(1, 8))
            if content_actions:
                story.append(Paragraph("<b>A. Action Items</b>", self.ps['subsec']))
                story.append(Spacer(1, 4))
                self._render_action_items(story, content_actions)
                story.append(Spacer(1, 8))
            if content_decisions:
                story.append(Paragraph("<b>B. Decisions & Commitments</b>", self.ps['subsec']))
                story.append(Spacer(1, 4))
                self._render_decisions(story, content_decisions)
            story.append(Spacer(1, 4))

        doc.build(story,
                  onFirstPage=self._draw_cover_header_footer,
                  onLaterPages=self._draw_content_header_footer)
        print(f"    Saved: {pdf_path}")

    # ════════════════════════════════════════════════════════
    # PDF 2: AGENT PROFILE
    # ════════════════════════════════════════════════════════

    def _create_agent_profile_pdf(self, sections: Dict[str, str], transcript_data: TranscriptData):
        pdf_path = os.path.join(self.summaries_folder, "agent_summary.pdf")
        doc = SimpleDocTemplate(
            pdf_path, pagesize=A4,
            leftMargin=0.6 * inch, rightMargin=0.6 * inch,
            topMargin=0.7 * inch, bottomMargin=0.5 * inch,
            title="Agent Profile & Intelligence Report",
            author="RAG Coaching Evaluation System",
            subject="Agent Intelligence Profile & Advanced Assessment",
        )
        story = []
        self._build_cover_page(story, transcript_data, pdf_title="AGENT PROFILE & INTELLIGENCE REPORT",
                               pdf_subtitle="Advanced Behavioral Intelligence & Compliance Assessment")
        story.append(PageBreak())

        counter = 0

        # 1. Five Core Discovery Pillars Assessment (LLM §13)
        content = self._find_section_content(sections, 13)
        counter = self._add_section_to_story(story, counter, 'FIVE CORE DISCOVERY PILLARS ASSESSMENT', self._render_discovery_matrix, content)

        # 2. Emotional Intelligence Assessment (LLM §14)
        content = self._find_section_content(sections, 14)
        counter = self._add_section_to_story(story, counter, 'EMOTIONAL INTELLIGENCE ASSESSMENT', self._render_emotional_intelligence, content)

        # 3. Ethics & Compliance Audit (LLM §15)
        content = self._find_section_content(sections, 15)
        counter = self._add_section_to_story(story, counter, 'ETHICS & COMPLIANCE AUDIT', self._render_ethics_compliance, content)

        # 4. Self-Awareness & Adaptability Profile (LLM §16)
        content = self._find_section_content(sections, 16)
        counter = self._add_section_to_story(story, counter, 'SELF-AWARENESS & ADAPTABILITY PROFILE', self._render_agent_self_awareness, content)

        # 5. Client Engagement & Tone Analysis (LLM §17 — NEW)
        content = self._find_section_content(sections, 17)
        counter = self._add_section_to_story(story, counter, 'CLIENT ENGAGEMENT & TONE ANALYSIS', self._render_tone_analysis, content)

        # 6. Negotiation & Persuasion Proficiency (LLM §18 — NEW)
        content = self._find_section_content(sections, 18)
        counter = self._add_section_to_story(story, counter, 'NEGOTIATION & PERSUASION PROFICIENCY', self._render_negotiation_proficiency, content)

        # 7. Behavioral Breakdown (LLM §3 — Communication patterns, body language, questioning)
        content = self._find_section_content(sections, 3)
        counter = self._add_section_to_story(story, counter, 'BEHAVIORAL BREAKDOWN & COMMUNICATION PATTERNS', self._render_behavioral_breakdown, content)

        # 8. Agent Tier Classification (LLM §10)
        content = self._find_section_content(sections, 10)
        counter = self._add_section_to_story(story, counter, 'AGENT TIER CLASSIFICATION', self._render_agent_tier, content)

        doc.build(story,
                  onFirstPage=self._draw_cover_header_footer,
                  onLaterPages=self._draw_content_header_footer)
        print(f"    Saved: {pdf_path}")

    # ════════════════════════════════════════════════════════
    # PDF 3: VISUALIZATIONS
    # ════════════════════════════════════════════════════════

    def _create_visualizations_pdf(self, sections: Dict[str, str], transcript_data: TranscriptData):
        pdf_path = os.path.join(self.summaries_folder, "visualizations.pdf")
        doc = SimpleDocTemplate(
            pdf_path, pagesize=A4,
            leftMargin=0.6 * inch, rightMargin=0.6 * inch,
            topMargin=0.7 * inch, bottomMargin=0.5 * inch,
            title="Visual Performance Analytics",
            author="RAG Coaching Evaluation System",
            subject="Performance Data Visualizations & Charts",
        )
        story = []
        self._build_cover_page(story, transcript_data, pdf_title="VISUAL PERFORMANCE ANALYTICS",
                               pdf_subtitle="Data-Driven Charts, Graphs & Performance Intelligence")
        story.append(PageBreak())

        # Build visualization section (all charts)
        self._build_visualization_section(story, sections)

        doc.build(story,
                  onFirstPage=self._draw_cover_header_footer,
                  onLaterPages=self._draw_content_header_footer)
        print(f"    Saved: {pdf_path}")

    # ── Header / Footer (matching Final.py enterprise style) ──

    def _draw_cover_header_footer(self, canv, doc):
        canv.saveState()
        w, h = A4
        # Top accent bar
        canv.setStrokeColor(self.C['primary_dark'])
        canv.setLineWidth(4)
        canv.line(0, h - 2, w, h - 2)
        # Footer bar
        canv.setFillColor(self.C['primary_dark'])
        canv.rect(0, 0, w, 0.32 * inch, fill=1, stroke=0)
        canv.setFillColor(colors.white)
        canv.setFont("Helvetica", 7.5)
        canv.drawCentredString(w / 2, 0.12 * inch, f"Page {doc.page}")
        canv.drawString(0.6 * inch, 0.12 * inch, f"(c) {datetime.now().year}")
        canv.drawRightString(w - 0.6 * inch, 0.12 * inch, "CONFIDENTIAL")
        canv.restoreState()

    def _draw_content_header_footer(self, canv, doc):
        canv.saveState()
        w, h = A4
        hdr_h = 0.55 * inch
        # Header bar
        canv.setFillColor(self.C['primary_dark'])
        canv.rect(0, h - hdr_h, w, hdr_h, fill=1, stroke=0)
        canv.setFillColor(colors.white)
        canv.setFont("Helvetica-Bold", 10)
        canv.drawString(0.6 * inch, h - 0.35 * inch, "REAL ESTATE AI COACHING")
        canv.setFont("Helvetica", 7)
        canv.drawString(0.6 * inch, h - 0.48 * inch, "Enterprise Performance Intelligence")
        canv.setFont("Helvetica-Bold", 9)
        canv.drawRightString(w - 0.6 * inch, h - 0.35 * inch, "COACHING REPORT")
        canv.setFont("Helvetica", 7)
        canv.drawRightString(w - 0.6 * inch, h - 0.48 * inch, "RAG-Powered Intelligence")
        # Footer bar
        ftr_h = 0.35 * inch
        canv.setFillColor(self.C['primary'])
        canv.rect(0, 0, w, ftr_h, fill=1, stroke=0)
        canv.setFillColor(colors.white)
        canv.setFont("Helvetica", 7.5)
        canv.drawCentredString(w / 2, 0.12 * inch, f"Page {doc.page}")
        canv.drawString(0.6 * inch, 0.12 * inch, f"(c) {datetime.now().year} | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        canv.drawRightString(w - 0.6 * inch, 0.12 * inch, "CONFIDENTIAL")
        canv.restoreState()

    # ── Cover Page ──

    def _build_cover_page(self, story, transcript_data: TranscriptData, pdf_title: str = "COACHING EVALUATION REPORT", pdf_subtitle: str = "Enterprise AI-Powered Performance Intelligence"):
        story.append(Spacer(1, 1.0 * inch))
        story.append(Paragraph(pdf_title, self.ps['cover_title']))
        story.append(Spacer(1, 0.05 * inch))
        story.append(Paragraph(pdf_subtitle, self.ps['cover_sub']))
        story.append(Spacer(1, 0.4 * inch))

        info = transcript_data.summary_info
        speakers = info.get("unique_speakers", [])
        dur_sec = info.get("total_duration_seconds", 0)
        dur_min = int(dur_sec // 60)
        dur_s = int(dur_sec % 60)
        duration_str = f"{dur_min} minutes {dur_s} seconds" if dur_min > 0 else f"{dur_s} seconds"
        # Agent name: find the speaker who is NOT a client (heuristic: speaker with most segments)
        if len(speakers) == 1:
            agent_name = speakers[0]
        elif len(speakers) > 1:
            # The agent typically speaks the most segments in a listing presentation
            speaker_counts = {}
            for seg in transcript_data.transcripts:
                if seg.speaker_name:
                    speaker_counts[seg.speaker_name] = speaker_counts.get(seg.speaker_name, 0) + 1
            # Pick speaker with most segments (likely the agent)
            agent_name = max(speaker_counts, key=speaker_counts.get) if speaker_counts else speakers[0]
        else:
            agent_name = "N/A"

        # Extract client names dynamically from transcript — detect proper nouns addressed by agent
        client_names = set()
        # Collect all unique speaker names that are not the agent
        for seg in transcript_data.transcripts:
            if seg.speaker_name and seg.speaker_name != agent_name and seg.speaker_name != 'Unknown':
                client_names.add(seg.speaker_name)
        # Also scan agent's dialogue for names (capitalized words after addressing patterns)
        name_patterns = re.findall(
            r'(?:So|Hey|Now|And|,)\s+([A-Z][a-z]{2,})(?:[,\s])', 
            ' '.join(seg.transcript for seg in transcript_data.transcripts if seg.speaker_name == agent_name)
        )
        # Filter out common non-name words
        common_words = {'The', 'This', 'That', 'When', 'What', 'Where', 'How', 'Why', 'Now',
                        'Here', 'There', 'Well', 'Yes', 'Yeah', 'Cool', 'Right', 'Just',
                        'Let', 'Got', 'But', 'And', 'Because', 'Matter', 'Number', 'Level',
                        'Royal', 'Okay', 'Does', 'Both', 'First', 'Second', 'Third',
                        'Some', 'Like', 'Also', 'Most', 'Very', 'Sure', 'Homes', 'Home'}
        for name in name_patterns:
            if name not in common_words and name != agent_name:
                client_names.add(name)
        client_display = ", ".join(sorted(client_names)) if client_names else "Client(s)"

        meeting_id = f"COACH_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        rows = [
            ['Report Details', ''],
            ['Meeting ID', meeting_id],
            ['Date', datetime.now().strftime('%B %d, %Y')],
            ['Agenda', 'Real Estate Listing Presentation & Coaching Evaluation'],
            ['Duration', duration_str],
            ['Agent Evaluated', agent_name],
            ['Report Type', 'Enterprise Coaching Evaluation'],
            ['RAG System', 'Active' if self.faiss_index else 'Inactive'],
        ]
        t = Table(rows, colWidths=[1.8 * inch, 4.6 * inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), self.C['primary_dark']),
            ('TEXTCOLOR', (0, 0), (1, 0), self.C['white']),
            ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (1, 0), 11),
            ('ALIGN', (0, 0), (1, 0), 'CENTER'),
            ('PADDING', (0, 0), (1, 0), 10),
            ('BACKGROUND', (0, 1), (0, -1), colors.HexColor("#E3F2FD")),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 1), (0, -1), 9),
            ('ALIGN', (0, 1), (0, -1), 'RIGHT'),
            ('PADDING', (0, 1), (-1, -1), 7),
            ('FONTNAME', (1, 1), (1, -1), 'Helvetica'),
            ('FONTSIZE', (1, 1), (1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#BBDEFB")),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.4 * inch))

        # Confidential notice
        story.append(Table(
            [[Paragraph("CONFIDENTIAL  --  FOR COACHING PURPOSES ONLY", self.ps['conf'])]],
            colWidths=[6.2 * inch],
            style=TableStyle([
                ('BOX', (0, 0), (0, 0), 1.5, self.C['danger']),
                ('PADDING', (0, 0), (0, 0), 10),
            ])
        ))

    # ════════════════════════════════════════════════════════
    # SECTION BUILDER — Routes each section to its renderer
    # ════════════════════════════════════════════════════════

    def _build_all_sections(self, story, sections: Dict[str, str]):
        """Legacy compatibility — routes all sections to renderers in original order.
        Not used in 3-PDF mode. Kept for backward compatibility."""
        # Parse section numbers from keys
        sec_num_map = {}
        for key in sections:
            m = re.match(r'^(\d+)\.?\s*', key)
            if m:
                sec_num_map[int(m.group(1))] = key
            else:
                sec_num_map[key] = key

        ordered_sections = [
            (1, 'EXECUTIVE BRIEF', self._render_executive_brief),
            (2, 'PERFORMANCE SCORE DASHBOARD', self._render_score_dashboard_text),
            (3, 'BEHAVIORAL BREAKDOWN', self._render_behavioral_breakdown),
            (4, 'STRENGTHS (EVIDENCE-ANCHORED)', self._render_strengths),
            (5, 'MISSED OPPORTUNITIES & CRITICAL FAILURES', self._render_objections),
            (6, 'TACTICAL CORRECTIONS & COACHING PLAYBOOK', self._render_tactical_playbook_combined),
            (7, '7-DAY INTENSIVE COACHING PLAN', self._render_7day_coaching_plan),
            (8, 'DECISIONS & COMMITMENTS', self._render_decisions),
            (0, 'ACTION ITEMS', self._render_action_items),
            (9, 'DEAL INTELLIGENCE SUMMARY', self._render_deal_intelligence_text),
            (10, 'AGENT TIER CLASSIFICATION', self._render_agent_tier),
            (11, 'CLIENT EMOTIONAL JOURNEY & RISK MAP', self._render_client_sentiment),
            (12, 'COMMUNICATION BALANCE & LISTENING EVALUATION', self._render_listening_intelligence),
            (13, 'FIVE CORE DISCOVERY PILLARS', self._render_discovery_matrix),
            (14, 'EMOTIONAL INTELLIGENCE ASSESSMENT', self._render_emotional_intelligence),
            (15, 'ETHICS & COMPLIANCE AUDIT', self._render_ethics_compliance),
            (16, 'SELF-AWARENESS & ADAPTABILITY PROFILE', self._render_agent_self_awareness),
            (17, 'CLIENT ENGAGEMENT & TONE ANALYSIS', self._render_tone_analysis),
            (18, 'NEGOTIATION & PERSUASION PROFICIENCY', self._render_negotiation_proficiency),
        ]

        section_counter = 0
        for sec_num, title, renderer in ordered_sections:
            # Find content by section number first, then by keyword match
            content = None
            if sec_num > 0 and sec_num in sec_num_map:
                content = sections.get(sec_num_map[sec_num], '')
            else:
                title_words = [w.lower() for w in title.split() if len(w) > 3]
                for key, val in sections.items():
                    if any(word in key.lower() for word in title_words):
                        content = val
                        break

            if not content or not content.strip():
                continue

            section_counter += 1

            # Tight spacing — no excessive blanks
            story.append(CondPageBreak(2.0 * inch))

            # Section header — clean numbered format: "1. EXECUTIVE BRIEF"
            story.append(SectionDivider(width=460, color="#0D47A1", thickness=1.5))
            story.append(Spacer(1, 2))
            story.append(Paragraph(f"{section_counter}. {title}", self.ps['sec']))
            story.append(Spacer(1, 4))
            renderer(story, content)
            story.append(Spacer(1, 4))

    # ════════════════════════════════════════════════════════
    # SECTION RENDERERS (A through M)
    # ════════════════════════════════════════════════════════

    # ── A. Executive Brief ──
    def _render_executive_brief(self, story, content: str):
        """Strategic diagnostic in a subtle shaded container."""
        inner = []
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
            line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
            if line.startswith('-') or line.startswith('*'):
                text = line.lstrip('-* ')
                inner.append(Paragraph(f"<bullet>&bull;</bullet>{text}", self.ps['bullet']))
            else:
                inner.append(Paragraph(line, self.ps['body']))
            inner.append(Spacer(1, 4))

        if inner:
            t = Table([[inner]], colWidths=[6.2 * inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F5F7FA")),
                ('BOX', (0, 0), (-1, -1), 0.5, self.C['border']),
                ('PADDING', (0, 0), (-1, -1), 12),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))
            story.append(t)
            story.append(Spacer(1, 8))

    # ── B. Score Dashboard (TEXT ONLY — chart goes to visualization section) ──
    def _render_score_dashboard_text(self, story, content: str):
        """Parse pipe-delimited scores and render as table. Charts deferred to visualization section."""
        scores = {}
        overall_score = None
        overall_just = ""
        rows = []

        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 4:
                label = parts[0].replace('**', '').strip()
                try:
                    score_val = int(parts[1])
                except ValueError:
                    continue
                max_val = int(parts[2]) if parts[2].isdigit() else 10
                justification = parts[3] if len(parts) > 3 else ""
                confidence = parts[4].strip() if len(parts) > 4 else ""

                if label.upper() == 'OVERALL':
                    overall_score = score_val
                    overall_just = justification
                else:
                    scores[label] = score_val
                    rows.append((label, score_val, max_val, justification, confidence))

        # Hero overall score display
        if overall_score is not None:
            pct = overall_score / 10
            border_color = "#43A047" if pct >= 0.75 else "#FF8F00" if pct >= 0.50 else "#E53935"
            score_style = ParagraphStyle('score_hero', parent=self.ps['score_big'],
                                         fontSize=36, leading=42, spaceAfter=0, spaceBefore=0)
            label_style = ParagraphStyle('score_lbl', parent=self.ps['score_label'],
                                         fontSize=9, leading=12, spaceBefore=4, spaceAfter=0)
            just_style = ParagraphStyle('score_just', parent=self.ps['body'],
                                        fontSize=8, leading=11, textColor=self.C['medium'],
                                        alignment=TA_CENTER, spaceBefore=2)
            hero = Table([
                [Paragraph(f"<font color='{border_color}'><b>{overall_score}/10</b></font>", score_style)],
                [Spacer(1, 4)],
                [Paragraph("OVERALL PERFORMANCE SCORE", label_style)],
                [Paragraph(f"<i>{overall_just}</i>", just_style)],
            ], colWidths=[6.2 * inch], rowHeights=[48, 4, 16, None])
            hero.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F5F7FA")),
                ('LINEBELOW', (0, -1), (-1, -1), 1.5, colors.HexColor(border_color)),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('LEFTPADDING', (0, 0), (-1, -1), 10),
                ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            story.append(hero)
            story.append(Spacer(1, 10))

        # Individual scores as table rows
        if rows:
            td_wrap = ParagraphStyle('td_wrap', parent=self.ps['td'], wordWrap='LTR', fontSize=8, leading=11)
            conf_style = ParagraphStyle('td_conf', parent=self.ps['td'], wordWrap='LTR', fontSize=7, leading=10,
                                        textColor=self.C['light'])
            table_data = [[
                Paragraph("<b>Dimension</b>", self.ps['th']),
                Paragraph("<b>Score</b>", self.ps['th']),
                Paragraph("<b>Justification</b>", self.ps['th']),
                Paragraph("<b>AI Confidence</b>", self.ps['th']),
            ]]
            for row_data in rows:
                label, score_val, max_val, just = row_data[0], row_data[1], row_data[2], row_data[3]
                confidence = row_data[4] if len(row_data) > 4 else ""
                pct = score_val / max_val
                sc = "#43A047" if pct >= 0.75 else "#FF8F00" if pct >= 0.50 else "#E53935"
                # Color-code confidence level
                conf_color = "#2E7D32" if 'high' in confidence.lower() else "#F57F17" if 'medium' in confidence.lower() else "#C62828" if 'low' in confidence.lower() else "#757575"
                table_data.append([
                    Paragraph(label, self.ps['td']),
                    Paragraph(f"<b><font color='{sc}'>{score_val}/{max_val}</font></b>", self.ps['td_c']),
                    Paragraph(f"<i>{just}</i>", td_wrap),
                    Paragraph(f"<font color='{conf_color}'><i>{confidence}</i></font>", conf_style),
                ])
            t = Table(table_data, colWidths=[1.4 * inch, 0.5 * inch, 3.3 * inch, 1.0 * inch])
            style_cmds = [
                ('BACKGROUND', (0, 0), (-1, 0), self.C['primary_dark']),
                ('TEXTCOLOR', (0, 0), (-1, 0), self.C['white']),
                ('GRID', (0, 0), (-1, -1), 0.5, self.C['border']),
                ('PADDING', (0, 0), (-1, -1), 5),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]
            for i in range(1, len(table_data)):
                if i % 2 == 0:
                    style_cmds.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor("#F5F7FA")))
            t.setStyle(TableStyle(style_cmds))
            story.append(t)

        # Store scores for visualization section
        self._cached_scores = scores

    # ── C. Behavioral Breakdown ──
    def _render_behavioral_breakdown(self, story, content: str):
        content = self._convert_timestamps_to_mmss(content)
        subsections = re.split(r'###\s*(.*)', content)
        # subsections: ['', 'Title1', 'Content1', 'Title2', 'Content2', ...]
        i = 1
        while i < len(subsections) - 1:
            sub_title = subsections[i].strip()
            sub_content = subsections[i + 1].strip()
            story.append(Paragraph(f"<b>{sub_title}</b>", self.ps['subsec']))
            for line in sub_content.split('\n'):
                line = line.strip()
                if not line:
                    continue
                # Highlight timestamps
                line = re.sub(r'\[(\d+[\.\d]*s?\s*-\s*\d+[\.\d]*s?)\]',
                              r"<font color='#1565C0'><b>[\1]</b></font>", line)
                line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
                if line.startswith('-') or line.startswith('*'):
                    text = line.lstrip('-* ')
                    story.append(Paragraph(f"<bullet>&bull;</bullet>{text}", self.ps['bullet']))
                else:
                    story.append(Paragraph(line, self.ps['body']))
                story.append(Spacer(1, 2))
            i += 2

        # Fallback: if no ### subheadings found, look for **bold headers** or numbered patterns
        if len(subsections) <= 1:
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Detect bold header lines (e.g. "**Communication Patterns**" alone on a line)
                bold_match = re.match(r'^\*\*(.*?)\*\*\s*$', line)
                if bold_match:
                    story.append(Spacer(1, 4))
                    story.append(Paragraph(f"<b>{bold_match.group(1)}</b>", self.ps['subsec']))
                    continue
                line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
                line = re.sub(r'\[(\d+[\.\d]*s?\s*-\s*\d+[\.\d]*s?)\]',
                              r"<font color='#1565C0'><b>[\1]</b></font>", line)
                if line.startswith('-') or line.startswith('*'):
                    text = line.lstrip('-* ')
                    story.append(Paragraph(f"<bullet>&bull;</bullet>{text}", self.ps['bullet']))
                else:
                    story.append(Paragraph(line, self.ps['body']))
                story.append(Spacer(1, 2))

    # ── D. Strengths ──
    def _render_strengths(self, story, content: str):
        """Parse structured STRENGTH/TIMESTAMP/EVIDENCE/IMPACT entries into a single professional table."""
        content = self._convert_timestamps_to_mmss(content)
        td_s = ParagraphStyle('td_strength', parent=self.ps['td'], fontSize=8, leading=11,
                              textColor=self.C['dark'], wordWrap='LTR')
        td_ev = ParagraphStyle('td_evidence', parent=self.ps['td'], fontSize=7.5, leading=10,
                               textColor=self.C['medium'], fontName='Helvetica-Oblique', wordWrap='LTR')
        td_imp = ParagraphStyle('td_impact', parent=self.ps['td'], fontSize=7.5, leading=10,
                                textColor=colors.HexColor('#2E7D32'), wordWrap='LTR')

        # Parse structured entries
        entries = []
        current_entry = {}
        for line in content.split('\n'):
            stripped = line.strip()
            if not stripped:
                if current_entry:
                    entries.append(current_entry)
                    current_entry = {}
                continue
            matched_field = False
            for label in ['STRENGTH', 'TIMESTAMP', 'EVIDENCE', 'IMPACT', 'WHY IT MATTERS']:
                if stripped.upper().startswith(label + ':'):
                    value = stripped[len(label) + 1:].strip().strip('"')
                    if label == 'STRENGTH' and current_entry:
                        entries.append(current_entry)
                        current_entry = {}
                    current_entry[label.upper()] = value
                    matched_field = True
                    break
            if not matched_field:
                # Treat as bullet/extra content
                cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', stripped)
                if cleaned.startswith('-') or cleaned.startswith('*'):
                    cleaned = cleaned.lstrip('-* ')
                if not current_entry:
                    current_entry['STRENGTH'] = cleaned
                elif 'EVIDENCE' not in current_entry:
                    current_entry.setdefault('STRENGTH', '')
                    current_entry['STRENGTH'] += ' ' + cleaned
        if current_entry:
            entries.append(current_entry)

        # Build proper table with columns: #, Strength, Timestamp, Evidence, Impact
        if entries and any('STRENGTH' in e for e in entries):
            rows = [[
                Paragraph("<b>#</b>", self.ps['th']),
                Paragraph("<b>Strength</b>", self.ps['th']),
                Paragraph("<b>When</b>", self.ps['th']),
                Paragraph("<b>Evidence</b>", self.ps['th']),
                Paragraph("<b>Impact</b>", self.ps['th']),
            ]]
            entry_num = 0
            for entry in entries:
                if not entry.get('STRENGTH'):
                    continue
                entry_num += 1
                name = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', entry.get('STRENGTH', ''))
                ts = entry.get('TIMESTAMP', '--')
                evidence = entry.get('EVIDENCE', '--').strip('"')
                impact = entry.get('IMPACT', entry.get('WHY IT MATTERS', '--'))
                rows.append([
                    Paragraph(f"<b>{entry_num}</b>", self.ps['td_c']),
                    Paragraph(f"<b>{name}</b>", td_s),
                    Paragraph(f"<font color='#1565C0'>{ts}</font>", self.ps['td_c']),
                    Paragraph(f'<i>"{evidence}"</i>' if evidence != '--' else '--', td_ev),
                    Paragraph(impact, td_imp),
                ])
            t = Table(rows, colWidths=[0.35 * inch, 1.55 * inch, 0.85 * inch, 2.05 * inch, 1.4 * inch])
            style_cmds = [
                ('BACKGROUND', (0, 0), (-1, 0), self.C['success']),
                ('TEXTCOLOR', (0, 0), (-1, 0), self.C['white']),
                ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#C8E6C9')),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('LEFTPADDING', (0, 0), (-1, -1), 4),
                ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('ALIGN', (0, 0), (0, -1), 'CENTER'),
                ('ALIGN', (2, 0), (2, -1), 'CENTER'),
            ]
            for i in range(1, len(rows)):
                if i % 2 == 0:
                    style_cmds.append(('BACKGROUND', (0, i), (-1, i), self.C['success_bg']))
            t.setStyle(TableStyle(style_cmds))
            story.append(t)
        else:
            # Fallback: parse as flat bullets into a simple 2-column table
            bullet_items = []
            for line in content.split('\n'):
                stripped = line.strip()
                if not stripped:
                    continue
                stripped = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', stripped)
                if stripped.startswith('-') or stripped.startswith('*'):
                    bullet_items.append(stripped.lstrip('-* '))
                elif not any(stripped.upper().startswith(l + ':') for l in ['STRENGTH','TIMESTAMP','EVIDENCE','IMPACT']):
                    bullet_items.append(stripped)
            if bullet_items:
                rows = [[Paragraph("<b>#</b>", self.ps['th']),
                         Paragraph("<b>Identified Strength</b>", self.ps['th'])]]
                for i, item in enumerate(bullet_items, 1):
                    rows.append([
                        Paragraph(f"<b>{i}</b>", self.ps['td_c']),
                        Paragraph(item, td_s),
                    ])
                t = Table(rows, colWidths=[0.4 * inch, 5.8 * inch])
                style_cmds = [
                    ('BACKGROUND', (0, 0), (-1, 0), self.C['success']),
                    ('TEXTCOLOR', (0, 0), (-1, 0), self.C['white']),
                    ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#C8E6C9')),
                    ('TOPPADDING', (0, 0), (-1, -1), 5),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                    ('LEFTPADDING', (0, 0), (-1, -1), 6),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('ALIGN', (0, 1), (0, -1), 'CENTER'),
                ]
                for i in range(1, len(rows)):
                    if i % 2 == 0:
                        style_cmds.append(('BACKGROUND', (0, i), (-1, i), self.C['success_bg']))
                t.setStyle(TableStyle(style_cmds))
                story.append(t)

    # ── E. Missed Opportunities & Critical Failures ──
    def _render_objections(self, story, content: str):
        """Render failures as professional structured cards with clear field labels."""
        content = self._convert_timestamps_to_mmss(content)
        entries = re.split(r'\n\s*\n', content)
        entry_num = 0
        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue
            fields = {}
            extra_lines = []
            for line in entry.split('\n'):
                line = line.strip()
                if not line:
                    continue
                matched = False
                for label in ['FAILURE TITLE', 'TIMESTAMP', 'QUOTE', 'WHAT HAPPENED',
                              'WHAT A TOP AGENT WOULD DO', 'WHAT WAS MISSED',
                              'REVENUE IMPACT', 'DEAL IMPACT', 'BUSINESS IMPACT']:
                    if line.upper().startswith(label + ':'):
                        value = line[len(label) + 1:].strip()
                        fields[label.upper()] = value
                        matched = True
                        break
                if not matched:
                    extra_lines.append(line)

            if not fields and not extra_lines:
                continue
            entry_num += 1

            # Build structured card
            title = fields.get('FAILURE TITLE', f'Issue {entry_num}')
            title = re.sub(r'\*\*(.*?)\*\*', r'\1', title)

            rows = []
            # Title row spanning full width
            rows.append([Paragraph(
                f"<b>Failure {entry_num}: {title}</b>",
                ParagraphStyle('ft', parent=self.ps['subsec'],
                               textColor=self.C['danger'], fontSize=11)),
                Paragraph('', self.ps['td'])])

            if 'TIMESTAMP' in fields:
                rows.append([
                    Paragraph("<b>When</b>", self.ps['td']),
                    Paragraph(f"<font color='#1565C0'><b>{fields['TIMESTAMP']}</b></font>", self.ps['td'])
                ])
            if 'QUOTE' in fields:
                q = fields['QUOTE'].strip('"').strip("'")
                rows.append([
                    Paragraph("<b>Agent Said</b>", self.ps['td']),
                    Paragraph(f'<i>"{q}"</i>', ParagraphStyle(
                        'fq', parent=self.ps['td'], textColor=self.C['medium'],
                        fontName='Helvetica-Oblique'))
                ])
            if 'WHAT HAPPENED' in fields:
                rows.append([
                    Paragraph("<b>What Went Wrong</b>", self.ps['td']),
                    Paragraph(fields['WHAT HAPPENED'], self.ps['td'])
                ])
            missed = fields.get('WHAT A TOP AGENT WOULD DO', fields.get('WHAT WAS MISSED', ''))
            if missed:
                rows.append([
                    Paragraph("<b>Correct Approach</b>", self.ps['td']),
                    Paragraph(f"<font color='#2E7D32'>{missed}</font>", self.ps['td'])
                ])
            impact = fields.get('REVENUE IMPACT', fields.get('DEAL IMPACT', fields.get('BUSINESS IMPACT', '')))
            if impact:
                rows.append([
                    Paragraph("<b>Revenue Impact</b>", self.ps['td']),
                    Paragraph(f"<font color='#C62828'><b>{impact}</b></font>", self.ps['td'])
                ])
            for el in extra_lines:
                el = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', el)
                if el.startswith('-') or el.startswith('*'):
                    rows.append([Paragraph('', self.ps['td']),
                                 Paragraph(f"<bullet>&bull;</bullet>{el.lstrip('-* ')}", self.ps['bullet'])])
                else:
                    rows.append([Paragraph('', self.ps['td']), Paragraph(el, self.ps['td'])])

            if rows:
                t = Table(rows, colWidths=[1.3 * inch, 4.9 * inch])
                style_cmds = [
                    ('BACKGROUND', (0, 0), (-1, 0), self.C['danger_bg']),
                    ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#FFF5F5')),
                    ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#FFCDD2')),
                    ('PADDING', (0, 0), (-1, -1), 6),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('SPAN', (0, 0), (1, 0)),
                ]
                t.setStyle(TableStyle(style_cmds))
                story.append(t)
                story.append(Spacer(1, 4))

    # ── F. Tactical Corrections & Coaching Playbook (Combined) ──
    def _render_tactical_playbook_combined(self, story, content: str):
        """Unified playbook: tactical corrections + coaching drills in structured cards."""
        content = self._convert_timestamps_to_mmss(content)
        entries = re.split(r'\n\s*\n', content)
        entry_num = 0

        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue

            # Parse structured fields
            fields = {}
            extra_lines = []
            for line in entry.split('\n'):
                line = line.strip()
                if not line:
                    continue
                matched = False
                for label in ['SCENARIO', 'CLIENT TRIGGER', 'RISK', 'ASSERTIVE APPROACH',
                              'CONSULTATIVE APPROACH', 'ASSERTIVE FRAME', 'CONSULTATIVE FRAME',
                              'CORRECT FRAME', 'WHY IT WORKS', 'ETHICS-SAFE NOTE']:
                    if line.upper().startswith(label + ':'):
                        fields[label.upper()] = line[len(label) + 1:].strip().strip('"')
                        matched = True
                        break
                if not matched:
                    extra_lines.append(line)

            if fields:
                entry_num += 1
                rows = []
                scenario = fields.get('SCENARIO', '')
                if scenario:
                    rows.append([
                        Paragraph(f"<b>Playbook {entry_num}: {scenario}</b>",
                                  ParagraphStyle('pb_t', parent=self.ps['subsec'],
                                                 textColor=self.C['primary_dark'], fontSize=10)),
                        Paragraph('', self.ps['td'])
                    ])
                if 'CLIENT TRIGGER' in fields:
                    rows.append([Paragraph("<b>Client Trigger</b>", self.ps['td']),
                                 Paragraph(f'<i>"{fields["CLIENT TRIGGER"]}"</i>', self.ps['td'])])
                if 'RISK' in fields:
                    rows.append([Paragraph("<b>Risk</b>", self.ps['td']),
                                 Paragraph(f"<font color='#C62828'>{fields['RISK']}</font>", self.ps['td'])])
                assertive = fields.get('ASSERTIVE APPROACH', fields.get('ASSERTIVE FRAME', fields.get('CORRECT FRAME', '')))
                if assertive:
                    rows.append([Paragraph("<b>Assertive Script</b>", self.ps['td']),
                                 Paragraph(f"<font color='#2E7D32'><i>\"{assertive}\"</i></font>", self.ps['td'])])
                consultative = fields.get('CONSULTATIVE APPROACH', fields.get('CONSULTATIVE FRAME', ''))
                if consultative:
                    rows.append([Paragraph("<b>Consultative Script</b>", self.ps['td']),
                                 Paragraph(f"<font color='#1565C0'><i>\"{consultative}\"</i></font>", self.ps['td'])])
                if 'WHY IT WORKS' in fields:
                    rows.append([Paragraph("<b>Why It Works</b>", self.ps['td']),
                                 Paragraph(f"<i>{fields['WHY IT WORKS']}</i>", self.ps['td'])])
                if 'ETHICS-SAFE NOTE' in fields:
                    rows.append([Paragraph("<b>Ethics Note</b>", self.ps['td']),
                                 Paragraph(f"<font color='#F57F17'><i>{fields['ETHICS-SAFE NOTE']}</i></font>", self.ps['td'])])

                if rows:
                    t = Table(rows, colWidths=[1.3 * inch, 4.9 * inch])
                    style_cmds = [
                        ('GRID', (0, 0), (-1, -1), 0.5, self.C['border']),
                        ('PADDING', (0, 0), (-1, -1), 6),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E3F2FD')),
                    ]
                    if scenario:
                        style_cmds.append(('SPAN', (0, 0), (1, 0)))
                        style_cmds.append(('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#E8EAF6')))
                    if entry_num % 2 == 0:
                        style_cmds.append(('BACKGROUND', (1, 1), (1, -1), colors.HexColor('#FAFAFA')))
                    t.setStyle(TableStyle(style_cmds))
                    story.append(t)
                    story.append(Spacer(1, 6))
            else:
                # Fallback: render as formatted text
                text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', entry)
                story.append(Paragraph(text, self.ps['body']))
                story.append(Spacer(1, 3))

    # ── H. 7-Day Focus ──
    def _render_7day_focus(self, story, content: str):
        content = self._convert_timestamps_to_mmss(content)
        story.append(Paragraph("<b>For the next 7 days, focus on ONLY these areas:</b>",
                               ParagraphStyle('focus_hdr', parent=self.ps['body_bold'],
                                              textColor=self.C['danger'], fontSize=10)))
        story.append(Spacer(1, 6))
        for line in content.split('\n'):
            line = line.strip()
            m = re.match(r'^(\d+)\.\s*(.*)', line)
            if m:
                num, text = m.group(1), m.group(2)
                text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
                t = Table([[
                    Paragraph(f"<b>{num}</b>", ParagraphStyle('fn', parent=self.ps['td_c'],
                              fontSize=14, textColor=self.C['white'])),
                    Paragraph(text, self.ps['body'])
                ]], colWidths=[0.4 * inch, 5.8 * inch])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, 0), self.C['primary_dark']),
                    ('BACKGROUND', (1, 0), (1, 0), colors.HexColor("#F5F7FA")),
                    ('BOX', (0, 0), (-1, -1), 0.5, self.C['border']),
                    ('PADDING', (0, 0), (-1, -1), 8),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                story.append(t)
                story.append(Spacer(1, 4))

    # ── I. 7-Week Development Roadmap ──
    def _render_7week_roadmap(self, story, content: str):
        content = self._convert_timestamps_to_mmss(content)
        rows = [[
            Paragraph("<b>Week</b>", self.ps['th']),
            Paragraph("<b>Focus Area</b>", self.ps['th']),
            Paragraph("<b>Drill / Exercise</b>", self.ps['th']),
        ]]
        for line in content.split('\n'):
            line = line.strip()
            m = re.match(r'Week\s*(\d+)\s*:\s*(.*?)(?:\s*--\s*|\s*[-:]\s*)(.*)', line, re.IGNORECASE)
            if m:
                wk, focus, drill = m.group(1), m.group(2).strip(), m.group(3).strip()
                rows.append([
                    Paragraph(f"<b>{wk}</b>", self.ps['td_c']),
                    Paragraph(focus, self.ps['td']),
                    Paragraph(drill, self.ps['td']),
                ])

        if len(rows) > 1:
            t = Table(rows, colWidths=[0.5 * inch, 2.5 * inch, 3.2 * inch])
            style_cmds = [
                ('BACKGROUND', (0, 0), (-1, 0), self.C['primary_dark']),
                ('TEXTCOLOR', (0, 0), (-1, 0), self.C['white']),
                ('GRID', (0, 0), (-1, -1), 0.5, self.C['border']),
                ('PADDING', (0, 0), (-1, -1), 6),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]
            for i in range(1, len(rows)):
                if i % 2 == 0:
                    style_cmds.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor("#F5F7FA")))
            t.setStyle(TableStyle(style_cmds))
            story.append(t)
        else:
            story.append(Paragraph(content, self.ps['body']))
        story.append(Spacer(1, 8))

    # ── J. Decisions & Commitments ──
    def _render_decisions(self, story, content: str):
        current = None
        for line in content.split('\n'):
            stripped = line.strip()
            if 'Decisions Made:' in stripped or 'Decisions:' in stripped:
                story.append(Paragraph("<b>Decisions Made</b>", self.ps['subsec']))
                current = 'decisions'
            elif 'Commitments Given:' in stripped or 'Commitments:' in stripped:
                story.append(Spacer(1, 6))
                story.append(Paragraph("<b>Commitments Given</b>", self.ps['subsec']))
                current = 'commitments'
            elif stripped.startswith('-') or stripped.startswith('*'):
                text = stripped.lstrip('-* ').strip()
                text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
                style = self.ps['bullet_success'] if current == 'decisions' else self.ps['bullet_accent']
                story.append(Paragraph(f"<bullet>&bull;</bullet>{text}", style))
                story.append(Spacer(1, 2))
            elif stripped:
                story.append(Paragraph(re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', stripped), self.ps['body']))

    # ── Action Items ──
    def _render_action_items(self, story, content: str):
        """Parse action items table from LLM output."""
        td_wrap = ParagraphStyle('td_act', parent=self.ps['td'], wordWrap='LTR', fontSize=8, leading=11)
        rows = [[
            Paragraph("<b>Task</b>", self.ps['th']),
            Paragraph("<b>Responsible</b>", self.ps['th']),
            Paragraph("<b>Deadline</b>", self.ps['th']),
            Paragraph("<b>Priority</b>", self.ps['th']),
            Paragraph("<b>Status</b>", self.ps['th']),
        ]]
        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('|---') or line.startswith('| Task') or line.startswith('|Task'):
                continue
            # Parse pipe-delimited table rows
            if '|' in line:
                parts = [p.strip() for p in line.split('|') if p.strip()]
                if len(parts) >= 4:
                    task = parts[0].replace('**', '')
                    responsible = parts[1] if len(parts) > 1 else ''
                    if responsible == 'Agent':
                        responsible = getattr(self, '_agent_name', 'Agent')
                    responsible = ' '.join(dict.fromkeys(responsible.split()))
                    deadline = parts[2] if len(parts) > 2 else ''
                    priority = parts[3] if len(parts) > 3 else 'Medium'
                    status = parts[4] if len(parts) > 4 else 'Pending'
                    rows.append([
                        Paragraph(task, td_wrap),
                        Paragraph(responsible, self.ps['td_c']),
                        Paragraph(deadline, td_wrap),
                        Paragraph(priority, self.ps['td_c']),
                        Paragraph(status, self.ps['td_c']),
                    ])
            elif line.startswith('-') or line.startswith('*'):
                text = line.lstrip('-* ').strip()
                rows.append([
                    Paragraph(text, td_wrap),
                    Paragraph('', self.ps['td_c']),
                    Paragraph('', td_wrap),
                    Paragraph('Medium', self.ps['td_c']),
                    Paragraph('Pending', self.ps['td_c']),
                ])

        if len(rows) > 1:
            t = Table(rows, colWidths=[2.2 * inch, 1.0 * inch, 1.2 * inch, 0.8 * inch, 0.8 * inch])
            style_cmds = [
                ('BACKGROUND', (0, 0), (-1, 0), self.C['primary_dark']),
                ('TEXTCOLOR', (0, 0), (-1, 0), self.C['white']),
                ('GRID', (0, 0), (-1, -1), 0.5, self.C['border']),
                ('PADDING', (0, 0), (-1, -1), 5),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]
            for i in range(1, len(rows)):
                if i % 2 == 0:
                    style_cmds.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor("#F5F7FA")))
            t.setStyle(TableStyle(style_cmds))
            story.append(t)
        else:
            # Fallback: render as text
            for line in content.split('\n'):
                line = line.strip()
                if line:
                    line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
                    story.append(Paragraph(line, self.ps['body']))
                    story.append(Spacer(1, 2))

    # ── K. Deal Intelligence (TEXT ONLY — chart goes to visualization section) ──
    def _render_deal_intelligence_text(self, story, content: str):
        """Parse pipe-delimited metrics and next steps. Hero + cards rendered here; chart deferred."""
        content = self._convert_timestamps_to_mmss(content)
        metrics = []
        next_steps = []
        in_next_steps = False
        deal_prob = None

        for line in content.split('\n'):
            stripped = line.strip()
            if not stripped:
                continue
            if 'Recommended Next Steps' in stripped:
                in_next_steps = True
                continue
            if in_next_steps:
                m = re.match(r'^\d+\.\s*(.*)', stripped)
                if m:
                    next_steps.append(m.group(1))
                elif stripped.startswith('-'):
                    next_steps.append(stripped.lstrip('- '))
                continue

            parts = [p.strip() for p in stripped.split('|')]
            if len(parts) >= 3:
                label = parts[0].replace('**', '')
                value = parts[1]
                just = parts[2] if len(parts) > 2 else ""
                confidence = parts[3].strip() if len(parts) > 3 else ""
                metrics.append((label, value, just, confidence))
                if 'probability' in label.lower():
                    deal_prob = value

        # Store for visualization
        self._cached_deal_metrics = metrics

        # Deal probability hero
        if deal_prob:
            try:
                prob_num = int(deal_prob.replace('%', '').strip())
            except ValueError:
                prob_num = 50
            color = "#43A047" if prob_num >= 70 else "#FF8F00" if prob_num >= 50 else "#E53935"
            prob_style = ParagraphStyle('dp_hero', parent=self.ps['score_big'],
                                        fontSize=40, leading=46, spaceAfter=0, spaceBefore=0)
            lbl_style = ParagraphStyle('dp_lbl', parent=self.ps['score_label'],
                                       fontSize=9, leading=12, spaceBefore=4, spaceAfter=0)
            hero = Table([
                [Paragraph(f"<font color='{color}'><b>{deal_prob}</b></font>", prob_style)],
                [Spacer(1, 4)],
                [Paragraph("OVERALL DEAL PROBABILITY", lbl_style)],
                [Paragraph("<i>AI-estimated confidence score based on observed signals</i>",
                          ParagraphStyle('dp_conf', parent=self.ps['conf'], fontSize=7,
                                         textColor=self.C['light']))],
            ], colWidths=[6.2 * inch], rowHeights=[52, 4, 18, 14])
            hero.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F5F7FA")),
                ('LINEBELOW', (0, -1), (-1, -1), 1.5, colors.HexColor(color)),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('LEFTPADDING', (0, 0), (-1, -1), 10),
                ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            story.append(hero)
            story.append(Spacer(1, 10))

        # Metric cards in 2-column grid
        non_prob = []
        for m_item in metrics:
            if 'probability' not in m_item[0].lower():
                non_prob.append(m_item)
        if non_prob:
            row_cells = []
            for m_item in non_prob:
                label = m_item[0]
                value = m_item[1]
                just = m_item[2] if len(m_item) > 2 else ""
                confidence = m_item[3] if len(m_item) > 3 else ""
                val_lower = value.lower()
                if val_lower in ['strong', 'high', 'low'] and 'risk' not in label.lower():
                    bg, tc = "#E8F5E9", "#2E7D32"
                elif val_lower in ['weak', 'low'] and 'risk' not in label.lower():
                    bg, tc = "#FFEBEE", "#C62828"
                elif val_lower in ['high'] and 'risk' in label.lower():
                    bg, tc = "#FFEBEE", "#C62828"
                elif val_lower in ['low'] and 'risk' in label.lower():
                    bg, tc = "#E8F5E9", "#2E7D32"
                else:
                    bg, tc = "#FFF8E1", "#F57F17"

                cell_content = [
                    Paragraph(f"<b>{value}</b>", ParagraphStyle('mv', parent=self.ps['body_bold'],
                              fontSize=14, textColor=colors.HexColor(tc), alignment=TA_CENTER)),
                    Spacer(1, 2),
                    Paragraph(label, ParagraphStyle('ml', parent=self.ps['body'],
                              fontSize=8, textColor=self.C['light'], alignment=TA_CENTER)),
                ]
                if just:
                    cell_content.append(Spacer(1, 2))
                    cell_content.append(Paragraph(f"<i>{just}</i>", ParagraphStyle(
                        'mj', parent=self.ps['body'], fontSize=7, textColor=self.C['light'], alignment=TA_CENTER)))
                if confidence:
                    cell_content.append(Spacer(1, 1))
                    cell_content.append(Paragraph(f"<i>{confidence}</i>", ParagraphStyle(
                        'mc', parent=self.ps['body'], fontSize=6, textColor=self.C['light'], alignment=TA_CENTER)))

                card = Table([[cell_content]], colWidths=[2.9 * inch])
                card.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(bg)),
                    ('PADDING', (0, 0), (-1, -1), 8),
                ]))
                row_cells.append(card)

            for i in range(0, len(row_cells), 2):
                pair = row_cells[i:i + 2]
                if len(pair) == 1:
                    pair.append(Spacer(1, 1))
                grid = Table([pair], colWidths=[3.1 * inch, 3.1 * inch])
                grid.setStyle(TableStyle([('PADDING', (0, 0), (-1, -1), 4)]))
                story.append(grid)
                story.append(Spacer(1, 4))

        # Next Steps
        if next_steps:
            story.append(Spacer(1, 6))
            story.append(Paragraph("<b>Recommended Next Steps</b>", self.ps['subsec']))
            for i, step in enumerate(next_steps, 1):
                story.append(Paragraph(f"<b>{i}.</b> {step}", self.ps['body']))
                story.append(Spacer(1, 2))

    # ── L. Comparative Performance Context ──
    def _render_comparative(self, story, content: str):
        td_wrap = ParagraphStyle('td_comp', parent=self.ps['td'], wordWrap='LTR', fontSize=8.5, leading=11.5)
        rows = [[
            Paragraph("<b>Metric</b>", self.ps['th']),
            Paragraph("<b>Baseline</b>", self.ps['th']),
            Paragraph("<b>Trend / Insight</b>", self.ps['th']),
        ]]
        note_text = ""
        for line in content.split('\n'):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.lower().startswith('note:'):
                note_text = stripped[5:].strip()
                continue
            parts = [p.strip() for p in stripped.split('|')]
            if len(parts) >= 2:
                label = parts[0].replace('**', '').strip()
                value = parts[1].strip()
                trend = parts[2].strip() if len(parts) >= 3 else '--'

                # Enrich baseline value with context if it's a bare number
                display_value = value
                if re.match(r'^\d+\.?\d*$', value):
                    num = float(value)
                    label_lower = label.lower()
                    if num <= 10:
                        # Likely a /10 score
                        display_value = f"{value}/10"
                    elif num <= 100 and any(w in label_lower for w in ['rate', 'ratio', 'percent', '%', 'resolution', 'clarity']):
                        display_value = f"{value}%"
                    else:
                        display_value = value
                elif re.match(r'^\d+%$', value):
                    display_value = value  # already has %
                elif '/' in value:
                    display_value = value  # already has /N format

                # Color-code the baseline value
                val_color = self.C['dark']
                try:
                    num_val = float(re.sub(r'[^\d.]', '', value))
                    if num_val <= 10:
                        pct = num_val / 10
                        val_color = self.C['success'] if pct >= 0.75 else colors.HexColor("#F57F17") if pct >= 0.50 else self.C['danger']
                except (ValueError, ZeroDivisionError):
                    pass

                # Determine trend color indicator
                trend_color = self.C['dark']
                trend_lower = trend.lower()
                if any(w in trend_lower for w in ['below', 'weak', 'poor', 'low', 'needs', 'decline']):
                    trend_color = self.C['danger']
                elif any(w in trend_lower for w in ['above', 'strong', 'good', 'high', 'exceed']):
                    trend_color = self.C['success']
                elif any(w in trend_lower for w in ['moderate', 'average', 'room']):
                    trend_color = colors.HexColor("#F57F17")
                # Get hex string from color
                if hasattr(trend_color, 'hexval'):
                    tc_hex = trend_color.hexval()
                elif hasattr(trend_color, 'red'):
                    tc_hex = '#%02x%02x%02x' % (int(trend_color.red*255), int(trend_color.green*255), int(trend_color.blue*255))
                else:
                    tc_hex = str(trend_color)
                if hasattr(val_color, 'hexval'):
                    vc_hex = val_color.hexval()
                elif hasattr(val_color, 'red'):
                    vc_hex = '#%02x%02x%02x' % (int(val_color.red*255), int(val_color.green*255), int(val_color.blue*255))
                else:
                    vc_hex = str(val_color)
                rows.append([
                    Paragraph(f"<b>{label}</b>", td_wrap),
                    Paragraph(f"<b><font color='{vc_hex}'>{display_value}</font></b>", self.ps['td_c']),
                    Paragraph(f"<font color='{tc_hex}'>{trend}</font>", td_wrap),
                ])

        if len(rows) > 1:
            t = Table(rows, colWidths=[2.0 * inch, 1.2 * inch, 3.0 * inch])
            style_cmds = [
                ('BACKGROUND', (0, 0), (-1, 0), self.C['primary_dark']),
                ('TEXTCOLOR', (0, 0), (-1, 0), self.C['white']),
                ('GRID', (0, 0), (-1, -1), 0.4, self.C['border']),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]
            for i in range(1, len(rows)):
                if i % 2 == 0:
                    style_cmds.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor("#F5F7FA")))
            t.setStyle(TableStyle(style_cmds))
            story.append(t)
            # Add explanatory note about the scale
            story.append(Spacer(1, 4))
            story.append(Paragraph(
                "<i>Note: Baseline scores shown as X/10 represent performance on a 10-point scale. "
                "Percentages (%) indicate rate-based metrics. Color coding: "
                "<font color='#2E7D32'>Green</font> = strong, "
                "<font color='#F57F17'>Amber</font> = moderate, "
                "<font color='#C62828'>Red</font> = needs improvement.</i>",
                ParagraphStyle('comp_note', parent=self.ps['body'], textColor=self.C['light'], fontSize=7.5)))
        else:
            # Fallback: render content as text
            for line in content.split('\n'):
                line = line.strip()
                if line:
                    line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
                    story.append(Paragraph(line, self.ps['body']))
                    story.append(Spacer(1, 2))

        if note_text:
            story.append(Spacer(1, 6))
            story.append(Paragraph(f"<i>Note: {note_text}</i>", ParagraphStyle(
                'note', parent=self.ps['body'], textColor=self.C['light'], fontSize=8)))
        story.append(Spacer(1, 6))

    # ── M. Agent Tier Classification ──
    def _render_agent_tier(self, story, content: str):
        tier_line = ""
        tier_scale = []
        promotion = []
        in_scale = False
        in_promo = False
        level_num = 3  # default

        for line in content.split('\n'):
            stripped = line.strip()
            if stripped.lower().startswith('agent tier:'):
                tier_line = stripped.split(':', 1)[1].strip()
            elif stripped.lower().startswith('tier scale:'):
                in_scale = True
                in_promo = False
            elif stripped.lower().startswith('promotion criteria'):
                in_promo = True
                in_scale = False
            elif in_scale and stripped.startswith('Level'):
                tier_scale.append(stripped)
            elif in_promo and (stripped.startswith('-') or stripped.startswith('*')):
                promotion.append(stripped.lstrip('-* ').strip())
            elif in_scale and not stripped:
                in_scale = False
            elif in_promo and not stripped:
                in_promo = False

        # Tier badge
        if tier_line:
            level_match = re.search(r'Level\s*(\d)', tier_line)
            level_num = int(level_match.group(1)) if level_match else 3
            level_colors = {
                1: ("#FFEBEE", "#C62828"),
                2: ("#FFF8E1", "#F57F17"),
                3: ("#E3F2FD", "#1565C0"),
                4: ("#E8F5E9", "#2E7D32"),
                5: ("#F3E5F5", "#6A1B9A"),
            }
            bg, tc = level_colors.get(level_num, ("#E3F2FD", "#1565C0"))

            badge = Table([
                [Paragraph(tier_line, ParagraphStyle('tb', parent=self.ps['tier'],
                           textColor=colors.HexColor(tc), fontSize=18, leading=24))],
            ], colWidths=[6.4 * inch])
            badge.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(bg)),
                ('BOX', (0, 0), (-1, -1), 1.5, colors.HexColor(tc)),
                ('TOPPADDING', (0, 0), (-1, -1), 16),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 16),
                ('LEFTPADDING', (0, 0), (-1, -1), 20),
                ('RIGHTPADDING', (0, 0), (-1, -1), 20),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            story.append(badge)
            story.append(Spacer(1, 10))

        # Tier scale table — use fallback if LLM didn't provide them
        if not tier_scale:
            tier_scale = [
                "Level 1: Trainee -- Fundamental gaps in sales methodology; needs intensive coaching",
                "Level 2: Developing -- Shows awareness of techniques but inconsistent execution",
                "Level 3: Competent -- Solid fundamentals with specific areas for growth",
                "Level 4: Advanced -- Strong performer with minor refinements needed",
                "Level 5: Elite -- Exceptional across all dimensions; ready to mentor others",
            ]
        if tier_scale:
            story.append(Paragraph("<b>Tier Scale</b>", self.ps['subsec']))
            trows = [[Paragraph("<b>Level</b>", self.ps['th']),
                      Paragraph("<b>Title</b>", self.ps['th']),
                      Paragraph("<b>Description</b>", self.ps['th'])]]
            for ts_line in tier_scale:
                m = re.match(r'Level\s*(\d+)\s*:\s*(.*?)(?:\s*--\s*|\s*[-:]\s*)(.*)', ts_line)
                if m:
                    trows.append([
                        Paragraph(m.group(1), self.ps['td_c']),
                        Paragraph(f"<b>{m.group(2).strip()}</b>", self.ps['td']),
                        Paragraph(m.group(3).strip(), self.ps['td']),
                    ])
            if len(trows) > 1:
                t = Table(trows, colWidths=[0.6 * inch, 1.5 * inch, 4.1 * inch])
                style_cmds = [
                    ('BACKGROUND', (0, 0), (-1, 0), self.C['primary_dark']),
                    ('TEXTCOLOR', (0, 0), (-1, 0), self.C['white']),
                    ('GRID', (0, 0), (-1, -1), 0.5, self.C['border']),
                    ('PADDING', (0, 0), (-1, -1), 5),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ]
                for i, ts_line in enumerate(tier_scale, 1):
                    lm = re.match(r'Level\s*(\d+)', ts_line)
                    if lm and int(lm.group(1)) == level_num:
                        style_cmds.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor("#E3F2FD")))
                t.setStyle(TableStyle(style_cmds))
                story.append(t)
                story.append(Spacer(1, 8))

        # Promotion criteria
        if promotion:
            story.append(Paragraph("<b>Promotion Criteria (Next Level)</b>", self.ps['subsec']))
            for crit in promotion:
                crit = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', crit)
                t = Table([[Paragraph(f"<bullet>&bull;</bullet>{crit}", self.ps['bullet_accent'])]],
                          colWidths=[6.2 * inch])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), self.C['warning_bg']),
                    ('PADDING', (0, 0), (-1, -1), 5),
                ]))
                story.append(t)
                story.append(Spacer(1, 3))

    # ════════════════════════════════════════════════════════
    # NEW INTELLIGENCE LAYER RENDERERS (N through S)
    # ════════════════════════════════════════════════════════

    # ── N. Client Sentiment Timeline ──
    def _render_client_sentiment(self, story, content: str):
        """Render client emotional journey with structured risk/solution cards per phase."""
        content = self._convert_timestamps_to_mmss(content)
        # Intro
        story.append(Paragraph(
            "This section maps the client's emotional state throughout the meeting. "
            "Each phase identifies the dominant sentiment, concrete risks to the deal, "
            "and the recommended corrective action the agent should have taken.",
            ParagraphStyle('sent_intro', parent=self.ps['body'], fontSize=9,
                           textColor=self.C['medium'], spaceAfter=8)))

        phases = []
        turning_points = []
        trust_arc = ""
        trajectory = ""
        current_phase = None
        current_lines = []
        in_turning = False
        in_trust = False

        for line in content.split('\n'):
            stripped = line.strip()
            if not stripped:
                continue
            phase_match = re.match(r'^(EARLY|MID|LATE)\s*PHASE.*:', stripped, re.IGNORECASE)
            if phase_match:
                if current_phase and current_lines:
                    phases.append((current_phase, '\n'.join(current_lines)))
                current_phase = phase_match.group(1).upper()
                current_lines = []
                in_turning = False
                in_trust = False
                continue
            if 'EMOTIONAL TURNING POINT' in stripped.upper():
                if current_phase and current_lines:
                    phases.append((current_phase, '\n'.join(current_lines)))
                    current_phase = None
                    current_lines = []
                in_turning = True
                in_trust = False
                continue
            if 'TRUST ARC' in stripped.upper():
                if current_phase and current_lines:
                    phases.append((current_phase, '\n'.join(current_lines)))
                    current_phase = None
                    current_lines = []
                in_turning = False
                in_trust = True
                val = stripped.split(':', 1)
                if len(val) > 1 and val[1].strip():
                    trust_arc = val[1].strip()
                continue
            if 'OVERALL SENTIMENT TRAJECTORY' in stripped.upper():
                in_turning = False
                in_trust = False
                val = stripped.split(':', 1)
                if len(val) > 1:
                    trajectory = val[1].strip()
                continue
            if in_turning:
                turning_points.append(stripped.lstrip('-* ').strip() if stripped.startswith(('-', '*')) else stripped)
            elif in_trust:
                trust_arc += " " + stripped
            elif current_phase:
                current_lines.append(stripped)

        if current_phase and current_lines:
            phases.append((current_phase, '\n'.join(current_lines)))

        # Phase cards — structured table rows with risk + solution
        phase_colors = {
            'EARLY': ('#E3F2FD', '#1565C0'),
            'MID': ('#FFF8E1', '#F57F17'),
            'LATE': ('#F3E5F5', '#6A1B9A'),
        }
        for phase_name, phase_content in phases:
            bg, tc = phase_colors.get(phase_name, ('#F5F7FA', '#424242'))
            # Parse fields from phase content
            fields = {}
            extra = []
            for pl in phase_content.split('\n'):
                pl = pl.strip()
                if not pl:
                    continue
                matched = False
                for lbl in ['SENTIMENT', 'EVIDENCE', 'TIMESTAMP', 'RISK', 'SOLUTION',
                            'DEAL RISK', 'RECOMMENDED ACTION', 'CLIENT SIGNAL']:
                    if pl.upper().startswith(lbl + ':'):
                        fields[lbl.upper()] = pl[len(lbl) + 1:].strip()
                        matched = True
                        break
                if not matched:
                    extra.append(pl)

            rows = []
            # Phase title row
            rows.append([
                Paragraph(f"<b>{phase_name} PHASE</b>",
                          ParagraphStyle('ph_' + phase_name, parent=self.ps['subsec'],
                                         textColor=colors.HexColor(tc), fontSize=10)),
                Paragraph('', self.ps['td'])
            ])
            if 'SENTIMENT' in fields:
                rows.append([Paragraph("<b>Sentiment</b>", self.ps['td']),
                             Paragraph(fields['SENTIMENT'], self.ps['td'])])
            if 'TIMESTAMP' in fields:
                rows.append([Paragraph("<b>When</b>", self.ps['td']),
                             Paragraph(f"<font color='#1565C0'><b>{fields['TIMESTAMP']}</b></font>", self.ps['td'])])
            if 'EVIDENCE' in fields or 'CLIENT SIGNAL' in fields:
                ev = fields.get('EVIDENCE', fields.get('CLIENT SIGNAL', ''))
                rows.append([Paragraph("<b>Evidence</b>", self.ps['td']),
                             Paragraph(f"<i>{ev}</i>", self.ps['td'])])
            risk = fields.get('RISK', fields.get('DEAL RISK', ''))
            if risk:
                rows.append([Paragraph("<b>Deal Risk</b>", self.ps['td']),
                             Paragraph(f"<font color='#C62828'><b>{risk}</b></font>", self.ps['td'])])
            solution = fields.get('SOLUTION', fields.get('RECOMMENDED ACTION', ''))
            if solution:
                rows.append([Paragraph("<b>Solution</b>", self.ps['td']),
                             Paragraph(f"<font color='#2E7D32'>{solution}</font>", self.ps['td'])])
            for el in extra:
                el = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', el)
                rows.append([Paragraph('', self.ps['td']), Paragraph(el, self.ps['td'])])

            if rows:
                t = Table(rows, colWidths=[1.3 * inch, 4.9 * inch])
                style_cmds = [
                    ('SPAN', (0, 0), (1, 0)),
                    ('BACKGROUND', (0, 0), (1, 0), colors.HexColor(bg)),
                    ('BACKGROUND', (0, 1), (0, -1), colors.HexColor(bg)),
                    ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor(tc)),
                    ('PADDING', (0, 0), (-1, -1), 6),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ]
                t.setStyle(TableStyle(style_cmds))
                story.append(t)
                story.append(Spacer(1, 5))

        # Turning Points
        if turning_points:
            story.append(Paragraph("<b>Emotional Turning Points</b>", self.ps['subsec']))
            for tp in turning_points:
                tp = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', tp)
                tp = re.sub(r'\[(\d+[\.\d]*s?\s*-\s*\d+[\.\d]*s?)\]',
                            r"<font color='#1565C0'><b>[\1]</b></font>", tp)
                story.append(Paragraph(f"<bullet>&bull;</bullet>{tp}", self.ps['bullet_accent']))
                story.append(Spacer(1, 2))

        # Trust Arc
        if trust_arc:
            story.append(Spacer(1, 4))
            t = Table([[Paragraph(f"<b>Trust Arc:</b> {trust_arc.strip()}", self.ps['body'])]],
                      colWidths=[6.2 * inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#E8F5E9')),
                ('PADDING', (0, 0), (-1, -1), 8),
            ]))
            story.append(t)

        # Trajectory
        if trajectory:
            story.append(Spacer(1, 6))
            story.append(Paragraph(f"<b>Overall Sentiment Trajectory:</b> <i>{trajectory}</i>",
                         ParagraphStyle('traj', parent=self.ps['body_bold'], fontSize=10,
                                        textColor=self.C['primary_dark'])))

        # Cache for visualization section
        self._cached_sentiment_phases = phases
        self._cached_turning_points = turning_points
        self._cached_trust_arc = trust_arc

    # ── N. Communication Balance & Listening Evaluation ──
    def _render_listening_intelligence(self, story, content: str):
        """Render listening evaluation with verdicts and recommendations per area."""
        content = self._convert_timestamps_to_mmss(content)
        subsections = {}
        current_key = None
        current_lines = []

        for line in content.split('\n'):
            stripped = line.strip()
            if not stripped:
                continue
            # Broader matching: look for known subsection headers
            header_match = re.match(
                r'^(TALK RATIO|INTERRUPTION ANALYSIS|REFLECTIVE LISTENING.*|'
                r'FOLLOW-UP QUESTION.*|ACTIVE LISTENING GAPS|CRITICAL LISTENING FAILURES?|'
                r'LISTENING FAILURE).*:', stripped, re.IGNORECASE)
            if header_match:
                if current_key and current_lines:
                    subsections[current_key] = '\n'.join(current_lines)
                raw_key = header_match.group(1).upper().strip()
                # Normalize to standard keys
                if 'TALK' in raw_key:
                    current_key = 'TALK RATIO'
                elif 'INTERRUPT' in raw_key:
                    current_key = 'INTERRUPTION ANALYSIS'
                elif 'REFLECT' in raw_key:
                    current_key = 'REFLECTIVE LISTENING'
                elif 'FOLLOW' in raw_key:
                    current_key = 'FOLLOW-UP QUESTION'
                elif 'CRITICAL' in raw_key or 'FAILURE' in raw_key:
                    current_key = 'CRITICAL LISTENING FAILURES'
                else:
                    current_key = raw_key
                remainder = stripped[header_match.end():].strip()
                current_lines = [remainder] if remainder else []
                continue
            # Also detect ** bold headers ** as subsection markers
            bold_header = re.match(r'^\*\*(TALK RATIO|INTERRUPTION|REFLECTIVE|FOLLOW.UP|CRITICAL|LISTENING).*?\*\*', stripped, re.IGNORECASE)
            if bold_header:
                if current_key and current_lines:
                    subsections[current_key] = '\n'.join(current_lines)
                raw_key = bold_header.group(1).upper().strip()
                if 'TALK' in raw_key:
                    current_key = 'TALK RATIO'
                elif 'INTERRUPT' in raw_key:
                    current_key = 'INTERRUPTION ANALYSIS'
                elif 'REFLECT' in raw_key:
                    current_key = 'REFLECTIVE LISTENING'
                elif 'FOLLOW' in raw_key:
                    current_key = 'FOLLOW-UP QUESTION'
                elif 'CRITICAL' in raw_key or 'LISTENING' in raw_key:
                    current_key = 'CRITICAL LISTENING FAILURES'
                else:
                    current_key = raw_key
                current_lines = []
                continue
            if current_key:
                current_lines.append(stripped)

        if current_key and current_lines:
            subsections[current_key] = '\n'.join(current_lines)

        # FALLBACK: If no subsections were detected at all, parse entire content generically
        if not subsections:
            # Try to extract talk ratio from anywhere in content
            agent_match = re.search(r'(?:Agent|Estimated Agent).*?(\d+)\s*%', content, re.IGNORECASE)
            client_match = re.search(r'(?:Client|Estimated Client).*?(\d+)\s*%', content, re.IGNORECASE)
            if agent_match and client_match:
                subsections['TALK RATIO'] = f"Estimated Agent Talk Time: {agent_match.group(1)}%\nEstimated Client Talk Time: {client_match.group(1)}%"
            # Put everything else as generic content
            remaining_lines = []
            for line in content.split('\n'):
                stripped = line.strip()
                if stripped and not re.match(r'Estimated (Agent|Client)', stripped, re.IGNORECASE):
                    remaining_lines.append(stripped)
            if remaining_lines:
                subsections.setdefault('GENERAL', '\n'.join(remaining_lines))

        # Talk Ratio visual — NO AI-estimated note
        talk_content = subsections.get('TALK RATIO', '')
        if talk_content:
            story.append(Paragraph("<b>Talk Ratio Analysis</b>", self.ps['subsec']))
            agent_pct = re.search(r'Agent.*?(\d+)%', talk_content)
            client_pct = re.search(r'Client.*?(\d+)%', talk_content)
            if agent_pct and client_pct:
                a_val = int(agent_pct.group(1))
                c_val = int(client_pct.group(1))
                a_color = '#E53935' if a_val > 65 else '#FF8F00' if a_val > 55 else '#43A047'
                c_color = '#43A047' if c_val > 35 else '#FF8F00'
                a_verdict = "Agent-dominated — too much talking" if a_val > 65 else "Balanced" if a_val <= 55 else "Slightly high"
                c_verdict = "Good engagement" if c_val > 35 else "Low participation — not enough probing"
                ratio_data = [
                    [Paragraph("<b>Speaker</b>", self.ps['th']),
                     Paragraph("<b>Talk Time</b>", self.ps['th']),
                     Paragraph("<b>Verdict</b>", self.ps['th']),
                     Paragraph("<b>Recommendation</b>", self.ps['th'])],
                    [Paragraph("Agent", self.ps['td']),
                     Paragraph(f"<b><font color='{a_color}'>{a_val}%</font></b>", self.ps['td_c']),
                     Paragraph(a_verdict, self.ps['td']),
                     Paragraph("Ask more open-ended questions; aim for 40-50% talk time" if a_val > 55
                               else "Maintain current approach", self.ps['td'])],
                    [Paragraph("Client", self.ps['td']),
                     Paragraph(f"<b><font color='{c_color}'>{c_val}%</font></b>", self.ps['td_c']),
                     Paragraph(c_verdict, self.ps['td']),
                     Paragraph("Use silence and probing to get client talking more" if c_val <= 35
                               else "Keep encouraging engagement", self.ps['td'])],
                ]
                rt = Table(ratio_data, colWidths=[1.0 * inch, 1.0 * inch, 2.0 * inch, 2.2 * inch])
                rt.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), self.C['primary_dark']),
                    ('TEXTCOLOR', (0, 0), (-1, 0), self.C['white']),
                    ('GRID', (0, 0), (-1, -1), 0.5, self.C['border']),
                    ('PADDING', (0, 0), (-1, -1), 6),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                story.append(rt)
            else:
                for tl in talk_content.split('\n'):
                    tl = tl.strip()
                    if tl:
                        tl = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', tl)
                        story.append(Paragraph(tl, self.ps['body']))
                        story.append(Spacer(1, 2))
            story.append(Spacer(1, 6))

        # Other subsections — structured with verdict color coding
        sub_config = [
            ('INTERRUPTION ANALYSIS', 'Interruption Analysis', '#FFEBEE', '#C62828'),
            ('REFLECTIVE LISTENING', 'Reflective Listening', '#E8F5E9', '#2E7D32'),
            ('FOLLOW-UP QUESTION', 'Follow-Up Question Quality', '#E3F2FD', '#1565C0'),
            ('ACTIVE LISTENING GAPS', 'Active Listening Gaps', '#FFF8E1', '#F57F17'),
            ('CRITICAL LISTENING FAILURES', 'Critical Listening Failures', '#FFEBEE', '#C62828'),
            ('GENERAL', 'Detailed Analysis', '#F5F7FA', '#424242'),
        ]
        for key, display, bg, tc in sub_config:
            sub_content = subsections.get(key, '')
            if not sub_content:
                continue
            inner = [
                Paragraph(f"<b>{display}</b>",
                          ParagraphStyle('li_' + key, parent=self.ps['subsec'],
                                         textColor=colors.HexColor(tc))),
                Spacer(1, 3),
            ]
            for sl in sub_content.split('\n'):
                sl = sl.strip()
                if not sl:
                    continue
                sl = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', sl)
                sl = re.sub(r'\[(\d+[\.\d]*s?\s*-\s*\d+[\.\d]*s?)\]',
                            r"<font color='#1565C0'><b>[\1]</b></font>", sl)
                if sl.startswith('-') or sl.startswith('*'):
                    inner.append(Paragraph(f"<bullet>&bull;</bullet>{sl.lstrip('-* ')}",
                                           self.ps['bullet']))
                elif re.match(r'^Score:', sl, re.IGNORECASE):
                    inner.append(Paragraph(f"<b>{sl}</b>", self.ps['body_bold']))
                elif re.match(r'^Verdict:', sl, re.IGNORECASE):
                    inner.append(Paragraph(f"<font color='{tc}'><b>{sl}</b></font>", self.ps['body_bold']))
                elif re.match(r'^Recommendation:', sl, re.IGNORECASE):
                    inner.append(Paragraph(f"<font color='#2E7D32'>{sl}</font>", self.ps['body']))
                else:
                    inner.append(Paragraph(sl, self.ps['body']))
                inner.append(Spacer(1, 2))

            card = Table([[inner]], colWidths=[6.2 * inch])
            card.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(bg)),
                ('PADDING', (0, 0), (-1, -1), 8),
                ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor(tc)),
            ]))
            story.append(card)
            story.append(Spacer(1, 5))

        # Cache talk ratio for visualizations
        talk_content = subsections.get('TALK RATIO', '')
        _a_match = re.search(r'Agent.*?(\d+)%', talk_content) if talk_content else None
        _c_match = re.search(r'Client.*?(\d+)%', talk_content) if talk_content else None
        self._cached_talk_ratio = {
            'agent': int(_a_match.group(1)) if _a_match else None,
            'client': int(_c_match.group(1)) if _c_match else None,
        }
        self._cached_listening_subsections = subsections

    # ── O. Five Core Discovery Pillars ──
    def _render_discovery_matrix(self, story, content: str):
        """Render the 5 essential discovery pillars every top agent must cover."""
        content = self._convert_timestamps_to_mmss(content)
        # Intro explanation
        story.append(Paragraph(
            "Every successful listing presentation requires thorough discovery across five essential pillars. "
            "Failing to explore any of these areas leaves money on the table and creates deal risk. "
            "Below is the agent's coverage assessment:",
            ParagraphStyle('disc_intro', parent=self.ps['body'], fontSize=9,
                           textColor=self.C['medium'], spaceAfter=8)))

        pillars = []
        score_line = ""
        missing_qs = []
        in_missing = False

        for line in content.split('\n'):
            stripped = line.strip()
            if not stripped:
                continue
            if 'DISCOVERY COMPLETENESS SCORE' in stripped.upper():
                score_line = stripped.split(':', 1)[-1].strip()
                in_missing = False
                continue
            if 'MISSING DISCOVERY QUESTION' in stripped.upper():
                in_missing = True
                continue
            if in_missing:
                if stripped.startswith('-') or stripped.startswith('*'):
                    missing_qs.append(stripped.lstrip('-* ').strip())
                elif re.match(r'^\d+\.', stripped):
                    missing_qs.append(re.sub(r'^\d+\.\s*', '', stripped).strip())
                continue

            parts = [p.strip() for p in stripped.split('|')]
            if len(parts) >= 3:
                pillar = parts[0].replace('**', '').strip()
                status = parts[1].strip()
                evidence = parts[2].strip() if len(parts) > 2 else ''
                confidence = parts[3].strip() if len(parts) > 3 else ''
                if pillar and not pillar.lower().startswith('pillar'):
                    pillars.append((pillar, status, evidence, confidence))

        # Matrix table
        if pillars:
            td_wrap = ParagraphStyle('td_disc', parent=self.ps['td'], wordWrap='LTR',
                                     fontSize=8, leading=11)
            rows = [[
                Paragraph("<b>Discovery Pillar</b>", self.ps['th']),
                Paragraph("<b>Status</b>", self.ps['th']),
                Paragraph("<b>Evidence</b>", self.ps['th']),
                Paragraph("<b>Confidence</b>", self.ps['th']),
            ]]
            for pillar, status, evidence, confidence in pillars:
                status_lower = status.lower()
                if 'confirmed' in status_lower:
                    s_color = '#2E7D32'
                elif 'partially' in status_lower:
                    s_color = '#F57F17'
                else:
                    s_color = '#C62828'
                rows.append([
                    Paragraph(f"<b>{pillar}</b>", self.ps['td']),
                    Paragraph(f"<font color='{s_color}'><b>{status}</b></font>", self.ps['td_c']),
                    Paragraph(f"<i>{evidence}</i>", td_wrap),
                    Paragraph(f"<i>{confidence}</i>",
                             ParagraphStyle('dc', parent=self.ps['td'], fontSize=7,
                                            textColor=self.C['light'])),
                ])

            t = Table(rows, colWidths=[1.4 * inch, 1.2 * inch, 2.6 * inch, 1.0 * inch])
            style_cmds = [
                ('BACKGROUND', (0, 0), (-1, 0), self.C['primary_dark']),
                ('TEXTCOLOR', (0, 0), (-1, 0), self.C['white']),
                ('GRID', (0, 0), (-1, -1), 0.5, self.C['border']),
                ('PADDING', (0, 0), (-1, -1), 5),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]
            for i, (_, status, _, _) in enumerate(pillars, 1):
                sl = status.lower()
                if 'confirmed' in sl:
                    style_cmds.append(('BACKGROUND', (1, i), (1, i), colors.HexColor('#E8F5E9')))
                elif 'partially' in sl:
                    style_cmds.append(('BACKGROUND', (1, i), (1, i), colors.HexColor('#FFF8E1')))
                elif 'not' in sl:
                    style_cmds.append(('BACKGROUND', (1, i), (1, i), colors.HexColor('#FFEBEE')))
            t.setStyle(TableStyle(style_cmds))
            story.append(t)
            story.append(Spacer(1, 6))

        # Score
        if score_line:
            score_match = re.search(r'(\d+)', score_line)
            s_val = int(score_match.group(1)) if score_match else 0
            s_color = '#43A047' if s_val >= 4 else '#FF8F00' if s_val >= 3 else '#E53935'
            story.append(Paragraph(
                f"<b>Pillar Coverage Score:</b> <font color='{s_color}'><b>{score_line}</b></font>",
                ParagraphStyle('dcs', parent=self.ps['body_bold'], fontSize=11,
                               textColor=self.C['primary_dark'])))
            story.append(Spacer(1, 6))

        # Missing questions
        if missing_qs:
            story.append(Paragraph("<b>Questions the Agent Must Ask Next Time</b>", self.ps['subsec']))
            for i, q in enumerate(missing_qs, 1):
                q = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', q)
                story.append(Paragraph(f"<b>{i}.</b> {q}", self.ps['bullet_danger']))
                story.append(Spacer(1, 2))

        # Cache for visualization
        self._cached_discovery_pillars = pillars

    # ── Q. Emotional Intelligence Assessment ──
    def _render_emotional_intelligence(self, story, content: str):
        """Render EQ assessment with score cards for each dimension."""
        content = self._convert_timestamps_to_mmss(content)
        eq_scores = {}
        overall_eq = ""
        ai_conf = ""
        current_dim = None
        current_lines = []

        dimensions = ['EMPATHY', 'ADAPTABILITY', 'SOCIAL AWARENESS', 'EMOTIONAL REGULATION']

        for line in content.split('\n'):
            stripped = line.strip()
            if not stripped:
                continue

            dim_match = None
            for dim in dimensions:
                if stripped.upper().startswith(dim + ' SCORE'):
                    dim_match = dim
                    break

            if dim_match:
                if current_dim and current_lines:
                    eq_scores[current_dim] = '\n'.join(current_lines)
                current_dim = dim_match
                score_part = stripped.split(':', 1)
                current_lines = [score_part[1].strip()] if len(score_part) > 1 and score_part[1].strip() else []
                continue

            if 'OVERALL EQ SCORE' in stripped.upper():
                if current_dim and current_lines:
                    eq_scores[current_dim] = '\n'.join(current_lines)
                    current_dim = None
                    current_lines = []
                val = stripped.split(':', 1)
                if len(val) > 1:
                    overall_eq = val[1].strip()
                continue

            if 'AI CONFIDENCE' in stripped.upper() and not current_dim:
                val = stripped.split(':', 1)
                if len(val) > 1:
                    ai_conf = val[1].strip()
                continue

            if current_dim:
                current_lines.append(stripped)

        if current_dim and current_lines:
            eq_scores[current_dim] = '\n'.join(current_lines)

        # Overall EQ hero
        if overall_eq:
            score_match = re.search(r'(\d+)\s*/\s*(\d+)', overall_eq)
            if score_match:
                s_val = int(score_match.group(1))
                pct = s_val / 10
                color = '#43A047' if pct >= 0.75 else '#FF8F00' if pct >= 0.50 else '#E53935'
                hero = Table([
                    [Paragraph(f"<font color='{color}'><b>{overall_eq}</b></font>",
                     ParagraphStyle('eq_hero', parent=self.ps['score_big'], fontSize=32, leading=38))],
                    [Spacer(1, 1)],
                    [Paragraph("EMOTIONAL INTELLIGENCE SCORE", self.ps['score_label'])],
                    [Paragraph(f"<i>{ai_conf}</i>" if ai_conf else "",
                     ParagraphStyle('eq_conf', parent=self.ps['conf'], fontSize=7))],
                ], colWidths=[6.2 * inch], rowHeights=[44, 6, 16, 14])
                hero.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F5F7FA')),
                    ('LINEBELOW', (0, -1), (-1, -1), 1.5, colors.HexColor(color)),
                    ('PADDING', (0, 0), (-1, -1), 8),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                story.append(hero)
                story.append(Spacer(1, 10))

        # Dimension cards
        dim_display = {
            'EMPATHY': ('Empathy', '#E3F2FD', '#1565C0'),
            'ADAPTABILITY': ('Adaptability', '#E8F5E9', '#2E7D32'),
            'SOCIAL AWARENESS': ('Social Awareness', '#FFF8E1', '#F57F17'),
            'EMOTIONAL REGULATION': ('Emotional Regulation', '#F3E5F5', '#6A1B9A'),
        }
        for dim, dim_content in eq_scores.items():
            display_name, bg, tc = dim_display.get(dim, (dim.title(), '#F5F7FA', '#424242'))
            inner = [Paragraph(f"<b>{display_name}</b>",
                     ParagraphStyle('eq_' + dim, parent=self.ps['subsec'],
                                    textColor=colors.HexColor(tc)))]
            for dl in dim_content.split('\n'):
                dl = dl.strip()
                if not dl:
                    continue
                dl = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', dl)
                dl = re.sub(r'\[(\d+[\.\d]*s?\s*-\s*\d+[\.\d]*s?)\]',
                            r"<font color='#1565C0'><b>[\1]</b></font>", dl)
                if dl.startswith('-') or dl.startswith('*'):
                    inner.append(Paragraph(f"<bullet>&bull;</bullet>{dl.lstrip('-* ')}",
                                           self.ps['bullet']))
                else:
                    inner.append(Paragraph(dl, self.ps['body']))
                inner.append(Spacer(1, 2))

            card = Table([[inner]], colWidths=[6.2 * inch])
            card.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(bg)),
                ('PADDING', (0, 0), (-1, -1), 8),
                ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor(tc)),
            ]))
            story.append(card)
            story.append(Spacer(1, 6))

        # Cache EQ dimension scores for visualization
        self._cached_eq_dimensions = {dim: eq_scores.get(dim, '') for dim in dimensions}
        self._cached_eq_overall = overall_eq

    # ── R. Ethics & Compliance Audit ──
    def _render_ethics_compliance(self, story, content: str):
        """Render ethics/compliance audit with risk indicators."""
        content = self._convert_timestamps_to_mmss(content)
        subsections = {}
        current_key = None
        current_lines = []

        for line in content.split('\n'):
            stripped = line.strip()
            if not stripped:
                continue
            header_match = re.match(
                r'^(PRESSURE TACTICS|OVER-PROMISING|REGULATORY|TRANSPARENCY|OVERALL ETHICS RISK|'
                r'ETHICS RECOMMENDATION).*:', stripped, re.IGNORECASE)
            if header_match:
                if current_key and current_lines:
                    subsections[current_key] = '\n'.join(current_lines)
                current_key = header_match.group(1).upper()
                remainder = stripped[header_match.end():].strip()
                current_lines = [remainder] if remainder else []
                continue
            if current_key:
                current_lines.append(stripped)

        if current_key and current_lines:
            subsections[current_key] = '\n'.join(current_lines)

        # Overall risk badge at top
        risk_content = subsections.get('OVERALL ETHICS RISK', '')
        if risk_content:
            risk_lower = risk_content.lower()
            if 'high' in risk_lower:
                r_bg, r_tc = '#FFEBEE', '#C62828'
            elif 'medium' in risk_lower:
                r_bg, r_tc = '#FFF8E1', '#F57F17'
            elif 'low' in risk_lower:
                r_bg, r_tc = '#E8F5E9', '#2E7D32'
            else:
                r_bg, r_tc = '#E3F2FD', '#1565C0'

            badge = Table([
                [Paragraph(
                    f"<b>Overall Ethics Risk: {risk_content.split(chr(10))[0].strip()}</b>",
                    ParagraphStyle('er', parent=self.ps['body_bold'], fontSize=12,
                                   textColor=colors.HexColor(r_tc), alignment=TA_CENTER))],
            ], colWidths=[6.2 * inch])
            badge.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(r_bg)),
                ('PADDING', (0, 0), (-1, -1), 12),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ]))
            story.append(badge)
            story.append(Spacer(1, 10))

        # Cache ethics data for visualization
        self._cached_ethics_subsections = subsections

        # Sub-audit sections
        audit_sections = [
            ('PRESSURE TACTICS', 'Pressure Tactics Assessment', '#FFEBEE', '#C62828'),
            ('OVER-PROMISING', 'Over-Promising Check', '#FFF8E1', '#F57F17'),
            ('REGULATORY', 'Regulatory Language Review', '#E3F2FD', '#1565C0'),
            ('TRANSPARENCY', 'Transparency Assessment', '#E8F5E9', '#2E7D32'),
        ]
        for key, display, bg, tc in audit_sections:
            sub_content = subsections.get(key, '')
            if not sub_content:
                continue
            inner = [Paragraph(f"<b>{display}</b>",
                     ParagraphStyle('ea_' + key, parent=self.ps['subsec'],
                                    textColor=colors.HexColor(tc)))]
            for al in sub_content.split('\n'):
                al = al.strip()
                if not al:
                    continue
                al = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', al)
                al = re.sub(r'\[(\d+[\.\d]*s?\s*-\s*\d+[\.\d]*s?)\]',
                            r"<font color='#1565C0'><b>[\1]</b></font>", al)
                for rl in ['High', 'Medium', 'Low', 'None']:
                    if f'Risk Level: {rl}' in al:
                        rl_color = ('#C62828' if rl == 'High' else '#F57F17'
                                    if rl == 'Medium' else '#2E7D32')
                        al = al.replace(
                            f'Risk Level: {rl}',
                            f"<b>Risk Level: <font color='{rl_color}'>{rl}</font></b>")
                if al.startswith('-') or al.startswith('*'):
                    inner.append(Paragraph(f"<bullet>&bull;</bullet>{al.lstrip('-* ')}",
                                           self.ps['bullet']))
                else:
                    inner.append(Paragraph(al, self.ps['body']))
                inner.append(Spacer(1, 2))

            card = Table([[inner]], colWidths=[6.2 * inch])
            card.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(bg)),
                ('PADDING', (0, 0), (-1, -1), 8),
            ]))
            story.append(card)
            story.append(Spacer(1, 6))

        # Ethics Recommendations
        rec_content = subsections.get('ETHICS RECOMMENDATION', '')
        if rec_content:
            story.append(Paragraph("<b>Ethics Recommendations</b>", self.ps['subsec']))
            for rl in rec_content.split('\n'):
                rl = rl.strip()
                if not rl:
                    continue
                rl = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', rl)
                if rl.startswith('-') or rl.startswith('*'):
                    story.append(Paragraph(f"<bullet>&bull;</bullet>{rl.lstrip('-* ')}",
                                           self.ps['bullet_success']))
                else:
                    story.append(Paragraph(rl, self.ps['body']))
                story.append(Spacer(1, 2))

    # ── S. Agent Self-Awareness Assessment ──
    def _render_agent_self_awareness(self, story, content: str):
        """Render agent self-awareness evaluation with categorized findings."""
        content = self._convert_timestamps_to_mmss(content)
        subsections = {}
        current_key = None
        current_lines = []
        sa_score = ""
        ai_conf = ""

        for line in content.split('\n'):
            stripped = line.strip()
            if not stripped:
                continue
            header_match = re.match(
                r'^(SIGNAL RECOGNITION|REAL-TIME ADAPTATION|SELF-CORRECTION|BLIND SPOTS).*:',
                stripped, re.IGNORECASE)
            if header_match:
                if current_key and current_lines:
                    subsections[current_key] = '\n'.join(current_lines)
                current_key = header_match.group(1).upper()
                remainder = stripped[header_match.end():].strip()
                current_lines = [remainder] if remainder else []
                continue
            if 'SELF-AWARENESS SCORE' in stripped.upper():
                if current_key and current_lines:
                    subsections[current_key] = '\n'.join(current_lines)
                    current_key = None
                    current_lines = []
                val = stripped.split(':', 1)
                if len(val) > 1:
                    sa_score = val[1].strip()
                continue
            if 'AI CONFIDENCE' in stripped.upper() and not current_key:
                val = stripped.split(':', 1)
                if len(val) > 1:
                    ai_conf = val[1].strip()
                continue
            if current_key:
                current_lines.append(stripped)

        if current_key and current_lines:
            subsections[current_key] = '\n'.join(current_lines)

        # Score hero — fixed layout to prevent text overlapping
        if sa_score:
            score_match = re.search(r'(\d+)\s*/\s*(\d+)', sa_score)
            if score_match:
                s_val = int(score_match.group(1))
                pct = s_val / 10
                color = '#43A047' if pct >= 0.75 else '#FF8F00' if pct >= 0.50 else '#E53935'
                hero = Table([
                    [Paragraph(f"<font color='{color}'><b>{sa_score}</b></font>",
                     ParagraphStyle('sa_hero', parent=self.ps['score_big'], fontSize=28, leading=34))],
                    [Spacer(1, 1)],  # spacer row to prevent overlap
                    [Paragraph("SELF-AWARENESS SCORE", self.ps['score_label'])],
                    [Paragraph(f"<i>{ai_conf}</i>" if ai_conf else "",
                     ParagraphStyle('sa_conf', parent=self.ps['conf'], fontSize=7))],
                ], colWidths=[6.2 * inch], rowHeights=[40, 6, 16, 14])
                hero.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F5F7FA')),
                    ('LINEBELOW', (0, -1), (-1, -1), 1.5, colors.HexColor(color)),
                    ('PADDING', (0, 0), (-1, -1), 8),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ]))
                story.append(hero)
                story.append(Spacer(1, 10))

        # Subsection cards
        sa_display = {
            'SIGNAL RECOGNITION': (
                'Signal Recognition', '#E3F2FD', '#1565C0',
                'Did the agent recognize client hesitation, confusion, or disengagement?'),
            'REAL-TIME ADAPTATION': (
                'Real-Time Adaptation', '#E8F5E9', '#2E7D32',
                'Did the agent adjust approach mid-conversation based on client cues?'),
            'SELF-CORRECTION': (
                'Self-Correction Instances', '#FFF8E1', '#F57F17',
                'Moments where the agent caught and corrected their own approach.'),
            'BLIND SPOTS': (
                'Blind Spots', '#FFEBEE', '#C62828',
                'Recurring patterns the agent appears unaware of.'),
        }
        for key, (display, bg, tc, desc) in sa_display.items():
            sub_content = subsections.get(key, '')
            if not sub_content:
                continue
            inner = [
                Paragraph(f"<b>{display}</b>",
                         ParagraphStyle('sa_' + key, parent=self.ps['subsec'],
                                        textColor=colors.HexColor(tc))),
                Paragraph(f"<i>{desc}</i>",
                         ParagraphStyle('sa_d_' + key, parent=self.ps['body'], fontSize=7.5,
                                        textColor=self.C['light'])),
                Spacer(1, 4),
            ]
            for sl in sub_content.split('\n'):
                sl = sl.strip()
                if not sl:
                    continue
                sl = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', sl)
                sl = re.sub(r'\[(\d+[\.\d]*s?\s*-\s*\d+[\.\d]*s?)\]',
                            r"<font color='#1565C0'><b>[\1]</b></font>", sl)
                if sl.startswith('-') or sl.startswith('*'):
                    inner.append(Paragraph(f"<bullet>&bull;</bullet>{sl.lstrip('-* ')}",
                                           self.ps['bullet']))
                elif re.match(r'^(Improvement|Recommendation|Action):', sl, re.IGNORECASE):
                    inner.append(Paragraph(f"<font color='#2E7D32'><b>{sl}</b></font>", self.ps['body_bold']))
                else:
                    inner.append(Paragraph(sl, self.ps['body']))
                inner.append(Spacer(1, 2))

            card = Table([[inner]], colWidths=[6.2 * inch])
            card.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(bg)),
                ('PADDING', (0, 0), (-1, -1), 8),
                ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor(tc)),
            ]))
            story.append(card)
            story.append(Spacer(1, 6))

    # ── T. 7-Day Intensive Coaching Plan (merged day-by-day with drills) ──
    def _render_7day_coaching_plan(self, story, content: str):
        """Render the comprehensive 7-day coaching plan with drills and exercises."""
        content = self._convert_timestamps_to_mmss(content)
        story.append(Paragraph(
            "<b>Complete 7-Day Coaching Sprint — One Skill Gap Per Day</b>",
            ParagraphStyle('plan_hdr', parent=self.ps['body_bold'],
                           textColor=self.C['danger'], fontSize=10)))
        story.append(Spacer(1, 6))

        rows = [[
            Paragraph("<b>Day</b>", self.ps['th']),
            Paragraph("<b>Focus Area</b>", self.ps['th']),
            Paragraph("<b>Drill / Exercise</b>", self.ps['th']),
            Paragraph("<b>Success Metric</b>", self.ps['th']),
        ]]
        # Parse "Day X: Focus -- Drill -- Metric" format
        for line in content.split('\n'):
            line = line.strip()
            m = re.match(r'Day\s*(\d+)\s*:\s*(.*?)(?:\s*--\s*)(.*?)(?:\s*--\s*)(.*)', line, re.IGNORECASE)
            if m:
                day, focus, drill, metric = m.group(1), m.group(2).strip(), m.group(3).strip(), m.group(4).strip()
                rows.append([
                    Paragraph(f"<b>{day}</b>", self.ps['td_c']),
                    Paragraph(focus, self.ps['td']),
                    Paragraph(drill, self.ps['td']),
                    Paragraph(metric, self.ps['td']),
                ])
            else:
                # Fallback: try "Day X: Focus -- Drill" (no metric)
                m2 = re.match(r'Day\s*(\d+)\s*:\s*(.*?)(?:\s*--\s*|-\s*)(.*)', line, re.IGNORECASE)
                if m2:
                    day, focus, drill = m2.group(1), m2.group(2).strip(), m2.group(3).strip()
                    rows.append([
                        Paragraph(f"<b>{day}</b>", self.ps['td_c']),
                        Paragraph(focus, self.ps['td']),
                        Paragraph(drill, self.ps['td']),
                        Paragraph("", self.ps['td']),
                    ])

        if len(rows) > 1:
            t = Table(rows, colWidths=[0.4 * inch, 1.6 * inch, 2.2 * inch, 2.0 * inch])
            style_cmds = [
                ('BACKGROUND', (0, 0), (-1, 0), self.C['primary_dark']),
                ('TEXTCOLOR', (0, 0), (-1, 0), self.C['white']),
                ('GRID', (0, 0), (-1, -1), 0.5, self.C['border']),
                ('PADDING', (0, 0), (-1, -1), 6),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]
            for i in range(1, len(rows)):
                if i % 2 == 0:
                    style_cmds.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor("#F5F7FA")))
            t.setStyle(TableStyle(style_cmds))
            story.append(t)
        else:
            # Fallback: render as numbered list if table parsing fails
            for line in content.split('\n'):
                line = line.strip()
                if not line:
                    continue
                m = re.match(r'^(\d+)\.\s*(.*)', line)
                if m:
                    num, text = m.group(1), m.group(2)
                    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
                    t = Table([[
                        Paragraph(f"<b>{num}</b>", ParagraphStyle('fn', parent=self.ps['td_c'],
                                  fontSize=14, textColor=self.C['white'])),
                        Paragraph(text, self.ps['body'])
                    ]], colWidths=[0.4 * inch, 5.8 * inch])
                    t.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (0, 0), self.C['primary_dark']),
                        ('BACKGROUND', (1, 0), (1, 0), colors.HexColor("#F5F7FA")),
                        ('BOX', (0, 0), (-1, -1), 0.5, self.C['border']),
                        ('PADDING', (0, 0), (-1, -1), 8),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ]))
                    story.append(t)
                    story.append(Spacer(1, 4))
                else:
                    line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
                    if line.startswith('-') or line.startswith('*'):
                        story.append(Paragraph(f"<bullet>&bull;</bullet>{line.lstrip('-* ')}", self.ps['bullet']))
                    else:
                        story.append(Paragraph(line, self.ps['body']))
                    story.append(Spacer(1, 3))
        story.append(Spacer(1, 8))

    # ── U. Client Engagement & Tone Analysis (NEW §17) ──
    def _render_tone_analysis(self, story, content: str):
        """Render tone, energy, and client engagement quality analysis."""
        content = self._convert_timestamps_to_mmss(content)
        subsections = {}
        current_key = None
        current_lines = []
        tone_scores = {}
        overall_score = ""
        ai_conf = ""

        for line in content.split('\n'):
            stripped = line.strip()
            if not stripped:
                continue
            header_match = re.match(
                r'^(TONE CONSISTENCY|ENERGY|ENTHUSIASM|CLIENT ENGAGEMENT|RAPPORT.BUILDING|'
                r'OVERALL TONE).*:', stripped, re.IGNORECASE)
            if header_match:
                if current_key and current_lines:
                    subsections[current_key] = '\n'.join(current_lines)
                current_key = header_match.group(1).upper().replace('&', 'AND')
                remainder = stripped[header_match.end():].strip()
                current_lines = [remainder] if remainder else []
                continue
            # Capture scores
            score_match = re.match(r'Score:\s*(\d+)\s*/\s*(\d+)', stripped, re.IGNORECASE)
            if score_match and current_key:
                tone_scores[current_key] = int(score_match.group(1))
                current_lines.append(stripped)
                continue
            if 'OVERALL TONE' in stripped.upper() and 'SCORE' in stripped.upper():
                val = stripped.split(':', 1)
                if len(val) > 1:
                    overall_score = val[1].strip()
                    sc_match = re.search(r'(\d+)\s*/\s*(\d+)', overall_score)
                    if sc_match:
                        tone_scores['OVERALL'] = int(sc_match.group(1))
                continue
            if 'AI CONFIDENCE' in stripped.upper() and not current_key:
                val = stripped.split(':', 1)
                if len(val) > 1:
                    ai_conf = val[1].strip()
                continue
            if current_key:
                current_lines.append(stripped)

        if current_key and current_lines:
            subsections[current_key] = '\n'.join(current_lines)

        # Cache tone scores for visualization
        self._cached_tone_scores = tone_scores

        # Score hero
        if overall_score:
            score_match = re.search(r'(\d+)\s*/\s*(\d+)', overall_score)
            if score_match:
                s_val = int(score_match.group(1))
                pct = s_val / 10
                color = '#43A047' if pct >= 0.75 else '#FF8F00' if pct >= 0.50 else '#E53935'
                hero = Table([
                    [Paragraph(f"<font color='{color}'><b>{overall_score}</b></font>",
                     ParagraphStyle('tone_hero', parent=self.ps['score_big'], fontSize=28, leading=34))],
                    [Spacer(1, 1)],
                    [Paragraph("OVERALL TONE & ENGAGEMENT SCORE", self.ps['score_label'])],
                    [Paragraph(f"<i>{ai_conf}</i>" if ai_conf else "",
                     ParagraphStyle('tone_conf', parent=self.ps['conf'], fontSize=7))],
                ], colWidths=[6.2 * inch], rowHeights=[40, 6, 16, 14])
                hero.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F5F7FA')),
                    ('LINEBELOW', (0, -1), (-1, -1), 1.5, colors.HexColor(color)),
                    ('PADDING', (0, 0), (-1, -1), 8),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ]))
                story.append(hero)
                story.append(Spacer(1, 10))

        # Subsection cards
        tone_display = {
            'TONE CONSISTENCY': ('Tone Consistency', '#E3F2FD', '#1565C0',
                                 'Was the agent professional, warm, authoritative, or inconsistent?'),
            'ENERGY': ('Energy & Enthusiasm', '#E8F5E9', '#2E7D32',
                       'Did the agent maintain appropriate energy throughout?'),
            'ENTHUSIASM': ('Energy & Enthusiasm', '#E8F5E9', '#2E7D32',
                           'Did the agent maintain appropriate energy throughout?'),
            'CLIENT ENGAGEMENT': ('Client Engagement Quality', '#FFF8E1', '#F57F17',
                                  'How effectively did the agent keep the client involved?'),
            'RAPPORT-BUILDING': ('Rapport-Building Language', '#F3E5F5', '#7B1FA2',
                                 'Specific phrases and techniques that built connection.'),
        }
        for key, (display, bg, tc, desc) in tone_display.items():
            sub_content = subsections.get(key, '')
            if not sub_content:
                continue
            inner = [
                Paragraph(f"<b>{display}</b>",
                         ParagraphStyle('tn_' + key[:8], parent=self.ps['subsec'],
                                        textColor=colors.HexColor(tc))),
                Paragraph(f"<i>{desc}</i>",
                         ParagraphStyle('tn_d_' + key[:8], parent=self.ps['body'], fontSize=7.5,
                                        textColor=self.C['light'])),
                Spacer(1, 4),
            ]
            for sl in sub_content.split('\n'):
                sl = sl.strip()
                if not sl:
                    continue
                sl = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', sl)
                sl = re.sub(r'\[(\d+[\.\d]*s?\s*-\s*\d+[\.\d]*s?)\]',
                            r"<font color='#1565C0'><b>[\1]</b></font>", sl)
                if sl.startswith('-') or sl.startswith('*'):
                    inner.append(Paragraph(f"<bullet>&bull;</bullet>{sl.lstrip('-* ')}",
                                           self.ps['bullet']))
                elif re.match(r'^Score:', sl, re.IGNORECASE):
                    inner.append(Paragraph(f"<font color='{tc}'><b>{sl}</b></font>", self.ps['body_bold']))
                else:
                    inner.append(Paragraph(sl, self.ps['body']))
                inner.append(Spacer(1, 2))

            card = Table([[inner]], colWidths=[6.2 * inch])
            card.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(bg)),
                ('PADDING', (0, 0), (-1, -1), 8),
                ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor(tc)),
            ]))
            story.append(card)
            story.append(Spacer(1, 6))

    # ── V. Negotiation & Persuasion Proficiency (NEW §18) ──
    def _render_negotiation_proficiency(self, story, content: str):
        """Render negotiation tactics, persuasion techniques, and deal-advancing analysis."""
        content = self._convert_timestamps_to_mmss(content)
        subsections = {}
        current_key = None
        current_lines = []
        neg_scores = {}
        overall_score = ""
        ai_conf = ""

        for line in content.split('\n'):
            stripped = line.strip()
            if not stripped:
                continue
            header_match = re.match(
                r'^(PERSUASION TECHNIQUES|NEGOTIATION POSITIONING|OBJECTION.TO.COMMITMENT|'
                r'VALUE FRAMING|CLOSING SIGNALS|OVERALL NEGOTIATION).*:', stripped, re.IGNORECASE)
            if header_match:
                if current_key and current_lines:
                    subsections[current_key] = '\n'.join(current_lines)
                current_key = header_match.group(1).upper().replace('-', '_')
                remainder = stripped[header_match.end():].strip()
                current_lines = [remainder] if remainder else []
                continue
            score_match = re.match(r'Score:\s*(\d+)\s*/\s*(\d+)', stripped, re.IGNORECASE)
            if score_match and current_key:
                neg_scores[current_key] = int(score_match.group(1))
                current_lines.append(stripped)
                continue
            if 'OVERALL NEGOTIATION' in stripped.upper() and 'SCORE' in stripped.upper():
                val = stripped.split(':', 1)
                if len(val) > 1:
                    overall_score = val[1].strip()
                    sc_match = re.search(r'(\d+)\s*/\s*(\d+)', overall_score)
                    if sc_match:
                        neg_scores['OVERALL'] = int(sc_match.group(1))
                continue
            if 'AI CONFIDENCE' in stripped.upper() and not current_key:
                val = stripped.split(':', 1)
                if len(val) > 1:
                    ai_conf = val[1].strip()
                continue
            if current_key:
                current_lines.append(stripped)

        if current_key and current_lines:
            subsections[current_key] = '\n'.join(current_lines)

        # Cache negotiation scores for visualization
        self._cached_negotiation_scores = neg_scores

        # Score hero
        if overall_score:
            score_match = re.search(r'(\d+)\s*/\s*(\d+)', overall_score)
            if score_match:
                s_val = int(score_match.group(1))
                pct = s_val / 10
                color = '#43A047' if pct >= 0.75 else '#FF8F00' if pct >= 0.50 else '#E53935'
                hero = Table([
                    [Paragraph(f"<font color='{color}'><b>{overall_score}</b></font>",
                     ParagraphStyle('neg_hero', parent=self.ps['score_big'], fontSize=28, leading=34))],
                    [Spacer(1, 1)],
                    [Paragraph("OVERALL NEGOTIATION & PERSUASION SCORE", self.ps['score_label'])],
                    [Paragraph(f"<i>{ai_conf}</i>" if ai_conf else "",
                     ParagraphStyle('neg_conf', parent=self.ps['conf'], fontSize=7))],
                ], colWidths=[6.2 * inch], rowHeights=[40, 6, 16, 14])
                hero.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F5F7FA')),
                    ('LINEBELOW', (0, -1), (-1, -1), 1.5, colors.HexColor(color)),
                    ('PADDING', (0, 0), (-1, -1), 8),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ]))
                story.append(hero)
                story.append(Spacer(1, 10))

        # Subsection cards
        neg_display = {
            'PERSUASION TECHNIQUES': ('Persuasion Techniques Used', '#E3F2FD', '#1565C0',
                                      'Social proof, scarcity, authority, reciprocity, anchoring, framing.'),
            'NEGOTIATION POSITIONING': ('Negotiation Positioning', '#E8F5E9', '#2E7D32',
                                        'Did the agent establish strong positioning or give away leverage?'),
            'OBJECTION_TO_COMMITMENT': ('Objection-to-Commitment Conversion', '#FFF8E1', '#F57F17',
                                        'Tracking each objection and whether it was converted toward commitment.'),
            'VALUE FRAMING': ('Value Framing Ability', '#F3E5F5', '#7B1FA2',
                              'How well did the agent frame value relative to price/competition?'),
            'CLOSING SIGNALS': ('Closing Signals & Trial Closes', '#FFEBEE', '#C62828',
                                'Did the agent use trial closes or test for commitment?'),
        }
        for key, (display, bg, tc, desc) in neg_display.items():
            sub_content = subsections.get(key, '')
            if not sub_content:
                continue
            inner = [
                Paragraph(f"<b>{display}</b>",
                         ParagraphStyle('ng_' + key[:8], parent=self.ps['subsec'],
                                        textColor=colors.HexColor(tc))),
                Paragraph(f"<i>{desc}</i>",
                         ParagraphStyle('ng_d_' + key[:8], parent=self.ps['body'], fontSize=7.5,
                                        textColor=self.C['light'])),
                Spacer(1, 4),
            ]
            for sl in sub_content.split('\n'):
                sl = sl.strip()
                if not sl:
                    continue
                sl = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', sl)
                sl = re.sub(r'\[(\d+[\.\d]*s?\s*-\s*\d+[\.\d]*s?)\]',
                            r"<font color='#1565C0'><b>[\1]</b></font>", sl)
                # Highlight effectiveness ratings
                for rating in ['Effective', 'Partially Effective', 'Ineffective',
                               'Resolved', 'Partially Resolved', 'Unresolved']:
                    if rating in sl:
                        r_color = '#43A047' if rating in ['Effective', 'Resolved'] else \
                                  '#FF8F00' if 'Partial' in rating else '#E53935'
                        sl = sl.replace(rating, f"<font color='{r_color}'><b>{rating}</b></font>")
                if sl.startswith('-') or sl.startswith('*'):
                    inner.append(Paragraph(f"<bullet>&bull;</bullet>{sl.lstrip('-* ')}",
                                           self.ps['bullet']))
                elif re.match(r'^Score:', sl, re.IGNORECASE):
                    inner.append(Paragraph(f"<font color='{tc}'><b>{sl}</b></font>", self.ps['body_bold']))
                elif any(marker in sl.upper() for marker in ['OBJECTION', '→', 'OUTCOME']):
                    inner.append(Paragraph(f"<b>{sl}</b>", self.ps['body_bold']))
                else:
                    inner.append(Paragraph(sl, self.ps['body']))
                inner.append(Spacer(1, 2))

            card = Table([[inner]], colWidths=[6.2 * inch])
            card.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(bg)),
                ('PADDING', (0, 0), (-1, -1), 8),
                ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor(tc)),
            ]))
            story.append(card)
            story.append(Spacer(1, 6))

    # ════════════════════════════════════════════════════════
    # VISUALIZATION SECTION — ALL CHARTS AT THE END
    # ════════════════════════════════════════════════════════

    def _build_visualization_section(self, story, sections: Dict[str, str]):
        """Comprehensive visual analytics dashboard with 16+ chart types."""
        scores = getattr(self, '_cached_scores', {})
        deal_metrics = getattr(self, '_cached_deal_metrics', [])
        talk_ratio = getattr(self, '_cached_talk_ratio', {})
        sentiment_phases = getattr(self, '_cached_sentiment_phases', [])
        discovery_pillars = getattr(self, '_cached_discovery_pillars', [])
        ethics_subs = getattr(self, '_cached_ethics_subsections', {})
        listening_subs = getattr(self, '_cached_listening_subsections', {})
        eq_dims = getattr(self, '_cached_eq_dimensions', {})

        story.append(SectionDivider(width=460, color="#0D47A1", thickness=2))
        story.append(Spacer(1, 2))
        story.append(Paragraph("VISUAL PERFORMANCE ANALYTICS DASHBOARD", self.ps['sec']))
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            "The following visualizations transform the evaluation data into actionable graphical insights. "
            "Each chart reveals a different dimension of the agent's performance.",
            self.ps['body']))
        story.append(Spacer(1, 10))

        # ═══════════════════════════════════════════
        # 1. CONVERSATION FLOW & CONTROL
        # ═══════════════════════════════════════════
        story.append(Paragraph("<b>1. Conversation Flow & Control</b>", self.ps['subsec']))
        story.append(Spacer(1, 6))

        # ── Talk-Time Distribution (Donut) — only from LLM-evaluated data ──
        agent_pct = talk_ratio.get('agent')
        client_pct = talk_ratio.get('client')
        if agent_pct is not None and client_pct is not None:
            story.append(Paragraph("<b>Talk-Time Distribution (Agent vs Client)</b>",
                         ParagraphStyle('v_a', parent=self.ps['body_bold'], fontSize=9)))
            story.append(Paragraph(
                "Ideal zone: 45-55% agent talk. Too much agent talk = pushy. Too little = weak guidance.",
                ParagraphStyle('v_a_d', parent=self.ps['body'], fontSize=7.5, textColor=self.C['medium'])))
            story.append(Spacer(1, 4))

            donut = DonutChartFlowable([
                ("Agent", agent_pct, "#1565C0"),
                ("Client", client_pct, "#43A047"),
            ], width=180, height=180)
            donut_table = Table([[donut]], colWidths=[6.2 * inch])
            donut_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F5F7FA")),
                ('PADDING', (0, 0), (-1, -1), 10),
            ]))
            story.append(donut_table)

            # Ideal zone indicator
            if 45 <= agent_pct <= 55:
                zone_text = "Agent is in the IDEAL talk-time zone."
                zone_color = "#2E7D32"
            elif agent_pct > 65:
                zone_text = "Agent-dominated conversation — reduce talking, ask more questions."
                zone_color = "#C62828"
            elif agent_pct < 40:
                zone_text = "Agent too passive — needs to guide the conversation more firmly."
                zone_color = "#F57F17"
            else:
                zone_text = "Slightly outside ideal range — minor adjustment needed."
                zone_color = "#F57F17"
            story.append(Paragraph(f"<font color='{zone_color}'><b>{zone_text}</b></font>",
                         ParagraphStyle('zone', parent=self.ps['body'], fontSize=8, alignment=TA_CENTER)))
            story.append(Spacer(1, 12))

            # ── Log to JSON ──
            self.viz_logger.log_chart(
                chart_id="talk_time_distribution",
                title="Talk-Time Distribution (Agent vs Client)",
                chart_type="doughnut",
                labels=["Agent", "Client"],
                datasets=[{"label": "Talk Time %", "values": [agent_pct, client_pct],
                           "colors": ["#1565C0", "#43A047"]}],
                description="Ideal zone: 45-55% agent talk. Shows conversation dominance balance.",
                metadata={"ideal_range": [45, 55], "zone_verdict": zone_text},
            )
        else:
            story.append(Paragraph(
                "<i>Talk-time ratio data not available from LLM analysis. Donut chart omitted.</i>",
                ParagraphStyle('no_donut', parent=self.ps['body'], fontSize=8,
                               textColor=self.C['light'], alignment=TA_CENTER)))
            story.append(Spacer(1, 12))
            # Set defaults for later dashboard use
            agent_pct = 0
            client_pct = 0

        # ═══════════════════════════════════════════
        # 2. CLIENT SENTIMENT & EMOTION
        # ═══════════════════════════════════════════
        story.append(CondPageBreak(2.5 * inch))
        story.append(Paragraph("<b>2. Client Sentiment & Emotion</b>", self.ps['subsec']))
        story.append(Spacer(1, 6))

        # ── Client Sentiment Over Time (Line Chart) ──
        story.append(Paragraph("<b>Client Sentiment Over Time</b>",
                     ParagraphStyle('v_c', parent=self.ps['body_bold'], fontSize=9)))
        story.append(Paragraph(
            "Shows where trust increased or dropped. Connects agent behavior to emotional outcome.",
            ParagraphStyle('v_c_d', parent=self.ps['body'], fontSize=7.5, textColor=self.C['medium'])))
        story.append(Spacer(1, 4))

        # Derive sentiment scores from phases
        phase_sentiment_map = {'EARLY': 5.0, 'MID': 6.0, 'LATE': 7.0}
        sentiment_data = []
        sentiment_labels = []
        if sentiment_phases:
            for phase_name, phase_content in sentiment_phases:
                # Try to extract numeric sentiment from content
                sent_match = re.search(r'(\d+)\s*/\s*10', phase_content)
                if sent_match:
                    sentiment_data.append(int(sent_match.group(1)))
                else:
                    # Derive from keywords
                    content_lower = phase_content.lower()
                    if any(w in content_lower for w in ['skeptic', 'hesitant', 'anxious', 'distrust']):
                        sentiment_data.append(3)
                    elif any(w in content_lower for w in ['neutral', 'cautious', 'guarded']):
                        sentiment_data.append(5)
                    elif any(w in content_lower for w in ['engaged', 'interested', 'warming']):
                        sentiment_data.append(7)
                    elif any(w in content_lower for w in ['confident', 'trust', 'committed', 'positive']):
                        sentiment_data.append(8)
                    else:
                        sentiment_data.append(phase_sentiment_map.get(phase_name, 5))
                sentiment_labels.append(f"{phase_name} Phase")

        if not sentiment_data or len(sentiment_data) < 2:
            # No real sentiment data extracted — skip this chart entirely
            story.append(Paragraph(
                "<i>Insufficient sentiment data extracted from transcript to generate this chart.</i>",
                ParagraphStyle('no_data', parent=self.ps['body'], fontSize=8,
                               textColor=self.C['light'], alignment=TA_CENTER)))
            story.append(Spacer(1, 10))
        else:
            line_chart = LineChartFlowable(
                sentiment_data, width=420, height=160,
                y_label="Sentiment", x_labels=sentiment_labels,
                line_color="#1565C0", fill_color="#E3F2FD",
                y_range=(0, 10))
            lc_table = Table([[line_chart]], colWidths=[6.2 * inch])
            lc_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F5F7FA")),
                ('PADDING', (0, 0), (-1, -1), 10),
            ]))
            story.append(lc_table)
            story.append(Spacer(1, 10))

            # ── Log to JSON ──
            self.viz_logger.log_chart(
                chart_id="client_sentiment_timeline",
                title="Client Sentiment Over Time",
                chart_type="line",
                labels=sentiment_labels,
                datasets=[{"label": "Sentiment Score", "values": sentiment_data,
                           "colors": ["#1565C0"], "fill": True, "fillColor": "#E3F2FD"}],
                description="Shows where trust increased or dropped across meeting phases.",
                options={"y_range": [0, 10], "y_label": "Sentiment"},
            )

        # ── Emotion Heatmap — ONLY shown if we have real sentiment phase data ──
        # The heatmap requires actual per-phase sentiment data; we do NOT fabricate values
        # This section is intentionally omitted because emotion heatmap data (Trust/Hesitation/Interest/Anxiety
        # per stage) cannot be reliably extracted from the transcript without explicit LLM scoring per emotion.
        # Showing fabricated mathematical estimates would violate the "no fake content" principle.
        story.append(Spacer(1, 4))

        # ═══════════════════════════════════════════
        # 3. LISTENING & DISCOVERY INTELLIGENCE
        # ═══════════════════════════════════════════
        story.append(CondPageBreak(2.5 * inch))
        story.append(Paragraph("<b>3. Listening & Discovery Intelligence</b>", self.ps['subsec']))
        story.append(Spacer(1, 6))

        # ── Listening vs Talking Radar ──
        story.append(Paragraph("<b>Listening vs Talking Radar</b>",
                     ParagraphStyle('v_e', parent=self.ps['body_bold'], fontSize=9)))
        story.append(Paragraph(
            "Managers instantly see if the agent is a listener or talker across 5 dimensions.",
            ParagraphStyle('v_e_d', parent=self.ps['body'], fontSize=7.5, textColor=self.C['medium'])))
        story.append(Spacer(1, 4))

        # Build listening radar from cached data — ONLY include dimensions with real scores
        listening_scores = {}
        # Talk Balance is calculable from actual talk ratio data
        if talk_ratio.get('agent') is not None:
            listening_scores['Talk Balance'] = max(1, 10 - abs(agent_pct - 50) / 5)
        # Parse scores from listening subsections — skip if no score found
        for key, display_name in [('INTERRUPTION ANALYSIS', 'Interruption Analysis'),
                                   ('REFLECTIVE LISTENING', 'Reflective Listening'),
                                   ('FOLLOW-UP QUESTION', 'Follow-Up Question'),
                                   ('ACTIVE LISTENING GAPS', 'Active Listening Gaps')]:
            sub = listening_subs.get(key, '')
            s_match = re.search(r'(\d+)\s*/\s*10', sub)
            if s_match:
                listening_scores[display_name] = int(s_match.group(1))
            # If no score found, we do NOT add a default — only real data

        if len(listening_scores) >= 3:
            listen_radar = RadarChartFlowable(listening_scores, width=280, height=280)
            lr_table = Table([[listen_radar]], colWidths=[6.2 * inch])
            lr_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F5F7FA")),
                ('PADDING', (0, 0), (-1, -1), 10),
            ]))
            story.append(lr_table)
            story.append(Spacer(1, 10))

            # ── Log to JSON ──
            self.viz_logger.log_chart(
                chart_id="listening_radar",
                title="Listening vs Talking Radar",
                chart_type="radar",
                labels=list(listening_scores.keys()),
                datasets=[{"label": "Listening Score", "values": list(listening_scores.values()),
                           "colors": ["#1565C0"], "max": 10}],
                description="Multi-dimensional listening assessment across 5 dimensions.",
                options={"max_value": 10},
            )
        else:
            story.append(Paragraph(
                "<i>Insufficient listening dimension scores extracted to generate radar chart (need at least 3 scored dimensions).</i>",
                ParagraphStyle('no_listen', parent=self.ps['body'], fontSize=8,
                               textColor=self.C['light'], alignment=TA_CENTER)))
            story.append(Spacer(1, 10))

        # ── Discovery Completeness Matrix ──
        story.append(CondPageBreak(2.0 * inch))
        story.append(Paragraph("<b>Discovery Completeness Matrix</b>",
                     ParagraphStyle('v_f', parent=self.ps['body_bold'], fontSize=9)))
        story.append(Paragraph(
            "Objective diagnostic: which pillars were covered, partially explored, or completely missed.",
            ParagraphStyle('v_f_d', parent=self.ps['body'], fontSize=7.5, textColor=self.C['medium'])))
        story.append(Spacer(1, 4))

        if discovery_pillars:
            disc_bar_data = []
            for pillar, status, evidence, confidence in discovery_pillars:
                status_lower = status.lower()
                if 'confirmed' in status_lower:
                    val, col = 9, "#43A047"
                elif 'partially' in status_lower:
                    val, col = 5, "#FF8F00"
                else:
                    val, col = 2, "#E53935"
                disc_bar_data.append((pillar, val, col))
            disc_chart = HBarChartFlowable(disc_bar_data, width=440, bar_height=20, max_val=10)
            dc_table = Table([[disc_chart]], colWidths=[6.2 * inch])
            dc_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F5F7FA")),
                ('PADDING', (0, 0), (-1, -1), 8),
            ]))
            story.append(dc_table)
            story.append(Spacer(1, 12))

            # ── Log to JSON ──
            self.viz_logger.log_chart(
                chart_id="discovery_completeness_matrix",
                title="Discovery Completeness Matrix",
                chart_type="horizontalBar",
                labels=[d[0] for d in disc_bar_data],
                datasets=[{"label": "Coverage Score", "values": [d[1] for d in disc_bar_data],
                           "colors": [d[2] for d in disc_bar_data]}],
                description="Objective diagnostic: which pillars were covered, partially explored, or completely missed.",
                options={"max_value": 10},
            )

        # ═══════════════════════════════════════════
        # 4. SALES EFFECTIVENESS
        # ═══════════════════════════════════════════
        story.append(CondPageBreak(3.5 * inch))
        story.append(Paragraph("<b>4. Sales Effectiveness</b>", self.ps['subsec']))
        story.append(Spacer(1, 6))

        # ── Deal Momentum Gauge ──
        story.append(Paragraph("<b>Deal Momentum Gauge</b>",
                     ParagraphStyle('v_h', parent=self.ps['body_bold'], fontSize=9)))
        story.append(Paragraph(
            "Cold (0-33%) / Warm (34-66%) / Hot (67-100%) — forecasts deal quality based on sentiment, "
            "commitment language, and objection handling.",
            ParagraphStyle('v_h_d', parent=self.ps['body'], fontSize=7.5, textColor=self.C['medium'])))
        story.append(Spacer(1, 4))

        # Calculate momentum from actual LLM-evaluated scores — no hardcoded fallback
        momentum_val = None
        # First, try to get deal probability from LLM deal metrics
        for dm in deal_metrics:
            if 'probability' in dm[0].lower():
                try:
                    momentum_val = int(dm[1].replace('%', '').strip())
                except (ValueError, AttributeError):
                    pass
                break
        # Fallback: derive from actual performance scores if available
        if momentum_val is None and scores:
            avg_score = sum(scores.values()) / len(scores)
            momentum_val = int(min(100, max(0, avg_score * 10)))
            if sentiment_data and len(sentiment_data) >= 2:
                final_sentiment = sentiment_data[-1]
                momentum_val = int((momentum_val + final_sentiment * 10) / 2)

        if momentum_val is None:
            story.append(Paragraph(
                "<i>Insufficient data to calculate deal momentum. No performance scores or deal probability available.</i>",
                ParagraphStyle('no_momentum', parent=self.ps['body'], fontSize=8,
                               textColor=self.C['light'], alignment=TA_CENTER)))
            story.append(Spacer(1, 12))
            momentum_val = 0  # Set for later dashboard use
        else:
            gauge = GaugeFlowable(momentum_val, max_val=100, label="Deal Momentum", width=260, height=150)
            # Gauge + Legend side by side
            momentum_zone = "HOT" if momentum_val >= 67 else "WARM" if momentum_val >= 34 else "COLD"
            zone_color = "#43A047" if momentum_val >= 67 else "#FF8F00" if momentum_val >= 34 else "#E53935"
            legend_content = [
                Paragraph(f"<b><font color='{zone_color}' size='16'>{momentum_val}%</font></b>",
                         ParagraphStyle('g_val', parent=self.ps['body'], alignment=TA_CENTER)),
                Spacer(1, 4),
                Paragraph(f"<b><font color='{zone_color}'>{momentum_zone} ZONE</font></b>",
                         ParagraphStyle('g_zone', parent=self.ps['body_bold'], fontSize=11, alignment=TA_CENTER)),
                Spacer(1, 8),
                Paragraph("<font color='#E53935'>■</font> Cold: 0-33% (Deal at risk)", self.ps['body']),
                Paragraph("<font color='#FF8F00'>■</font> Warm: 34-66% (Needs nurturing)", self.ps['body']),
                Paragraph("<font color='#43A047'>■</font> Hot: 67-100% (Strong momentum)", self.ps['body']),
            ]
            g_table = Table([[gauge, legend_content]], colWidths=[3.5 * inch, 2.7 * inch])
            g_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (0, 0), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F5F7FA")),
                ('PADDING', (0, 0), (-1, -1), 10),
                ('BOX', (0, 0), (-1, -1), 0.5, self.C['border']),
            ]))
            story.append(g_table)
            story.append(Spacer(1, 12))

            # ── Log to JSON ──
            self.viz_logger.log_chart(
                chart_id="deal_momentum_gauge",
                title="Deal Momentum Gauge",
                chart_type="gauge",
                labels=["Cold", "Warm", "Hot"],
                datasets=[{"label": "Deal Momentum", "values": [momentum_val]}],
                description="Cold (0-33%) / Warm (34-66%) / Hot (67-100%) — forecasts deal quality.",
                metadata={"zone": momentum_zone, "zone_color": zone_color,
                          "thresholds": {"cold": [0, 33], "warm": [34, 66], "hot": [67, 100]}},
                options={"max_value": 100},
            )

        # ═══════════════════════════════════════════
        # 5. STAGE-WISE PERFORMANCE BREAKDOWN
        # ═══════════════════════════════════════════
        story.append(CondPageBreak(3.0 * inch))
        story.append(Paragraph("<b>5. Stage-Wise Performance Breakdown</b>", self.ps['subsec']))
        story.append(Spacer(1, 6))

        # ── Funnel Drop-Off ──
        story.append(Paragraph("<b>Sales Funnel Drop-Off Visualization</b>",
                     ParagraphStyle('v_i', parent=self.ps['body_bold'], fontSize=9)))
        story.append(Paragraph(
            "Shows where momentum was lost across conversation stages. Each stage scored 0-10.",
            ParagraphStyle('v_i_d', parent=self.ps['body'], fontSize=7.5, textColor=self.C['medium'])))
        story.append(Spacer(1, 4))

        # Build funnel ONLY from real scores — no estimation or fabrication
        funnel_stages = []
        stage_keys = ['Rapport', 'Discovery', 'Solution Fit', 'Objection Handling', 'Closing']
        stage_score_map = {
            'Rapport': ['rapport', 'engagement', 'client engagement'],
            'Discovery': ['discovery', 'needs discovery', 'needs'],
            'Solution Fit': ['value proposition', 'solution', 'communication'],
            'Objection Handling': ['objection', 'handling'],
            'Closing': ['closing', 'deal progression', 'close'],
        }
        stage_colors = ['#43A047', '#1565C0', '#FF8F00', '#7B1FA2', '#E53935']
        for i, stage in enumerate(stage_keys):
            keywords = stage_score_map.get(stage, [stage.lower()])
            for s_label, s_val in scores.items():
                if any(kw in s_label.lower() for kw in keywords):
                    funnel_stages.append((f"{stage} ({s_val:.0f}/10)", s_val, stage_colors[i]))
                    break
            # If no matching score found for this stage, we skip it — no fake estimation

        if funnel_stages:
            funnel = FunnelFlowable(funnel_stages, width=400, height=250)
            f_table = Table([[funnel]], colWidths=[6.2 * inch])
            f_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F5F7FA")),
                ('PADDING', (0, 0), (-1, -1), 12),
                ('BOX', (0, 0), (-1, -1), 0.5, self.C['border']),
            ]))
            story.append(f_table)

            # ── Log to JSON ──
            self.viz_logger.log_chart(
                chart_id="sales_funnel_dropoff",
                title="Sales Funnel Drop-Off Visualization",
                chart_type="funnel",
                labels=[s[0] for s in funnel_stages],
                datasets=[{"label": "Stage Score", "values": [s[1] for s in funnel_stages],
                           "colors": [s[2] for s in funnel_stages]}],
                description="Shows where momentum was lost across conversation stages.",
                options={"max_value": 10},
            )
        else:
            story.append(Paragraph(
                "<i>No matching score dimensions found for funnel stages. Funnel chart omitted.</i>",
                ParagraphStyle('no_funnel', parent=self.ps['body'], fontSize=8,
                               textColor=self.C['light'], alignment=TA_CENTER)))
        story.append(Spacer(1, 10))

        # ── Stage Score Bar Chart ──
        story.append(CondPageBreak(2.0 * inch))
        story.append(Paragraph("<b>Stage Score Bar Chart</b>",
                     ParagraphStyle('v_j', parent=self.ps['body_bold'], fontSize=9)))
        story.append(Paragraph(
            "Each stage scored independently. Managers immediately know where to coach.",
            ParagraphStyle('v_j_d', parent=self.ps['body'], fontSize=7.5, textColor=self.C['medium'])))
        story.append(Spacer(1, 4))

        if scores:
            score_bar_data = []
            for label, val in scores.items():
                pct = val / 10
                col = "#43A047" if pct >= 0.75 else "#FF8F00" if pct >= 0.50 else "#E53935"
                score_bar_data.append((label, val, col))
            sb_chart = HBarChartFlowable(score_bar_data, width=440, bar_height=18, max_val=10)
            sb_table = Table([[sb_chart]], colWidths=[6.2 * inch])
            sb_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F5F7FA")),
                ('PADDING', (0, 0), (-1, -1), 8),
            ]))
            story.append(sb_table)
            story.append(Spacer(1, 12))

            # ── Log to JSON ──
            self.viz_logger.log_chart(
                chart_id="performance_score_bars",
                title="Performance Score Dashboard",
                chart_type="horizontalBar",
                labels=[d[0] for d in score_bar_data],
                datasets=[{"label": "Score", "values": [d[1] for d in score_bar_data],
                           "colors": [d[2] for d in score_bar_data]}],
                description="Each evaluated dimension scored independently on a 1-10 scale.",
                options={"max_value": 10},
            )

        # ═══════════════════════════════════════════
        # 6. BEHAVIORAL & LANGUAGE INTELLIGENCE
        # ═══════════════════════════════════════════
        story.append(CondPageBreak(2.5 * inch))
        story.append(Paragraph("<b>6. Behavioral & Language Intelligence</b>", self.ps['subsec']))
        story.append(Spacer(1, 6))

        # ── Confidence vs Uncertainty Language — derived from actual transcript ──
        # Count confident, hedging, and filler phrases from the agent's actual dialogue
        transcript_data_ref = getattr(self, '_transcript_data', None)
        if transcript_data_ref and transcript_data_ref.transcripts:
            agent_segments = [seg.transcript for seg in transcript_data_ref.transcripts
                              if seg.speaker_name and seg.speaker_name != 'Client']
        else:
            agent_segments = []
        agent_text = ' '.join(agent_segments).lower()

        confident_phrases = ['i will', 'we will', 'absolutely', 'definitely', 'i guarantee',
                             'without a doubt', 'i assure', 'i promise', 'my goal is',
                             'what i want', 'here\'s what', 'the fact is', 'i believe',
                             'i recommend', 'my recommendation', 'i\'m going to',
                             'we\'re going to', 'let me walk you through', 'let me explain']
        hedging_phrases = ['i think', 'maybe', 'perhaps', 'probably', 'might', 'could be',
                           'i guess', 'sort of', 'kind of', 'it depends', 'not sure',
                           'i don\'t know', 'possibly', 'hopefully', 'we\'ll see']
        filler_phrases = ['you know', 'i mean', 'like', 'um', 'uh', 'basically',
                          'obviously', 'literally', 'actually', 'right right right']

        confident_count = sum(agent_text.count(p) for p in confident_phrases)
        hedging_count = sum(agent_text.count(p) for p in hedging_phrases)
        filler_count = sum(agent_text.count(p) for p in filler_phrases)

        total_lang = confident_count + hedging_count + filler_count
        if total_lang > 0:
            story.append(Paragraph("<b>Confidence vs Uncertainty Language</b>",
                         ParagraphStyle('v_k', parent=self.ps['body_bold'], fontSize=9)))
            story.append(Paragraph(
                "Counts of confident vs hedging phrases from the agent's actual dialogue in this transcript.",
                ParagraphStyle('v_k_d', parent=self.ps['body'], fontSize=7.5, textColor=self.C['medium'])))
            story.append(Spacer(1, 4))

            conf_data = [
                ("Language Type", [
                    ("Confident", confident_count, "#43A047"),
                    ("Hedging", hedging_count, "#FF8F00"),
                    ("Filler Words", filler_count, "#E53935"),
                ]),
            ]
            conf_chart = StackedBarFlowable(conf_data, width=420, bar_height=28)
            cc_table = Table([[conf_chart]], colWidths=[6.2 * inch])
            cc_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F5F7FA")),
                ('PADDING', (0, 0), (-1, -1), 8),
            ]))
            story.append(cc_table)
            story.append(Spacer(1, 4))
            story.append(Paragraph(
                f"<i>Detected {confident_count} confident, {hedging_count} hedging, and {filler_count} filler phrase instances in agent dialogue.</i>",
                ParagraphStyle('lang_note', parent=self.ps['body'], fontSize=7.5,
                               textColor=self.C['light'], alignment=TA_CENTER)))
            story.append(Spacer(1, 10))

            # ── Log to JSON ──
            self.viz_logger.log_chart(
                chart_id="confidence_vs_uncertainty_language",
                title="Confidence vs Uncertainty Language",
                chart_type="stackedBar",
                labels=["Language Type"],
                datasets=[
                    {"label": "Confident", "values": [confident_count], "colors": ["#43A047"]},
                    {"label": "Hedging", "values": [hedging_count], "colors": ["#FF8F00"]},
                    {"label": "Filler Words", "values": [filler_count], "colors": ["#E53935"]},
                ],
                description="Counts of confident vs hedging phrases from the agent's actual dialogue.",
                metadata={"total_phrases": total_lang,
                          "confident_count": confident_count,
                          "hedging_count": hedging_count,
                          "filler_count": filler_count},
            )

        # ── Question Quality Distribution — derived from actual transcript ──
        # Classify actual questions from agent's dialogue
        agent_questions = []
        for seg_text in agent_segments:
            sentences = re.split(r'[.!?]+', seg_text)
            for s in sentences:
                s = s.strip()
                if '?' in seg_text and s and any(w in s.lower() for w in ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'do', 'does', 'did', 'would', 'could', 'can', 'is', 'are', 'was', 'were', 'have', 'has']):
                    agent_questions.append(s)

        if agent_questions:
            open_ended = sum(1 for q in agent_questions if any(w in q.lower() for w in ['what', 'how', 'why', 'tell me', 'describe', 'explain', 'walk me through']))
            closed = sum(1 for q in agent_questions if any(w in q.lower() for w in ['is ', 'are ', 'do ', 'does ', 'did ', 'can ', 'was ', 'were ', 'have you', 'has ']))
            leading = sum(1 for q in agent_questions if any(w in q.lower() for w in ['don\'t you', 'wouldn\'t you', 'isn\'t it', 'right?', 'agree?', 'correct?', 'fair?']))
            assumptive = sum(1 for q in agent_questions if any(w in q.lower() for w in ['when you', 'once we', 'after we', 'so you\'re']))

            q_total = open_ended + closed + leading + assumptive
            if q_total > 0:
                story.append(Paragraph("<b>Question Type Distribution</b>",
                             ParagraphStyle('v_l', parent=self.ps['body_bold'], fontSize=9)))
                story.append(Paragraph(
                    f"Classified {len(agent_questions)} questions from the agent's actual dialogue.",
                    ParagraphStyle('v_l_d', parent=self.ps['body'], fontSize=7.5, textColor=self.C['medium'])))
                story.append(Spacer(1, 4))

                q_data = [
                    ("All Questions", [
                        ("Open-Ended", open_ended, "#43A047"),
                        ("Closed", closed, "#1565C0"),
                        ("Leading", leading, "#FF8F00"),
                        ("Assumptive", assumptive, "#E53935"),
                    ]),
                ]
                q_chart = StackedBarFlowable(q_data, width=420, bar_height=28)
                qc_table = Table([[q_chart]], colWidths=[6.2 * inch])
                qc_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#FAFAFA")),
                    ('PADDING', (0, 0), (-1, -1), 8),
                ]))
                story.append(qc_table)
                story.append(Spacer(1, 4))
                story.append(Paragraph(
                    f"<i>Open-Ended: {open_ended} | Closed: {closed} | Leading: {leading} | Assumptive: {assumptive}</i>",
                    ParagraphStyle('q_note', parent=self.ps['body'], fontSize=7.5,
                                   textColor=self.C['light'], alignment=TA_CENTER)))

                # ── Log to JSON ──
                self.viz_logger.log_chart(
                    chart_id="question_type_distribution",
                    title="Question Type Distribution",
                    chart_type="stackedBar",
                    labels=["All Questions"],
                    datasets=[
                        {"label": "Open-Ended", "values": [open_ended], "colors": ["#43A047"]},
                        {"label": "Closed", "values": [closed], "colors": ["#1565C0"]},
                        {"label": "Leading", "values": [leading], "colors": ["#FF8F00"]},
                        {"label": "Assumptive", "values": [assumptive], "colors": ["#E53935"]},
                    ],
                    description=f"Classified {len(agent_questions)} questions from agent dialogue.",
                    metadata={"total_questions": len(agent_questions),
                              "open_ended": open_ended, "closed": closed,
                              "leading": leading, "assumptive": assumptive},
                )
        story.append(Spacer(1, 12))

        # ═══════════════════════════════════════════
        # 7. RISK, ETHICS & COMPLIANCE
        # ═══════════════════════════════════════════
        story.append(CondPageBreak(2.5 * inch))
        story.append(Paragraph("<b>7. Risk, Ethics & Compliance</b>", self.ps['subsec']))
        story.append(Spacer(1, 6))

        # ── Pressure & Risk Indicator ──
        story.append(Paragraph("<b>Pressure & Risk Indicator</b>",
                     ParagraphStyle('v_m', parent=self.ps['body_bold'], fontSize=9)))
        story.append(Paragraph(
            "Enterprise-critical: flags urgency pressure, over-promising, discount pushing, scarcity language.",
            ParagraphStyle('v_m_d', parent=self.ps['body'], fontSize=7.5, textColor=self.C['medium'])))
        story.append(Spacer(1, 4))

        risk_categories = ['PRESSURE TACTICS', 'OVER-PROMISING', 'REGULATORY', 'TRANSPARENCY']
        risk_bar_data = []
        for key in risk_categories:
            sub = ethics_subs.get(key, '')
            # Parse risk level
            sub_lower = sub.lower()
            if 'high' in sub_lower or 'flag' in sub_lower or 'violation' in sub_lower:
                val, col = 8, "#E53935"
            elif 'medium' in sub_lower or 'moderate' in sub_lower:
                val, col = 5, "#FF8F00"
            elif 'low' in sub_lower or 'none' in sub_lower or 'compliant' in sub_lower:
                val, col = 2, "#43A047"
            else:
                val, col = 4, "#FF8F00"
            risk_bar_data.append((key.title(), val, col))

        if risk_bar_data:
            risk_chart = HBarChartFlowable(risk_bar_data, width=420, bar_height=22, max_val=10)
            rc_table = Table([[risk_chart]], colWidths=[6.2 * inch])
            rc_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F5F7FA")),
                ('PADDING', (0, 0), (-1, -1), 8),
            ]))
            story.append(rc_table)
            story.append(Spacer(1, 10))

            # ── Log to JSON ──
            self.viz_logger.log_chart(
                chart_id="pressure_risk_indicator",
                title="Pressure & Risk Indicator",
                chart_type="horizontalBar",
                labels=[d[0] for d in risk_bar_data],
                datasets=[{"label": "Risk Level", "values": [d[1] for d in risk_bar_data],
                           "colors": [d[2] for d in risk_bar_data]}],
                description="Enterprise-critical: flags urgency pressure, over-promising, discount pushing, scarcity language.",
                metadata={"scale_note": "Higher = more risk detected"},
                options={"max_value": 10},
            )

        # ── N. Trust vs Pressure Quadrant ──
        story.append(CondPageBreak(2.5 * inch))
        story.append(Paragraph("<b>Trust vs Pressure Quadrant</b>",
                     ParagraphStyle('v_n', parent=self.ps['body_bold'], fontSize=9)))
        story.append(Paragraph(
            "Best agents live in HIGH TRUST / LOW PRESSURE (bottom-right). This plots the agent's position.",
            ParagraphStyle('v_n_d', parent=self.ps['body'], fontSize=7.5, textColor=self.C['medium'])))
        story.append(Spacer(1, 4))

        # Calculate trust & pressure scores — ONLY from real data
        trust_val = None
        pressure_val = None
        if scores:
            # Trust: average of empathy-related scores
            trust_scores = [v for k, v in scores.items()
                            if any(w in k.lower() for w in ['rapport', 'empathy', 'trust', 'listening', 'emotional'])]
            if trust_scores:
                trust_val = sum(trust_scores) / len(trust_scores)
            # Pressure: from ethics assessment
            pressure_sub = ethics_subs.get('PRESSURE TACTICS', '').lower()
            if 'high' in pressure_sub:
                pressure_val = 8
            elif 'medium' in pressure_sub or 'moderate' in pressure_sub:
                pressure_val = 5
            elif 'low' in pressure_sub or 'none' in pressure_sub:
                pressure_val = 2

        if trust_val is None or pressure_val is None:
            story.append(Paragraph(
                "<i>Insufficient data to plot Trust vs Pressure quadrant. Need both trust-related scores and ethics assessment data.</i>",
                ParagraphStyle('no_quad', parent=self.ps['body'], fontSize=8,
                               textColor=self.C['light'], alignment=TA_CENTER)))
            story.append(Spacer(1, 12))
        else:
            quadrant = QuadrantFlowable(
                [("This Agent", trust_val, pressure_val)],
                width=280, height=280,
                x_label="Trust-Building →",
                y_label="Pressure →")
            quad_table = Table([[quadrant]], colWidths=[6.2 * inch])
            quad_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#FAFAFA")),
                ('PADDING', (0, 0), (-1, -1), 10),
            ]))
            story.append(quad_table)
            story.append(Spacer(1, 12))

            # ── Log to JSON ──
            self.viz_logger.log_chart(
                chart_id="trust_vs_pressure_quadrant",
                title="Trust vs Pressure Quadrant",
                chart_type="scatter",
                labels=["This Agent"],
                datasets=[{"label": "Agent Position",
                           "values": [{"x": round(trust_val, 1), "y": round(pressure_val, 1)}],
                           "colors": ["#1565C0"]}],
                description="Best agents live in HIGH TRUST / LOW PRESSURE (bottom-right).",
                metadata={"quadrants": [
                    {"name": "High Pressure / Low Trust", "x_range": [0, 5], "y_range": [5, 10], "color": "#FFEBEE"},
                    {"name": "High Pressure / High Trust", "x_range": [5, 10], "y_range": [5, 10], "color": "#FFF8E1"},
                    {"name": "Low Pressure / Low Trust", "x_range": [0, 5], "y_range": [0, 5], "color": "#E3F2FD"},
                    {"name": "Low Pressure / High Trust", "x_range": [5, 10], "y_range": [0, 5], "color": "#E8F5E9"},
                ], "ideal_zone": {"x": 7.5, "y": 2.5}},
                options={"x_label": "Trust-Building", "y_label": "Pressure", "axis_range": [0, 10]},
            )

        # ═══════════════════════════════════════════
        # 8. AGENT COACHING & GROWTH
        # ═══════════════════════════════════════════
        story.append(CondPageBreak(2.5 * inch))
        story.append(Paragraph("<b>8. Agent Coaching & Growth</b>", self.ps['subsec']))
        story.append(Spacer(1, 6))

        # ── Agent Skill Profile (Radar) ──
        if scores and len(scores) >= 3:
            story.append(Paragraph("<b>Agent Skill Profile Radar</b>",
                         ParagraphStyle('v_o', parent=self.ps['body_bold'], fontSize=9)))
            story.append(Paragraph(
                "Multi-dimensional view — used for long-term growth tracking across all evaluated dimensions.",
                ParagraphStyle('v_o_d', parent=self.ps['body'], fontSize=7.5, textColor=self.C['medium'])))
            story.append(Spacer(1, 4))

            chart = RadarChartFlowable(scores, width=300, height=300)
            chart_table = Table([[chart]], colWidths=[6.2 * inch])
            chart_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('PADDING', (0, 0), (-1, -1), 10),
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F5F7FA")),
            ]))
            story.append(chart_table)
            story.append(Spacer(1, 12))

            # ── Log to JSON ──
            self.viz_logger.log_chart(
                chart_id="agent_skill_profile_radar",
                title="Agent Skill Profile Radar",
                chart_type="radar",
                labels=list(scores.keys()),
                datasets=[{"label": "Skill Score", "values": list(scores.values()),
                           "colors": ["#1565C0"], "max": 10}],
                description="Multi-dimensional skill profile for long-term growth tracking.",
                options={"max_value": 10},
            )

        # ═══════════════════════════════════════════
        # 9. EMOTIONAL INTELLIGENCE DIMENSIONS
        # ═══════════════════════════════════════════
        eq_dims = getattr(self, '_cached_eq_dimensions', {})
        eq_scores_viz = {}
        for dim_name, dim_content in eq_dims.items():
            sc = re.search(r'(\d+)\s*/\s*(\d+)', str(dim_content))
            if sc:
                eq_scores_viz[dim_name.title()] = int(sc.group(1))

        if eq_scores_viz and len(eq_scores_viz) >= 3:
            story.append(CondPageBreak(3.0 * inch))
            story.append(Paragraph("<b>9. Emotional Intelligence Dimensions</b>", self.ps['subsec']))
            story.append(Spacer(1, 6))
            story.append(Paragraph("<b>EQ Radar — Multi-Dimensional Emotional Intelligence</b>",
                         ParagraphStyle('v_eq', parent=self.ps['body_bold'], fontSize=9)))
            story.append(Paragraph(
                "Plots each EQ dimension scored by the LLM. Managers can identify emotional skill gaps at a glance.",
                ParagraphStyle('v_eq_d', parent=self.ps['body'], fontSize=7.5, textColor=self.C['medium'])))
            story.append(Spacer(1, 4))

            eq_radar = RadarChartFlowable(eq_scores_viz, width=280, height=280)
            eq_table = Table([[eq_radar]], colWidths=[6.2 * inch])
            eq_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F5F7FA")),
                ('PADDING', (0, 0), (-1, -1), 10),
            ]))
            story.append(eq_table)
            story.append(Spacer(1, 12))

            # ── Log to JSON ──
            self.viz_logger.log_chart(
                chart_id="emotional_intelligence_radar",
                title="EQ Radar — Emotional Intelligence Dimensions",
                chart_type="radar",
                labels=list(eq_scores_viz.keys()),
                datasets=[{"label": "EQ Score", "values": list(eq_scores_viz.values()),
                           "colors": ["#7B1FA2"], "max": 10}],
                description="Plots each EQ dimension to identify emotional skill gaps.",
                options={"max_value": 10},
            )

        # ═══════════════════════════════════════════
        # 10. TONE & ENGAGEMENT METRICS
        # ═══════════════════════════════════════════
        tone_scores = getattr(self, '_cached_tone_scores', {})
        tone_viz = {k: v for k, v in tone_scores.items() if k != 'OVERALL' and isinstance(v, (int, float))}

        if tone_viz:
            story.append(CondPageBreak(2.5 * inch))
            story.append(Paragraph("<b>10. Tone & Engagement Metrics</b>", self.ps['subsec']))
            story.append(Spacer(1, 6))
            story.append(Paragraph("<b>Client Engagement & Tone Score Breakdown</b>",
                         ParagraphStyle('v_tone', parent=self.ps['body_bold'], fontSize=9)))
            story.append(Paragraph(
                "How well the agent managed tone, energy, and client rapport throughout the conversation.",
                ParagraphStyle('v_tone_d', parent=self.ps['body'], fontSize=7.5, textColor=self.C['medium'])))
            story.append(Spacer(1, 4))

            tone_bar_data = []
            for label, val in tone_viz.items():
                pct = val / 10
                col = "#43A047" if pct >= 0.75 else "#FF8F00" if pct >= 0.50 else "#E53935"
                tone_bar_data.append((label.title().replace('_', ' '), val, col))
            tone_chart = HBarChartFlowable(tone_bar_data, width=440, bar_height=22, max_val=10)
            tc_table = Table([[tone_chart]], colWidths=[6.2 * inch])
            tc_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F5F7FA")),
                ('PADDING', (0, 0), (-1, -1), 8),
            ]))
            story.append(tc_table)
            story.append(Spacer(1, 12))

            # ── Log to JSON ──
            self.viz_logger.log_chart(
                chart_id="tone_engagement_scores",
                title="Client Engagement & Tone Score Breakdown",
                chart_type="horizontalBar",
                labels=[d[0] for d in tone_bar_data],
                datasets=[{"label": "Tone Score", "values": [d[1] for d in tone_bar_data],
                           "colors": [d[2] for d in tone_bar_data]}],
                description="How well the agent managed tone, energy, and client rapport.",
                options={"max_value": 10},
            )

        # ═══════════════════════════════════════════
        # 11. NEGOTIATION & PERSUASION METRICS
        # ═══════════════════════════════════════════
        neg_scores = getattr(self, '_cached_negotiation_scores', {})
        neg_viz = {k: v for k, v in neg_scores.items() if k != 'OVERALL' and isinstance(v, (int, float))}

        if neg_viz:
            story.append(CondPageBreak(2.5 * inch))
            story.append(Paragraph("<b>11. Negotiation & Persuasion Metrics</b>", self.ps['subsec']))
            story.append(Spacer(1, 6))
            story.append(Paragraph("<b>Negotiation Proficiency Score Breakdown</b>",
                         ParagraphStyle('v_neg', parent=self.ps['body_bold'], fontSize=9)))
            story.append(Paragraph(
                "How effectively the agent negotiated, persuaded, and advanced the deal.",
                ParagraphStyle('v_neg_d', parent=self.ps['body'], fontSize=7.5, textColor=self.C['medium'])))
            story.append(Spacer(1, 4))

            neg_bar_data = []
            for label, val in neg_viz.items():
                pct = val / 10
                col = "#43A047" if pct >= 0.75 else "#FF8F00" if pct >= 0.50 else "#E53935"
                neg_bar_data.append((label.title().replace('_', ' '), val, col))
            neg_chart = HBarChartFlowable(neg_bar_data, width=440, bar_height=22, max_val=10)
            nc_table = Table([[neg_chart]], colWidths=[6.2 * inch])
            nc_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F5F7FA")),
                ('PADDING', (0, 0), (-1, -1), 8),
            ]))
            story.append(nc_table)
            story.append(Spacer(1, 12))

            # ── Log to JSON ──
            self.viz_logger.log_chart(
                chart_id="negotiation_proficiency_scores",
                title="Negotiation Proficiency Score Breakdown",
                chart_type="horizontalBar",
                labels=[d[0] for d in neg_bar_data],
                datasets=[{"label": "Negotiation Score", "values": [d[1] for d in neg_bar_data],
                           "colors": [d[2] for d in neg_bar_data]}],
                description="How effectively the agent negotiated, persuaded, and advanced the deal.",
                options={"max_value": 10},
            )

        # ═══════════════════════════════════════════
        # 12. MANAGER-FACING SUMMARY
        # ═══════════════════════════════════════════

        # Final Score — only from real data
        if not scores:
            story.append(Paragraph(
                "<i>No performance scores available — dashboard metrics cannot be calculated.</i>",
                ParagraphStyle('no_dash', parent=self.ps['body'], fontSize=9,
                               textColor=self.C['light'], alignment=TA_CENTER)))
            return

        overall = sum(scores.values()) / len(scores)
        pct = overall / 10
        sc = "#43A047" if pct >= 0.75 else "#FF8F00" if pct >= 0.50 else "#E53935"

        # Dashboard row 1: Overall Score Card + Deal Momentum Card
        if momentum_val > 0:
            momentum_zone = "HOT" if momentum_val >= 67 else "WARM" if momentum_val >= 34 else "COLD"
            mz_color = "#43A047" if momentum_val >= 67 else "#FF8F00" if momentum_val >= 34 else "#E53935"
            momentum_display = f"{momentum_val}%"
        else:
            momentum_zone = "N/A"
            mz_color = "#757575"
            momentum_display = "N/A"

        # Use a proper multi-row table to avoid text overlap
        dash_row1 = Table([
            # Row 0: Big numbers
            [Paragraph(f"<b>{overall:.1f}</b>",
                       ParagraphStyle('ds_num', fontName='Helvetica-Bold', fontSize=28,
                                      leading=34, textColor=colors.HexColor(sc), alignment=TA_CENTER)),
             Paragraph(f"<b>{momentum_display}</b>",
                       ParagraphStyle('dm_num', fontName='Helvetica-Bold', fontSize=28,
                                      leading=34, textColor=colors.HexColor(mz_color), alignment=TA_CENTER))],
            # Row 1: Sub-labels
            [Paragraph("out of 10",
                       ParagraphStyle('ds_sub', fontName='Helvetica', fontSize=8,
                                      leading=10, textColor=self.C['light'], alignment=TA_CENTER)),
             Paragraph(f"{momentum_zone} Zone",
                       ParagraphStyle('dm_sub', fontName='Helvetica-Bold', fontSize=9,
                                      leading=11, textColor=colors.HexColor(mz_color), alignment=TA_CENTER))],
            # Row 2: Category labels
            [Paragraph("OVERALL PERFORMANCE",
                       ParagraphStyle('ds_cat', fontName='Helvetica', fontSize=8,
                                      leading=10, textColor=self.C['light'], alignment=TA_CENTER,
                                      spaceBefore=4)),
             Paragraph("DEAL MOMENTUM",
                       ParagraphStyle('dm_cat', fontName='Helvetica', fontSize=8,
                                      leading=10, textColor=self.C['light'], alignment=TA_CENTER,
                                      spaceBefore=4))],
        ], colWidths=[3.1 * inch, 3.1 * inch],
           rowHeights=[40, 16, 20])
        dash_row1.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, 0), 'MIDDLE'),
            ('VALIGN', (0, 1), (-1, -1), 'TOP'),
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#F5F7FA")),
            ('BACKGROUND', (1, 0), (1, -1), colors.HexColor("#F5F7FA")),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 1), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ('BOX', (0, 0), (0, -1), 1, colors.HexColor(sc)),
            ('BOX', (1, 0), (1, -1), 1, colors.HexColor(mz_color)),
        ]))
        story.append(dash_row1)
        story.append(Spacer(1, 8))

        # Dashboard row 2: Key Metrics (4-column)
        risk_level = ethics_subs.get('OVERALL ETHICS RISK', '')
        risk_first = risk_level.split('\n')[0].strip() if risk_level else ''
        if not risk_first:
            risk_first = 'N/A'
            risk_color = "#757575"
        else:
            risk_color = "#C62828" if 'high' in risk_first.lower() else "#2E7D32" if 'low' in risk_first.lower() or 'none' in risk_first.lower() else "#F57F17"
        if agent_pct and client_pct:
            talk_color = "#C62828" if agent_pct > 65 else "#2E7D32" if 45 <= agent_pct <= 55 else "#F57F17"
            talk_display = f"{agent_pct}% / {client_pct}%"
        else:
            talk_color = "#757575"
            talk_display = "N/A"
        if sentiment_data and len(sentiment_data) >= 2:
            sent_direction = "Improving" if sentiment_data[-1] > sentiment_data[0] else "Declining" if sentiment_data[-1] < sentiment_data[0] else "Stable"
        else:
            sent_direction = "N/A"
        sent_color = "#2E7D32" if sent_direction == "Improving" else "#C62828" if sent_direction == "Declining" else "#F57F17" if sent_direction == "Stable" else "#757575"

        dash_row2 = Table([
            [Paragraph("<b>Talk Ratio<br/>(Agent/Client)</b>", self.ps['th']),
             Paragraph("<b>Ethics Risk</b>", self.ps['th']),
             Paragraph("<b>Sentiment Trend</b>", self.ps['th']),
             Paragraph("<b>Top Weakness</b>", self.ps['th'])],
            [Paragraph(f"<font color='{talk_color}'><b>{talk_display}</b></font>", self.ps['td_c']),
             Paragraph(f"<font color='{risk_color}'><b>{risk_first}</b></font>", self.ps['td_c']),
             Paragraph(f"<font color='{sent_color}'><b>{sent_direction}</b></font>", self.ps['td_c']),
             Paragraph(f"<font color='#C62828'><b>{sorted(scores.items(), key=lambda x: x[1])[0][0] if scores else 'N/A'}</b></font>", self.ps['td_c'])],
        ], colWidths=[1.55 * inch, 1.55 * inch, 1.55 * inch, 1.55 * inch])
        dash_row2.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.C['primary_dark']),
            ('TEXTCOLOR', (0, 0), (-1, 0), self.C['white']),
            ('GRID', (0, 0), (-1, -1), 0.5, self.C['border']),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(dash_row2)
        story.append(Spacer(1, 8))

        # ── Log manager dashboard to JSON ──
        top_weakness = sorted(scores.items(), key=lambda x: x[1])[0][0] if scores else 'N/A'
        self.viz_logger.log_chart(
            chart_id="manager_dashboard_summary",
            title="Manager-Facing Summary Dashboard",
            chart_type="scoreCard",
            labels=["Overall Performance", "Deal Momentum", "Talk Ratio", "Ethics Risk",
                    "Sentiment Trend", "Top Weakness"],
            datasets=[{
                "label": "Dashboard Metrics",
                "values": [
                    {"metric": "Overall Performance", "value": round(overall, 1), "max": 10, "color": sc},
                    {"metric": "Deal Momentum", "value": momentum_val, "max": 100,
                     "zone": momentum_zone, "color": mz_color},
                    {"metric": "Talk Ratio (Agent/Client)", "value": talk_display, "color": talk_color},
                    {"metric": "Ethics Risk", "value": risk_first, "color": risk_color},
                    {"metric": "Sentiment Trend", "value": sent_direction, "color": sent_color},
                    {"metric": "Top Weakness", "value": top_weakness, "color": "#C62828"},
                ],
            }],
            description="Executive summary dashboard with key performance indicators.",
        )

        # Top 3 coaching priorities as a proper table
        story.append(Paragraph("<b>Top 3 Coaching Priorities</b>", self.ps['subsec']))
        if scores:
            sorted_scores = sorted(scores.items(), key=lambda x: x[1])
            prio_rows = [[Paragraph("<b>#</b>", self.ps['th']),
                          Paragraph("<b>Skill Area</b>", self.ps['th']),
                          Paragraph("<b>Score</b>", self.ps['th']),
                          Paragraph("<b>Action Required</b>", self.ps['th'])]]
            for i, (label, val) in enumerate(sorted_scores[:3], 1):
                pct_v = val / 10
                col = "#E53935" if pct_v < 0.50 else "#FF8F00" if pct_v < 0.75 else "#43A047"
                action = 'CRITICAL — immediate intervention needed' if pct_v < 0.50 else 'Needs focused practice' if pct_v < 0.75 else 'Maintain and refine'
                prio_rows.append([
                    Paragraph(f"<b>{i}</b>", self.ps['td_c']),
                    Paragraph(f"<b>{label}</b>", self.ps['td']),
                    Paragraph(f"<font color='{col}'><b>{val}/10</b></font>", self.ps['td_c']),
                    Paragraph(f"<font color='{col}'>{action}</font>", self.ps['td']),
                ])
            prio_t = Table(prio_rows, colWidths=[0.4 * inch, 2.2 * inch, 0.8 * inch, 2.8 * inch])
            prio_t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), self.C['danger']),
                ('TEXTCOLOR', (0, 0), (-1, 0), self.C['white']),
                ('GRID', (0, 0), (-1, -1), 0.5, self.C['border']),
                ('PADDING', (0, 0), (-1, -1), 5),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            story.append(prio_t)

            # ── Log to JSON ──
            self.viz_logger.log_chart(
                chart_id="top_coaching_priorities",
                title="Top 3 Coaching Priorities",
                chart_type="table",
                labels=["Rank", "Skill Area", "Score", "Action Required"],
                datasets=[{
                    "label": "Coaching Priorities",
                    "values": [
                        {"rank": i + 1, "skill": label, "score": val, "max": 10,
                         "action": 'CRITICAL — immediate intervention needed' if val / 10 < 0.50
                         else 'Needs focused practice' if val / 10 < 0.75 else 'Maintain and refine'}
                        for i, (label, val) in enumerate(sorted_scores[:3])
                    ],
                }],
                description="The three lowest-scoring dimensions requiring immediate coaching attention.",
            )

        # ── Deal Intelligence Signal Map ──
        if deal_metrics:
            story.append(Spacer(1, 8))
            story.append(CondPageBreak(2.0 * inch))
            story.append(Paragraph("<b>Deal Intelligence Signal Map</b>", self.ps['subsec']))
            story.append(Spacer(1, 4))
            for metric_item in deal_metrics:
                label = metric_item[0]
                value = metric_item[1]
                just = metric_item[2] if len(metric_item) > 2 else ''
                val_lower = value.lower().strip()
                if val_lower in ['strong', 'high'] and 'risk' not in label.lower():
                    ind_color = "#43A047"
                elif val_lower in ['weak', 'low'] and 'risk' not in label.lower():
                    ind_color = "#E53935"
                elif val_lower in ['high'] and 'risk' in label.lower():
                    ind_color = "#E53935"
                elif val_lower in ['low'] and 'risk' in label.lower():
                    ind_color = "#43A047"
                elif '%' in value:
                    try:
                        pval = int(value.replace('%', '').strip())
                        ind_color = "#43A047" if pval >= 70 else "#FF8F00" if pval >= 50 else "#E53935"
                    except ValueError:
                        ind_color = "#FF8F00"
                else:
                    ind_color = "#FF8F00"
                row = Table([[
                    Paragraph(label, self.ps['td']),
                    Paragraph(f"<b><font color='{ind_color}'>{value}</font></b>", self.ps['td_c']),
                    Paragraph(f"<i>{just}</i>" if just else "", self.ps['td']),
                ]], colWidths=[1.8 * inch, 1.0 * inch, 3.4 * inch])
                row.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F5F7FA")),
                    ('PADDING', (0, 0), (-1, -1), 5),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                story.append(row)
                story.append(Spacer(1, 3))

            # ── Log to JSON ──
            self.viz_logger.log_chart(
                chart_id="deal_intelligence_signal_map",
                title="Deal Intelligence Signal Map",
                chart_type="table",
                labels=["Metric", "Value", "Justification"],
                datasets=[{
                    "label": "Deal Signals",
                    "values": [
                        {"metric": m[0], "value": m[1],
                         "justification": m[2] if len(m) > 2 else "",
                         "confidence": m[3].strip() if len(m) > 3 else ""}
                        for m in deal_metrics
                    ],
                }],
                description="AI-evaluated deal intelligence signals with confidence ratings.",
            )

    # ════════════════════════════════════════════════════════
    # LEGACY COMPAT METHODS
    # ════════════════════════════════════════════════════════

    def create_professional_header_footer(self, canvas, doc):
        self._draw_content_header_footer(canvas, doc)

    def create_first_page_header_footer(self, canvas, doc):
        self._draw_cover_header_footer(canvas, doc)

    def create_unified_first_page_header_footer(self, canvas, doc):
        self._draw_cover_header_footer(canvas, doc)

    def create_unified_header_footer(self, canvas, doc):
        self._draw_content_header_footer(canvas, doc)


# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  REAL ESTATE COACHING EVALUATION SYSTEM v3.0")
    print("  Enterprise Intelligence Engine — 3-PDF + JSON Output")
    print("  PDF 1: Coaching Summary")
    print("  PDF 2: Agent Profile")
    print("  PDF 3: Visualizations & Analytics")
    print("  JSON:  visualizations_backup.json (frontend-ready)")
    print("=" * 60)

    default_file = "conversation.json"
    transcript_file = input(f"  Enter JSON transcript path (Enter for '{default_file}'): ").strip().strip('"')
    if not transcript_file:
        transcript_file = default_file

    evaluator = RealEstateSalesMeetingSummarizer(transcript_file=transcript_file)
    print(f"\n  Generating brutally honest evaluation (3 PDFs)...")
    evaluator.generate_unified_coaching_evaluation()


if __name__ == "__main__":
    main()