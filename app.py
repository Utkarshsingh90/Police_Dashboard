import streamlit as st
import json
from datetime import datetime
from io import BytesIO
import re
from transformers import pipeline
import torch
from typing import Dict, List, Optional
import pandas as pd
from langdetect import detect, LangDetectException
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch
import warnings
import spacy
import os
from collections import defaultdict
import pdfplumber

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Police Recognition Analytics",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# DARK THEME CSS with Perfect Text Visibility (Copied from your code)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
   .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        font-family: 'Inter', sans-serif;
    }
   .main-header {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        padding: 30px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 20px;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.3);
    }
   .info-box {
        padding: 20px;
        border-radius: 12px;
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
        border-left: 6px solid #3b82f6;
        margin: 15px 0;
        color: #e0e7ff!important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.2);
    }
   .info-box h4 { color: #93c5fd!important; margin-bottom: 10px; }
   .success-box {
        padding: 18px;
        border-radius: 12px;
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        border-left: 6px solid #10b981;
        margin: 15px 0;
        color: #d1fae5!important;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.2);
    }
   .success-box h4 { color: #6ee7b7!important; }
   .warning-box {
        padding: 18px;
        border-radius: 12px;
        background: linear-gradient(135deg, #92400e 0%, #b45309 100%);
        border-left: 6px solid #f59e0b;
        margin: 15px 0;
        color: #fef3c7!important;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.2);
    }
   .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: #ffffff!important;
        text-align: center;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
    }
   .metric-card:hover { transform: translateY(-5px); box-shadow: 0 12px 30px rgba(102, 126, 234, 0.4); }
   .metric-card h2,.metric-card p { color: #ffffff!important; }
   .stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: #1f2937; padding: 10px; border-radius: 12px; }
   .stTabs [data-baseweb="tab"] {
        height: 55px;
        padding: 0 25px;
        background-color: #374151;
        border-radius: 10px;
        color: #d1d5db!important;
        font-weight: 600;
        border: 2px solid #4b5563;
        transition: all 0.3s ease;
    }
   .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff!important;
        border-color: #667eea;
    }
   .stTextArea textarea {
        background-color: #1f2937!important;
        color: #f3f4f6!important;
        border: 2px solid #4b5563!important;
        font-size: 16px!important;
        line-height: 1.6!important;
        border-radius: 8px!important;
    }
   .stTextArea textarea:focus {
        border-color: #667eea!important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.3)!important;
        background-color: #111827!important;
    }
   .stTextArea textarea::placeholder { color: #9ca3af!important; opacity: 1!important; }
   .stTextInput input {
        background-color: #1f2937!important;
        color: #f3f4f6!important;
        border: 2px solid #4b5563!important;
        font-size: 16px!important;
        border-radius: 8px!important;
    }
   .stTextInput input:focus { border-color: #667eea!important; box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.3)!important; }
   label,.stMarkdown, h1, h2, h3, h4, h5, h6, p, span, div { color: #f3f4f6!important; }
   .stMarkdown h1,.stMarkdown h2,.stMarkdown h3 { color: #e0e7ff!important; }
   .stRadio > label { color: #f3f4f6!important; font-weight: 600!important; }
   .stRadio > div { color: #d1d5db!important; }
   .stRadio [role="radiogroup"] label { color: #e5e7eb!important; }
   .stMarkdown { color: #f3f4f6!important; }
   h1, h2, h3 { color: #e0e7ff!important; }
   .uploadedFile { background-color: #374151!important; color: #f3f4f6!important; border: 2px solid #4b5563!important; }
   [data-testid="stFileUploader"] { background-color: #1f2937; border: 2px dashed #4b5563; border-radius: 10px; padding: 20px; }
   [data-testid="stFileUploader"] label { color: #d1d5db!important; }
   .stButton button { font-weight: 600; border-radius: 8px; padding: 12px 24px; transition: all 0.3s ease; color: #ffffff!important; }
   .stButton button[kind="primary"] { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border: none; }
   .stButton button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4); }
   .streamlit-expanderHeader { background-color: #1f2937!important; color: #f3f4f6!important; font-weight: 600; border-radius: 8px; border: 1px solid #4b5563; }
   .streamlit-expanderHeader:hover { background-color: #374151!important; }
   .streamlit-expanderContent { background-color: #111827; border: 1px solid #4b5563; border-top: none; }
   .dataframe { color: #f3f4f6!important; background-color: #1f2937; }
   [data-testid="stMetricValue"] { color: #e0e7ff!important; font-size: 2rem!important; font-weight: 700!important; }
   [data-testid="stMetricLabel"] { color: #9ca3af!important; }
   .stSuccess { background-color: #065f46!important; color: #d1fae5!important; border-left: 5px solid #10b981; }
   .stError { background-color: #991b1b!important; color: #fecaca!important; border-left: 5px solid #ef4444; }
   .stWarning { background-color: #92400e!important; color: #fef3c7!important; border-left: 5px solid #f59e0b; }
   .stInfo { background-color: #1e3a8a!important; color: #dbeafe!important; border-left: 5px solid #3b82f6; }
   .stDownloadButton button { background: linear-gradient(135deg, #059669 0%, #047857 100%); color: #ffffff!important; }
   .stSelectbox,.stMultiSelect { color: #f3f4f6!important; }
   .stSelectbox > div > div { background-color: #1f2937!important; color: #f3f4f6!important; border: 2px solid #4b5563!important; }
   .stSpinner > div { border-top-color: #667eea!important; }
   a { color: #93c5fd!important; }
   a:hover { color: #bfdbfe!important; }
   hr { border-color: #4b5563!important; }
   .element-container { color: #f3f4f6!important; }
   .vega-embed text { fill: #d1d5db!important; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Cache models
@st.cache_resource(show_spinner="Loading AI Models...")
def load_models():
    """Load ML models"""
    try:
        device = 0 if torch.cuda.is_available() else -1

        # 1. Sentiment Analyzer (English)
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=device
        )

        # 2. Summarizer (English)
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=device
        )

        # 3. Q&A Model (English)
        qa_model = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            device=device
        )

        # 4. spaCy EntityRuler (English)
        nlp = spacy.blank("en")
        try:
            ruler = nlp.add_pipe("entity_ruler")
        except Exception:
            # older/newer spacy versions might require different names — try flexible add
            ruler = nlp.add_pipe("entity_ruler", last=True)
        # load patterns.jsonl if exists; otherwise continue (main will check file existence)
        if os.path.exists("patterns.jsonl"):
            ruler.from_disk("patterns.jsonl")

        return sentiment_analyzer, summarizer, qa_model, nlp
    except Exception as e:
        # Surface the error in Streamlit (caller will check)
        return None, None, None, None

def detect_language(text: str) -> str:
    """Detect language. Default to 'en' if detection fails."""
    try:
        return detect(text)
    except LangDetectException:
        return "en"  # Default fallback
    except Exception:
        return "en"

def extract_entities_spacy(text: str, nlp: spacy.Language) -> Dict:
    """
    Use spaCy EntityRuler (patterns.jsonl) to extract entities.
    Return default empty lists for missing labels.
    """
    doc = nlp(text)
    entities = defaultdict(list)
    for ent in doc.ents:
        entities[ent.label_].append(ent.text)

    return {
        "officers": list(set(entities.get("OFFICER", []))),
        "departments": list(set(entities.get("DEPARTMENT", []))),
        "locations": list(set(entities.get("LOCATION", []))),
        "crime_types": list(set(entities.get("CRIME_TYPE", []))),
        "law_sections": list(set(entities.get("LAW_SECTION", []))),
    }

def analyze_sentiment(text: str, sentiment_analyzer) -> Dict:
    """Sentiment analysis (English)"""
    try:
        result = sentiment_analyzer(text[:512])
        if isinstance(result, list):
            result = result[0]
        label = result.get('label', 'NEUTRAL')
        score = float(result.get('score', 0.5))
        normalized_score = score if label == 'POSITIVE' else -score
        return {"label": label, "score": score, "normalized_score": normalized_score}
    except Exception as e:
        return {"label": "NEUTRAL", "score": 0.5, "normalized_score": 0.0}

def extract_competency_tags(text: str) -> List[str]:
    """Extract competency tags (English)"""
    competencies = {
        "community_engagement": ["community", "engagement", "outreach", "relationship", "trust", "friendly", "public"],
        "de-escalation": ["de-escalate", "calm", "peaceful", "resolved", "mediation", "defused", "negotiation"],
        "rapid_response": ["quick", "fast", "immediate", "prompt", "timely", "rapid", "swift", "emergency"],
        "professionalism": ["professional", "courteous", "respectful", "polite", "dignified", "conduct"],
        "life_saving": ["saved", "rescue", "life-saving", "emergency", "critical", "revived", "medical"],
        "investigation": ["investigation", "solved", "detective", "evidence", "arrest", "caught", "crime"],
        "compassion": ["compassion", "care", "kindness", "empathy", "understanding", "helped", "sympathetic"],
        "bravery": ["brave", "courage", "heroic", "danger", "risk", "fearless", "valor", "heroism"]
    }

    text_lower = text.lower()
    found_tags = []

    for tag, keywords in competencies.items():
        if any(keyword in text_lower for keyword in keywords):
            found_tags.append(tag)

    return found_tags if found_tags else ["general_commendation"]

def generate_summary(text: str, summarizer) -> str:
    """Generate comprehensive summary (English)"""
    try:
        if len(text) < 100:
            return text.strip()
        text_to_summarize = text[:2000]  # keep to a reasonable token length
        summary_result = summarizer(text_to_summarize, max_length=150, min_length=40, do_sample=False)
        if isinstance(summary_result, list):
            summary_text = summary_result[0].get('summary_text', '')
        else:
            summary_text = summary_result.get('summary_text', '')
        return summary_text.strip()
    except Exception:
        sentences = text.split('.')[:3]
        return '. '.join(s.strip() for s in sentences if s.strip()) + '.'

def calculate_recognition_score(sentiment_score: float, tags: List[str], text_length: int) -> float:
    """Calculate recognition score"""
    base_score = (sentiment_score + 1) / 2  # map -1..1 to 0..1
    high_value_tags = ["life_saving", "bravery", "de-escalation"]
    tag_boost = sum(0.15 for tag in tags if tag in high_value_tags)
    length_boost = min(0.1, (text_length / 1000) * 0.1)
    final_score = min(1.0, base_score + tag_boost + length_boost)
    return round(final_score, 3)

def create_pdf_summary(result: Dict) -> BytesIO:
    """Create PDF summary report"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=30,
        alignment=1
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#764ba2'),
        spaceAfter=12,
        spaceBefore=12
    )

    story.append(Paragraph("🚔 Police Recognition Report", title_style))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("Summary", heading_style))
    story.append(Paragraph(result.get('summary', ''), styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    # Metrics table
    metrics_data = [
        ['Metric', 'Value'],
        ['Recognition Score', str(result.get('recognition_score', 'N/A'))],
        ['Sentiment', str(result.get('sentiment_label', 'N/A'))],
        ['Text Length', str(result.get('text_length', 0))],
        ['Language', result.get('language_name', 'English (en)')]
    ]
    metrics_table = Table(metrics_data, colWidths=[3*inch, 3*inch])
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ])
    metrics_table.setStyle(table_style)
    story.append(metrics_table)
    story.append(Spacer(1, 0.3*inch))

    # Entities
    story.append(Paragraph("Identified Officers", heading_style))
    officers = result.get('extracted_officers', [])
    if officers:
        for officer in officers:
            story.append(Paragraph(f"• {officer}", styles['Normal']))
    else:
        story.append(Paragraph("• None identified", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("Departments & Locations", heading_style))
    depts = result.get('extracted_departments', [])
    locs = result.get('extracted_locations', [])
    if depts:
        for d in depts:
            story.append(Paragraph(f"• {d}", styles['Normal']))
    else:
        story.append(Paragraph("• None identified", styles['Normal']))
    if locs:
        for l in locs:
            story.append(Paragraph(f"• {l}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("Crime Types & Sections", heading_style))
    crimes = result.get('extracted_crime_types', [])
    sections = result.get('extracted_law_sections', [])
    if crimes:
        for c in crimes:
            story.append(Paragraph(f"• {c}", styles['Normal']))
    else:
        story.append(Paragraph("• None identified", styles['Normal']))
    if sections:
        for s in sections:
            story.append(Paragraph(f"• {s}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("Competency Tags", heading_style))
    tags = result.get('suggested_tags', [])
    if tags:
        for tag in tags:
            story.append(Paragraph(f"• {tag.replace('_', ' ').title()}", styles['Normal']))
    else:
        story.append(Paragraph("• None", styles['Normal']))

    doc.build(story)
    buffer.seek(0)
    return buffer

def process_text(text: str, models_tuple) -> Optional[Dict]:
    """Main processing pipeline (English-only)"""
    sentiment_analyzer, summarizer, qa_model, nlp = models_tuple

    original_text = text or ""
    detected_lang = detect_language(original_text)

    # Filter for English-only
    if detected_lang != 'en':
        st.warning(f"Skipped: This application currently only supports English (detected '{detected_lang}').")
        return None

    processing_text = original_text

    # Use spaCy EntityRuler
    entities = extract_entities_spacy(processing_text, nlp) if nlp is not None else {
        "officers": [], "departments": [], "locations": [], "crime_types": [], "law_sections": []
    }

    # Run other models
    sentiment = analyze_sentiment(processing_text, sentiment_analyzer) if sentiment_analyzer is not None else {"label": "NEUTRAL", "normalized_score": 0.0}
    tags = extract_competency_tags(processing_text)
    summary = generate_summary(processing_text, summarizer) if summarizer is not None else processing_text[:200]

    score = calculate_recognition_score(
        sentiment.get('normalized_score', 0.0),
        tags,
        len(processing_text)
    )

    result = {
        "timestamp": datetime.now().isoformat(),
        "original_text": original_text,
        "detected_language": detected_lang,
        "language_name": "English",
        "translated_text": None,
        "summary": summary,
        "extracted_officers": entities.get('officers', []),
        "extracted_departments": entities.get('departments', []),
        "extracted_locations": entities.get('locations', []),
        "extracted_crime_types": entities.get('crime_types', []),
        "extracted_law_sections": entities.get('law_sections', []),
        "sentiment_label": sentiment.get('label', 'NEUTRAL'),
        "sentiment_score": sentiment.get('normalized_score', 0.0),
        "suggested_tags": tags,
        "recognition_score": score,
        "text_length": len(processing_text)
    }

    return result

def answer_question(question: str, context: str, qa_model) -> str:
    """Q&A"""
    try:
        if qa_model is None:
            return "Q&A model not loaded."
        result = qa_model(question=question, context=context[:2000])
        if isinstance(result, dict):
            return result.get('answer', 'No answer found.')
        # some pipelines return a list
        if isinstance(result, list) and len(result) > 0:
            return result[0].get('answer', 'No answer found.')
        return "No answer found."
    except Exception as e:
        return f"Unable to answer: {str(e)}"

# Main App
def main():
    st.markdown('<h1 class="main-header">🚔 Police Recognition Analytics Platform</h1>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <h4>🎯 Welcome to the AI-Powered Police Recognition System (English-Only)</h4>
        Analyze public feedback, news articles, and documents to identify and recognize outstanding police work. 
        This version supports **English-language text only**.
    </div>
    """, unsafe_allow_html=True)

    # NEW: Check if the pattern file exists.
    if not os.path.exists("patterns.jsonl"):
        st.error("❌ CRITICAL ERROR: 'patterns.jsonl' not found.")
        st.info("Please run the `create_gazetteer.py` script from your terminal ONCE to build the entity database from your JSON files.")
        return

    # Load models
    models = load_models()
    if models is None or any(m is None for m in models):
        st.error("❌ Failed to load AI models. Please check the terminal and ensure dependencies and model access are available.")
        return

    sentiment_analyzer, summarizer, qa_model, nlp = models

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/police-badge.png", width=100)
        st.title("📍 Navigation")

        st.markdown("---")
        st.subheader("📊 Statistics")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("📝 Total", len(st.session_state.processed_data))
        with col2:
            if st.session_state.processed_data:
                avg_score = sum(d['recognition_score'] for d in st.session_state.processed_data) / len(st.session_state.processed_data)
                st.metric("⭐ Avg", f"{avg_score:.2f}")
            else:
                st.metric("⭐ Avg", "N/A")

        st.markdown("---")
        st.subheader("🌐 Supported Languages")
        st.info("**English (en) only.** All other languages will be filtered out.")

        st.markdown("---")
        st.subheader("👮 EntityRulerset")
        st.info(f"Loaded 'patterns.jsonl' successfully. This file was built from your JSON data files and powers the entity extraction.")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.processed_data = []
                st.session_state.chat_history = []
                st.experimental_rerun()
        with col2:
            if st.button("🔄 Refresh", use_container_width=True):
                st.experimental_rerun()

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Process Feedback", "Dashboard", "Q&A Chat", "Export Data"])

    # --- Tab 1: Process Feedback ---
    with tab1:
        st.header("📝 Process New Feedback")

        col1, col2 = st.columns([1, 2])

        with col1:
            input_method = st.radio(
                "📥 Select Input Method:",
                ["✍️ Text Input", "📤 Upload File"],
                horizontal=True
            )

            text_to_process = ""

            if input_method == "✍️ Text Input":
                text_to_process = st.text_area(
                    "Enter English feedback, article, or document:",
                    height=250,
                    placeholder="Example:\nOfficer Smith from Central Police Station showed exceptional bravery...\n(Non-English text will be ignored)",
                    key="main_text_input"
                )
            else:
                uploaded_file = st.file_uploader(
                    "📤 Upload Document (TXT or PDF)",
                    type=['txt', 'pdf']
                )

                if uploaded_file:
                    try:
                        if uploaded_file.type == "text/plain" or uploaded_file.name.endswith('.txt'):
                            # text/plain or fallback to bytes decode
                            raw = uploaded_file.getvalue()
                            if isinstance(raw, bytes):
                                text_to_process = raw.decode("utf-8", errors="ignore")
                            else:
                                text_to_process = str(raw)
                            st.success(f"✅ Loaded {len(text_to_process)} characters from TXT")
                        elif uploaded_file.type == "application/pdf" or uploaded_file.name.endswith('.pdf'):
                            with pdfplumber.open(uploaded_file) as pdf:
                                pages_text = []
                                for page in pdf.pages:
                                    pages_text.append(page.extract_text() or "")
                                text_to_process = "\n".join(pages_text)
                            st.success(f"✅ Extracted {len(text_to_process)} characters from PDF")
                        else:
                            st.warning("Unsupported file type; please upload TXT or PDF.")
                    except Exception as e:
                        st.error(f"❌ Error reading file: {str(e)}")
                        text_to_process = ""

                    if text_to_process:
                        with st.expander("📄 Preview"):
                            st.text_area("Content:", text_to_process[:500] + ("..." if len(text_to_process) > 500 else ""), height=150, key="preview", disabled=True)

        with col2:
            st.markdown("""
            <div class="info-box">
                <h4>🌍 Language</h4>
                • English-Only Support
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="success-box">
                <h4>✨ AI Features</h4>
                ✅ Sentiment Analysis<br>
                ✅ <b>Custom Entity Extraction</b><br>
                ✅ Auto-Summary<br>
                ✅ Competency Tags<br>
                ✅ PDF Reports
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        if st.button("🚀 Analyze Feedback", type="primary", use_container_width=True):
            if text_to_process and text_to_process.strip():
                with st.spinner("🔍 Analyzing English text..."):
                    try:
                        result = process_text(text_to_process, models)
                        if result:
                            st.session_state.processed_data.append(result)
                            st.success("✅ Analysis Complete!")

                            # Metrics cards
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h2>{result['recognition_score']}</h2>
                                    <p>Recognition Score</p>
                                </div>
                                """, unsafe_allow_html=True)
                            with col2:
                                emoji = "😊" if result['sentiment_label'] == 'POSITIVE' else "😐"
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h2>{emoji}</h2>
                                    <p>{result['sentiment_label']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            with col3:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h2>{len(result['extracted_officers'])}</h2>
                                    <p>Officers</p>
                                </div>
                                """, unsafe_allow_html=True)
                            with col4:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h2>{len(result['extracted_crime_types'])}</h2>
                                    <p>Crime Types</p>
                                </div>
                                """, unsafe_allow_html=True)

                            st.markdown("---")

                            # Details
                            with st.expander("📋 View Details", expanded=True):
                                st.subheader("📝 Summary")
                                st.markdown(f"""
                                <div class="info-box">
                                    {result['summary']}
                                </div>
                                """, unsafe_allow_html=True)

                                col1, col2 = st.columns(2)
                                with col1:
                                    st.subheader("👮 Extracted Entities")
                                    st.markdown("**Officers:**")
                                    if result['extracted_officers']:
                                        for o in result['extracted_officers']:
                                            st.markdown(f"• **{o}**")
                                    else:
                                        st.markdown("• None identified")

                                    st.markdown("**Departments:**")
                                    if result['extracted_departments']:
                                        for d in result['extracted_departments']:
                                            st.markdown(f"• **{d}**")
                                    else:
                                        st.markdown("• None identified")

                                    st.markdown("**Locations:**")
                                    if result['extracted_locations']:
                                        for l in result['extracted_locations']:
                                            st.markdown(f"• **{l}**")
                                    else:
                                        st.markdown("• None identified")
                                with col2:
                                    st.subheader("🏷️ Crime & Tags")
                                    st.markdown("**Crime Types:**")
                                    if result['extracted_crime_types']:
                                        for t in result['extracted_crime_types']:
                                            st.markdown(f"• **{t}**")
                                    else:
                                        st.markdown("• None identified")

                                    st.markdown("**Law Sections:**")
                                    if result['extracted_law_sections']:
                                        for s in result['extracted_law_sections']:
                                            st.markdown(f"• **{s}**")
                                    else:
                                        st.markdown("• None identified")

                                    st.markdown("**Competency Tags:**")
                                    for t in result['suggested_tags']:
                                        st.markdown(f"• **{t.replace('_', ' ').title()}**")

                            # Export
                            st.markdown("---")
                            st.subheader("📥 Export")

                            col1, col2 = st.columns(2)
                            with col1:
                                pdf_buffer = create_pdf_summary(result)
                                st.download_button(
                                    "📄 PDF Summary",
                                    data=pdf_buffer.getvalue(),
                                    file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                    mime="application/pdf",
                                    use_container_width=True
                                )
                            with col2:
                                export_result = result.copy()
                                # ensure lists are JSON-serializable
                                for k in ['extracted_officers', 'extracted_departments', 'extracted_locations', 'extracted_crime_types', 'extracted_law_sections']:
                                    export_result[k] = list(export_result.get(k, []))
                                st.download_button(
                                    "📋 JSON Data",
                                    data=json.dumps(export_result, indent=2, ensure_ascii=False),
                                    file_name=f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json",
                                    use_container_width=True
                                )
                        else:
                            st.info("No result returned (likely non-English).")
                    except Exception as e:
                        st.error(f"❌ An error occurred during processing: {str(e)}")
            else:
                st.warning("⚠️ Please enter text or upload a document to analyze.")

    # --- Tab 2: Dashboard ---
    with tab2:
        st.header("📊 Dashboard")

        if st.session_state.processed_data:
            df = pd.DataFrame(st.session_state.processed_data)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Total", len(df))
            with col2:
                st.metric("⭐ Avg Score", f"{df['recognition_score'].mean():.2f}")
            with col3:
                pos_pct = (df['sentiment_label'] == 'POSITIVE').sum() / len(df) * 100
                st.metric("😊 Positive", f"{pos_pct:.0f}%")
            with col4:
                total_officers = sum(len(o) for o in df['extracted_officers'])
                st.metric("👮 Officers", total_officers)

            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🏆 Top Officers Mentioned")
                all_officers = [o for officers in df['extracted_officers'] for o in officers if isinstance(o, str) and not o.startswith("(")]
                if all_officers:
                    st.bar_chart(pd.Series(all_officers).value_counts().head(10))
                else:
                    st.info("No officers extracted yet.")
            with col2:
                st.subheader("📊 Top Crime Types Mentioned")
                all_crimes = [c for crimes in df['extracted_crime_types'] for c in crimes if isinstance(c, str)]
                if all_crimes:
                    st.bar_chart(pd.Series(all_crimes).value_counts().head(10))
                else:
                    st.info("No crime types extracted yet.")

            st.markdown("---")
            st.subheader("📜 Recent Submissions")
            df_display = df[[
                "summary",
                "recognition_score",
                "sentiment_label",
                "extracted_officers",
                "extracted_locations",
                "extracted_crime_types"
            ]].tail(5).iloc[::-1]
            st.dataframe(df_display, use_container_width=True)
        else:
            st.info("ℹ️ No data processed yet. Please analyze feedback in the first tab.")

    # --- Tab 3: Q&A ---
    with tab3:
        st.header("💬 Q&A Chat")

        if st.session_state.processed_data:
            st.markdown("""
            <div class="info-box">
                Ask questions about the feedback you have analyzed. The AI will search
                through all submitted texts to find the answer.
            </div>
            """, unsafe_allow_html=True)

            all_texts = " ".join([d['original_text'] for d in st.session_state.processed_data])
            question = st.text_input(
                "💭 Ask your question:",
                placeholder="Example: What acts of bravery were mentioned?",
                key="qa_q"
            )

            if st.button("🔍 Get Answer", type="primary"):
                if question and all_texts:
                    with st.spinner("🤔 Searching for the answer..."):
                        answer = answer_question(question, all_texts, qa_model)
                        st.session_state.chat_history.append({"q": question, "a": answer})
                        st.experimental_rerun()
                elif not all_texts:
                    st.warning("Please process some feedback first.")
                else:
                    st.warning("Please enter a question.")

            if st.session_state.chat_history:
                st.markdown("---")
                st.subheader("💬 Chat History")
                for i, chat in enumerate(reversed(st.session_state.chat_history)):
                    st.markdown(f"""
                    <div class="info-box">
                        <b>❓ You:</b> {chat['q']}
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="success-box">
                        <b>✅ Answer:</b> {chat['a']}
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("---")
        else:
            st.warning("ℹ️ Please process feedback in the 'Process Feedback' tab before using Q&A.")

    # --- Tab 4: Export ---
    with tab4:
        st.header("📈 Export Data")

        if st.session_state.processed_data:
            df = pd.DataFrame(st.session_state.processed_data)
            st.subheader("📊 Data Table")
            all_cols = df.columns.tolist()
            default_cols = [c for c in [
                'timestamp',
                'recognition_score',
                'sentiment_label',
                'extracted_officers',
                'extracted_locations',
                'extracted_crime_types',
                'summary'
            ] if c in all_cols]
            selected = st.multiselect("Select columns to display:", all_cols, default=default_cols)
            if selected:
                st.dataframe(df[selected], use_container_width=True, height=400)

            st.markdown("---")
            st.subheader("📥 Bulk Export")
            col1, col2 = st.columns(2)
            with col1:
                csv_df = df.copy()
                # Convert list columns to strings for CSV export
                for col in csv_df.columns:
                    csv_df[col] = csv_df[col].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
                csv_data = csv_df.to_csv(index=False)
                st.download_button(
                    "📄 CSV",
                    data=csv_data,
                    file_name=f"bulk_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col2:
                json_data = df.to_json(orient='records', indent=2, force_ascii=False)
                st.download_button(
                    "📋 JSON",
                    data=json_data,
                    file_name=f"bulk_data_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        else:
            st.info("ℹ️ No data available to export. Please process feedback in the first tab.")

if __name__ == "__main__":
    main()
