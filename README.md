# Audio Transcription & Processing Suite

A comprehensive Streamlit application that provides AI-powered text processing tools including audio transcription, transcript cleaning, summarization, and quote extraction.

## Features

- **Audio Transcription**: Upload audio files and get AI-powered transcripts with speaker identification
- **Transcript Cleaning**: Clean and format raw transcripts with grammar correction, punctuation, and business sanitization
- **Transcript Summarization**: Convert cleaned transcripts into top 12 interview questions with customizable focus
- **Quote Extraction**: Extract relevant quotes from transcripts organized by user-defined themes
- **Speaker Diarization**: Automatic speaker identification with timestamps
- **Multiple Input Methods**: Upload files or paste text directly
- **Theme-Based Organization**: Organize quotes by specific topics or themes
- **Secure Credentials**: Session-based API key storage (never saved to servers)
- **Prompt Library**: Organized, reusable AI prompts for easy maintenance and customization
- **Download Results**: Get transcripts as JSON, cleaned text, formatted questions, or organized quotes

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Application

```bash
# Option 1: Use the run script (macOS/Linux)
./run.sh

# Option 2: Manual activation and run
source venv/bin/activate
streamlit run app.py

# Option 3: Direct streamlit command (if venv is activated)
streamlit run app.py
```

## How It Works

### Audio Transcription:
1. Enter your OpenAI API key in the sidebar
2. Upload an audio file (MP3, WAV, M4A, FLAC, OGG, WebM)
3. Click "Transcribe Audio" to process with OpenAI Whisper
4. View results with speaker identification and timestamps
5. Download transcript as JSON file

### Transcript Cleaning:
1. Upload a transcript file or paste text directly
2. Select cleaning options (grammar, punctuation, formatting, etc.)
3. Click "Clean Transcript" to process with GPT-4
4. Compare before/after results
5. Download cleaned text or full results

### Transcript Summarization:
1. Upload cleaned transcript or paste text
2. Optionally provide focus guidance (e.g., "emphasize technical skills")
3. Click "Generate Interview Questions" to extract top 12 questions
4. Edit questions as needed
5. Download questions as text or full analysis as JSON

### Quote Extraction:
1. Upload cleaned transcript or paste text
2. Define themes for quote extraction (e.g., "Leadership, Innovation, Challenges")
3. Click "Extract Quotes" to find relevant quotes for each theme
4. Review quotes organized by theme with speaker attribution
5. Download organized quotes as text or full results as JSON



## Transcript Output Format

The generated transcript includes:
- Full text transcription
- Segment-level timestamps with speaker identification
- Word-level timestamps (when available)
- Audio duration and detected language
- Processing metadata

```json
{
  "filename": "audio.mp3",
  "duration": 120.5,
  "language": "en",
  "full_text": "Complete transcription...",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 5.2,
      "text": "Hello, this is speaker one.",
      "speaker": "Speaker_1",
      "confidence": -0.3
    }
  ],
  "word_timestamps": [...],
  "processing_info": {...}
}
```

## Supported Audio Formats

- MP3
- WAV
- M4A
- FLAC
- OGG
- WebM

## Usage Notes

- **Required**: OpenAI API key for all functionality
- Audio processing happens in the Streamlit app using OpenAI's API
- Supports audio files up to Streamlit's default upload limit (200MB)
- Processing time depends on audio length and OpenAI API response time
- Transcript cleaning and summarization use GPT-4 for intelligent text processing
- All AI prompts are organized in a reusable prompt library for easy customization

## Text Size Limits

The application includes built-in size validation to ensure optimal processing:

| Operation | Safe Limit | Maximum Limit | Notes |
|-----------|------------|---------------|-------|
| **Transcript Cleaning** | 20,000 chars | 25,000 chars | ~5,000-6,250 tokens |
| **Transcript Summarization** | 15,000 chars | 20,000 chars | ~3,750-5,000 tokens |
| **Quote Extraction** | 18,000 chars | 23,000 chars | ~4,500-5,750 tokens |

**What happens with large texts:**
- **Within safe limits**: Processing works optimally
- **Between safe and maximum**: Warning displayed, processing may be slower
- **Above maximum**: Processing blocked with suggestion to split content

**Tips for large transcripts:**
- Split long transcripts into logical sections
- Process each section separately
- Combine results manually if needed
- Consider using shorter, focused excerpts for better results

## Security Notes

- API keys are stored only in your browser session
- Keys are never saved to servers or databases
- Audio files are temporarily processed and then cleaned up
- All processing happens through OpenAI's secure API

## Development

### Virtual Environment
The project uses a virtual environment to manage dependencies:
- `venv/` - Virtual environment folder (excluded from git)
- `requirements.txt` - Production dependencies
- `requirements-dev.txt` - Development dependencies
- `run.sh` - Quick start script for macOS/Linux

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run with development tools
black app.py prompts/  # Code formatting
flake8 app.py prompts/ # Linting
mypy app.py           # Type checking
```