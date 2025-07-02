import streamlit as st
import json
import tempfile
from datetime import datetime
import os
from dotenv import load_dotenv
from openai import OpenAI
import re

# Import prompt library
from prompts.transcript_cleaning import (
    get_cleaning_system_prompt,
    get_cleaning_user_prompt,
    build_cleaning_instructions
)
from prompts.transcript_summarization import (
    get_summarization_system_prompt,
    get_summarization_user_prompt
)
from prompts.quote_extraction import (
    get_quote_extraction_system_prompt,
    get_quote_extraction_user_prompt,
    parse_themes_input
)


# Load environment variables (fallback for local development)
load_dotenv()

def get_credentials():
    """Get credentials from session state or environment variables"""
    # Initialize session state for credentials
    if 'credentials_configured' not in st.session_state:
        st.session_state.credentials_configured = False
    
    # Try to get from environment first (for local development)
    env_openai_key = os.getenv('OPENAI_API_KEY')
    
    # If environment variables exist, use them and mark as configured
    if env_openai_key:
        st.session_state.openai_api_key = env_openai_key
        st.session_state.credentials_configured = True
        return True
    
    # Otherwise, check session state
    return (hasattr(st.session_state, 'openai_api_key') and 
            st.session_state.openai_api_key and 
            st.session_state.credentials_configured)

def setup_credentials_sidebar():
    """Setup credentials input in sidebar"""
    st.sidebar.title("🔐 Credentials")
    
    # Check if credentials are already configured
    if get_credentials():
        st.sidebar.success("✅ OpenAI API Key Configured")
        
        # Show current configuration (masked)
        if hasattr(st.session_state, 'openai_api_key') and st.session_state.openai_api_key:
            st.sidebar.text(f"OpenAI: sk-...{st.session_state.openai_api_key[-4:]}")
        
        # Option to reconfigure
        if st.sidebar.button("🔄 Reconfigure"):
            st.session_state.credentials_configured = False
            st.rerun()
    else:
        st.sidebar.warning("⚠️ Enter OpenAI API Key")
        
        with st.sidebar.expander("🔑 Enter Your API Key", expanded=True):
            # OpenAI API Key (Required)
            openai_key = st.text_input(
                "OpenAI API Key",
                type="password",
                placeholder="sk-...",
                help="Required for transcription and cleaning"
            )
            
            if st.button("💾 Save API Key", type="primary"):
                if openai_key:
                    # Save to session state
                    st.session_state.openai_api_key = openai_key
                    st.session_state.credentials_configured = True
                    st.sidebar.success("✅ API Key saved!")
                    st.rerun()
                else:
                    st.sidebar.error("❌ OpenAI API Key is required")

def get_openai_client():
    """Get OpenAI client with session credentials"""
    if hasattr(st.session_state, 'openai_api_key') and st.session_state.openai_api_key:
        try:
            return OpenAI(api_key=st.session_state.openai_api_key)
        except TypeError as e:
            if 'proxies' in str(e):
                # Handle version compatibility issue
                st.error("OpenAI library version compatibility issue. Please update to latest version.")
                return None
            raise e
    return None

def estimate_tokens(text):
    """Estimate token count for text (rough approximation)"""
    return len(text) // 4

def check_text_size_limits(text, operation_type="processing"):
    """
    Check if text size is within reasonable limits for OpenAI API
    Returns (is_valid, warning_message, estimated_tokens)
    """
    estimated_tokens = estimate_tokens(text)
    char_count = len(text)
    
    # Define limits based on operation type
    limits = {
        "cleaning": {"safe": 20000, "max": 25000, "output_tokens": 4000},
        "summarization": {"safe": 15000, "max": 20000, "output_tokens": 2000},
        "quote_extraction": {"safe": 18000, "max": 23000, "output_tokens": 3000}
    }
    
    limit_info = limits.get(operation_type, limits["cleaning"])
    safe_chars = limit_info["safe"]
    max_chars = limit_info["max"]
    
    if char_count <= safe_chars:
        return True, None, estimated_tokens
    elif char_count <= max_chars:
        warning = f"⚠️ Large text detected ({char_count:,} characters, ~{estimated_tokens:,} tokens). Processing may be slower or hit API limits."
        return True, warning, estimated_tokens
    else:
        error = f"❌ Text too large ({char_count:,} characters, ~{estimated_tokens:,} tokens). Maximum recommended: {max_chars:,} characters. Consider splitting into smaller sections."
        return False, error, estimated_tokens



def transcribe_audio_with_diarization(audio_file, original_filename):
    """
    Transcribe audio file using OpenAI Whisper API with speaker diarization
    """
    try:
        openai_client = get_openai_client()
        if not openai_client:
            return {'error': 'OpenAI client not configured'}
            
        # Create temporary file from uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=get_file_extension(original_filename)) as temp_file:
            temp_file.write(audio_file.read())
            temp_file_path = temp_file.name
        
        try:
            # First, get basic transcription with timestamps
            with open(temp_file_path, 'rb') as audio_file_obj:
                transcript_response = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file_obj,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"]
                )
            
            # Get detailed transcription with word-level timestamps for better diarization
            with open(temp_file_path, 'rb') as audio_file_obj:
                detailed_response = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file_obj,
                    response_format="verbose_json",
                    timestamp_granularities=["word"]
                )
            
            # Process segments and attempt basic speaker diarization
            segments_with_speakers = process_segments_for_diarization(
                transcript_response.segments,
                detailed_response.words if hasattr(detailed_response, 'words') else []
            )
            
            # Create comprehensive transcript data
            transcript_data = {
                'filename': original_filename,
                'duration': transcript_response.duration,
                'language': transcript_response.language,
                'full_text': transcript_response.text,
                'segments': segments_with_speakers,
                'word_timestamps': detailed_response.words if hasattr(detailed_response, 'words') else [],
                'processing_info': {
                    'model': 'whisper-1',
                    'timestamp_granularity': ['segment', 'word'],
                    'diarization_method': 'gap_based_estimation',
                    'processed_at': datetime.now().isoformat()
                }
            }
            
            return transcript_data
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        st.error(f"Error transcribing audio: {str(e)}")
        return {
            'error': str(e),
            'filename': original_filename,
            'processing_info': {
                'status': 'failed',
                'error_details': str(e)
            }
        }

def process_segments_for_diarization(segments, words):
    """
    Process transcript segments to estimate speaker changes
    This is a simplified diarization based on silence gaps and timing
    """
    processed_segments = []
    current_speaker = 1
    
    for i, segment in enumerate(segments):
        # Simple heuristic: assume speaker change if there's a significant gap
        if i > 0:
            prev_segment = segments[i-1]
            gap_duration = segment['start'] - prev_segment['end']
            
            # If gap is longer than 2 seconds, assume potential speaker change
            if gap_duration > 2.0:
                current_speaker = 2 if current_speaker == 1 else 1
        
        processed_segment = {
            'id': segment.get('id', i),
            'start': segment['start'],
            'end': segment['end'],
            'text': segment['text'],
            'speaker': f"Speaker_{current_speaker}",
            'confidence': segment.get('avg_logprob', 0),
            'duration': segment['end'] - segment['start']
        }
        
        processed_segments.append(processed_segment)
    
    return processed_segments

def get_file_extension(filename):
    """Get file extension from filename"""
    return os.path.splitext(filename)[1]




def extract_quotes_from_transcript(cleaned_transcript, themes_input, filename):
    """
    Extract relevant quotes from transcript based on themes using OpenAI LLM
    """
    try:
        openai_client = get_openai_client()
        if not openai_client:
            return {'error': 'OpenAI client not configured'}
        
        # Parse themes
        themes_list = parse_themes_input(themes_input)
        if not themes_list:
            return {'error': 'No themes provided'}
        
        themes_formatted = ", ".join(themes_list)
        
        # Get prompts from library
        system_prompt = get_quote_extraction_system_prompt(themes_formatted)
        user_prompt = get_quote_extraction_user_prompt(cleaned_transcript)

        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=3000
        )
        
        quotes_text = response.choices[0].message.content.strip()
        
        # Parse quotes by theme
        quotes_by_theme = parse_quotes_response(quotes_text, themes_list)
        
        # Create result data
        result_data = {
            'original_transcript': cleaned_transcript,
            'themes_input': themes_input,
            'themes_list': themes_list,
            'quotes_raw': quotes_text,
            'quotes_by_theme': quotes_by_theme,
            'processing_info': {
                'model': 'gpt-4',
                'processed_at': datetime.now().isoformat(),
                'transcript_length': len(cleaned_transcript),
                'themes_count': len(themes_list),
                'total_quotes': sum(len(quotes) for quotes in quotes_by_theme.values()),
                'filename': filename
            }
        }
        
        return result_data
        
    except Exception as e:
        st.error(f"Error extracting quotes: {str(e)}")
        return {
            'error': str(e),
            'original_transcript': cleaned_transcript,
            'processing_info': {
                'status': 'failed',
                'error_details': str(e)
            }
        }

def parse_quotes_response(quotes_text, themes_list):
    """
    Parse the AI response into structured quotes by theme
    """
    quotes_by_theme = {}
    current_theme = None
    
    lines = quotes_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if line is a theme header
        if line.startswith('**') and line.endswith('**'):
            theme_text = line.replace('**', '').replace('Theme:', '').strip()
            # Find matching theme from original list
            for theme in themes_list:
                if theme.lower() in theme_text.lower() or theme_text.lower() in theme.lower():
                    current_theme = theme
                    if current_theme not in quotes_by_theme:
                        quotes_by_theme[current_theme] = []
                    break
        
        # Check if line is a quote
        elif line.startswith('- "') and current_theme:
            # Extract quote and speaker
            quote_line = line[2:].strip()  # Remove "- "
            if quote_line.startswith('"') and '"' in quote_line[1:]:
                quote_end = quote_line.find('"', 1)
                quote = quote_line[1:quote_end]
                speaker_part = quote_line[quote_end+1:].strip()
                speaker = speaker_part.replace(' - ', '').strip() if speaker_part else "Unknown Speaker"
                
                quotes_by_theme[current_theme].append({
                    'quote': quote,
                    'speaker': speaker
                })
    
    # Ensure all themes have entries (even if empty)
    for theme in themes_list:
        if theme not in quotes_by_theme:
            quotes_by_theme[theme] = []
    
    return quotes_by_theme

def summarize_transcript_to_questions(cleaned_transcript, guidance_prompt, filename):
    """
    Convert cleaned transcript to interview questions using OpenAI LLM
    """
    try:
        openai_client = get_openai_client()
        if not openai_client:
            return {'error': 'OpenAI client not configured'}
        
        # Get prompts from library
        system_prompt = get_summarization_system_prompt(guidance_prompt)
        user_prompt = get_summarization_user_prompt(cleaned_transcript)

        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        questions_text = response.choices[0].message.content.strip()
        
        # Parse questions into a list
        questions_list = []
        for line in questions_text.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('Q')):
                # Remove numbering and clean up
                question = re.sub(r'^\d+\.?\s*', '', line)
                question = re.sub(r'^Q\d*\.?\s*', '', question)
                if question:
                    questions_list.append(question.strip())
        
        # Ensure we have exactly 12 questions
        if len(questions_list) > 12:
            questions_list = questions_list[:12]
        elif len(questions_list) < 12:
            # Pad with placeholder if needed
            while len(questions_list) < 12:
                questions_list.append(f"[Additional question {len(questions_list) + 1} - content may need manual review]")
        
        # Create result data
        result_data = {
            'original_transcript': cleaned_transcript,
            'guidance_prompt': guidance_prompt,
            'questions_raw': questions_text,
            'questions_list': questions_list,
            'processing_info': {
                'model': 'gpt-4',
                'processed_at': datetime.now().isoformat(),
                'transcript_length': len(cleaned_transcript),
                'questions_count': len(questions_list),
                'filename': filename
            }
        }
        
        return result_data
        
    except Exception as e:
        st.error(f"Error summarizing transcript: {str(e)}")
        return {
            'error': str(e),
            'original_transcript': cleaned_transcript,
            'processing_info': {
                'status': 'failed',
                'error_details': str(e)
            }
        }

def clean_transcript_with_llm(raw_transcript, cleaning_options):
    """
    Clean and format transcript using OpenAI LLM
    """
    try:
        openai_client = get_openai_client()
        if not openai_client:
            return {'error': 'OpenAI client not configured'}
            
        # Get prompts from library
        instruction_text = build_cleaning_instructions(cleaning_options)
        system_prompt = get_cleaning_system_prompt(instruction_text)
        user_prompt = get_cleaning_user_prompt(raw_transcript)

        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=4000
        )
        
        cleaned_text = response.choices[0].message.content.strip()
        
        # Create result data
        result_data = {
            'original_text': raw_transcript,
            'cleaned_text': cleaned_text,
            'cleaning_options': cleaning_options,
            'processing_info': {
                'model': 'gpt-4',
                'processed_at': datetime.now().isoformat(),
                'original_length': len(raw_transcript),
                'cleaned_length': len(cleaned_text),
                'reduction_percentage': round((1 - len(cleaned_text) / len(raw_transcript)) * 100, 1) if len(raw_transcript) > 0 else 0
            }
        }
        
        return result_data
        
    except Exception as e:
        st.error(f"Error cleaning transcript: {str(e)}")
        return {
            'error': str(e),
            'original_text': raw_transcript,
            'processing_info': {
                'status': 'failed',
                'error_details': str(e)
            }
        }

def audio_transcription_page():
    """Audio transcription page"""
    st.title("🎙️ Audio Transcription with OpenAI Whisper")
    st.markdown("Upload an audio file and get an AI-powered transcript with speaker identification")
    
    # Status indicator
    if get_openai_client():
        st.success("✅ OpenAI Whisper Ready")
    else:
        st.error("❌ OpenAI Not Configured")
    
    st.divider()
    
    # File upload section
    uploaded_file = st.file_uploader(
        "📁 Choose an audio file", 
        type=['mp3', 'wav', 'm4a', 'flac', 'ogg', 'webm'],
        help="Supported formats: MP3, WAV, M4A, FLAC, OGG, WebM"
    )
    
    if uploaded_file is not None:
        # File info in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Name", uploaded_file.name)
        with col2:
            st.metric("Size", f"{uploaded_file.size / (1024*1024):.1f} MB")
        with col3:
            st.metric("Type", uploaded_file.type.split('/')[-1].upper())
        
        # Action button
        transcribe_btn = st.button("🎯 Transcribe Audio", type="primary", use_container_width=True)
        
        if transcribe_btn:
            uploaded_file.seek(0)
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔄 Processing audio with OpenAI Whisper...")
            progress_bar.progress(25)
            
            # Transcribe audio
            transcript_data = transcribe_audio_with_diarization(uploaded_file, uploaded_file.name)
            progress_bar.progress(75)
            
            if 'error' not in transcript_data:
                progress_bar.progress(100)
                status_text.text("✅ Transcription completed!")
                
                st.divider()
                
                # Results section
                st.subheader("📋 Transcript Results")
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Duration", f"{transcript_data['duration']:.1f}s")
                with col2:
                    st.metric("Language", transcript_data['language'].upper())
                with col3:
                    st.metric("Segments", len(transcript_data['segments']))
                
                # Full transcript
                st.subheader("📝 Full Transcript")
                st.text_area("", transcript_data['full_text'], height=150, label_visibility="collapsed")
                
                # Speaker segments
                st.subheader("Speaker Segments")
                for i, segment in enumerate(transcript_data['segments']):
                    with st.container():
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            st.write(f"**{segment['speaker']}**")
                            st.caption(f"{segment['start']:.1f}s - {segment['end']:.1f}s")
                        with col2:
                            st.write(segment['text'])
                    if i < len(transcript_data['segments']) - 1:
                        st.divider()
                
                st.divider()
                
                # Download action
                transcript_json = json.dumps(transcript_data, indent=2)
                st.download_button(
                    label="📥 Download Transcript (JSON)",
                    data=transcript_json,
                    file_name=f"transcript_{uploaded_file.name}.json",
                    mime="application/json",
                    use_container_width=True
                )
                
            else:
                progress_bar.progress(0)
                status_text.text("❌ Transcription failed")
                st.error("Failed to transcribe audio")
                with st.expander("Error Details"):
                    st.json(transcript_data)



def quote_extraction_page():
    """Quote extraction page"""
    st.title("💬 Quote Extraction")
    st.markdown("Extract relevant quotes from cleaned transcripts based on specific themes")
    
    # Status indicator
    if get_openai_client():
        st.success("✅ OpenAI GPT-4 Ready")
    else:
        st.error("❌ OpenAI Not Configured")
    
    st.divider()
    
    # Input method selection
    input_method = st.radio(
        "📝 Choose input method:",
        ["Paste Cleaned Text", "Upload Cleaned File"],
        horizontal=True
    )
    
    cleaned_transcript = ""
    filename = "extracted_quotes"
    
    if input_method == "Paste Cleaned Text":
        cleaned_transcript = st.text_area(
            "📋 Paste your cleaned transcript here:",
            height=200,
            placeholder="Paste your cleaned transcript here... This should be the output from the transcript cleaning step or any well-formatted interview/conversation transcript."
        )
        filename = "pasted_transcript"
    else:
        uploaded_file = st.file_uploader(
            "📁 Upload a cleaned transcript file",
            type=['txt', 'json'],
            help="Supported formats: TXT, JSON (from cleaning results)"
        )
        
        if uploaded_file is not None:
            filename = uploaded_file.name.split('.')[0]
            try:
                if uploaded_file.type == "application/json":
                    # Handle JSON files from cleaning results
                    json_data = json.loads(uploaded_file.read().decode('utf-8'))
                    if 'cleaned_text' in json_data:
                        cleaned_transcript = json_data['cleaned_text']
                    elif 'full_text' in json_data:
                        cleaned_transcript = json_data['full_text']
                    else:
                        cleaned_transcript = str(json_data)
                else:
                    # Handle text files
                    cleaned_transcript = uploaded_file.read().decode('utf-8')
                
                st.success(f"✅ File loaded: {len(cleaned_transcript)} characters")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    if cleaned_transcript.strip():
        st.divider()
        
        # Themes section
        st.subheader("🎯 Themes for Quote Extraction")
        st.markdown("**Specify the themes you want to extract quotes about:**")
        
        # Quick theme examples
        st.markdown("**Quick Examples:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💼 Business Themes", help="Common business-related themes"):
                st.session_state.themes_example = "Leadership\nInnovation\nTeamwork\nChallenges\nSuccess"
        
        with col2:
            if st.button("🎓 Personal Growth", help="Personal development themes"):
                st.session_state.themes_example = "Learning\nOvercoming obstacles\nCareer growth\nMentorship\nLife lessons"
        
        with col3:
            if st.button("🔧 Technical Topics", help="Technology and technical themes"):
                st.session_state.themes_example = "Problem solving\nTechnical challenges\nInnovation\nBest practices\nFuture trends"
        
        # Get initial value from session state if available
        initial_themes = getattr(st.session_state, 'themes_example', '')
        
        themes_input = st.text_area(
            "Enter themes (one per line or separated by commas):",
            value=initial_themes,
            height=150,
            placeholder="Examples:\nLeadership\nInnovation\nTeamwork\nChallenges\nSuccess\n\nOr: Leadership, Innovation, Teamwork, Challenges, Success",
            help="Enter each theme on a new line or separate with commas. The AI will find quotes related to each theme."
        )
        
        # Clear the session state after using it
        if hasattr(st.session_state, 'themes_example'):
            del st.session_state.themes_example
        
        # Show themes preview if provided
        if themes_input.strip():
            themes_list = parse_themes_input(themes_input)
            st.success(f"✅ {len(themes_list)} themes identified: {', '.join(themes_list)}")
        
        # Preview transcript
        with st.expander("👀 Preview Cleaned Transcript"):
            st.text_area("", cleaned_transcript[:1000] + "..." if len(cleaned_transcript) > 1000 else cleaned_transcript, height=150, label_visibility="collapsed")
        
        # Size validation
        is_valid, size_message, estimated_tokens = check_text_size_limits(cleaned_transcript, "quote_extraction")
        
        if size_message:
            if is_valid:
                st.warning(size_message)
            else:
                st.error(size_message)
        
        # Show text statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Characters", f"{len(cleaned_transcript):,}")
        with col2:
            st.metric("Est. Tokens", f"{estimated_tokens:,}")
        with col3:
            st.metric("Words", f"{len(cleaned_transcript.split()):,}")
        
        # Extract button
        extract_btn = st.button("💬 Extract Quotes", type="primary", use_container_width=True, disabled=not is_valid)
        
        if extract_btn and is_valid:
            if not themes_input.strip():
                st.error("Please provide at least one theme for quote extraction")
                return
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔄 Extracting quotes by theme...")
            progress_bar.progress(25)
            
            # Extract quotes
            result_data = extract_quotes_from_transcript(cleaned_transcript, themes_input, filename)
            progress_bar.progress(75)
            
            if 'error' not in result_data:
                progress_bar.progress(100)
                status_text.text("✅ Quote extraction completed!")
                
                st.divider()
                
                # Results section
                st.subheader("💬 Extracted Quotes by Theme")
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Themes Processed", len(result_data['themes_list']))
                with col2:
                    st.metric("Total Quotes", result_data['processing_info']['total_quotes'])
                with col3:
                    st.metric("Source Length", f"{result_data['processing_info']['transcript_length']:,} chars")
                
                # Display quotes by theme
                quotes_by_theme = result_data['quotes_by_theme']
                all_quotes_text = ""
                
                for theme in result_data['themes_list']:
                    st.subheader(f"🎯 {theme}")
                    quotes = quotes_by_theme.get(theme, [])
                    
                    if quotes:
                        all_quotes_text += f"\n## {theme}\n\n"
                        for i, quote_data in enumerate(quotes, 1):
                            quote = quote_data['quote']
                            speaker = quote_data['speaker']
                            st.write(f"**{i}.** \"{quote}\" - *{speaker}*")
                            all_quotes_text += f"{i}. \"{quote}\" - {speaker}\n"
                        all_quotes_text += "\n"
                    else:
                        st.info("No relevant quotes found for this theme.")
                        all_quotes_text += f"\n## {theme}\n\nNo relevant quotes found for this theme.\n\n"
                    
                    st.divider()
                
                # Show themes used
                st.subheader("🎯 Themes Processed")
                st.info(f"**Themes:** {', '.join(result_data['themes_list'])}")
                
                st.divider()
                
                # Download actions
                col1, col2 = st.columns(2)
                
                with col1:
                    # Download quotes as text
                    st.download_button(
                        label="📥 Download Quotes (TXT)",
                        data=all_quotes_text,
                        file_name=f"quotes_{filename}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                with col2:
                    # Download full results as JSON
                    result_json = json.dumps(result_data, indent=2)
                    st.download_button(
                        label="📥 Download Full Results (JSON)",
                        data=result_json,
                        file_name=f"quote_extraction_{filename}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
            else:
                progress_bar.progress(0)
                status_text.text("❌ Quote extraction failed")
                st.error("Failed to extract quotes")
                with st.expander("Error Details"):
                    st.json(result_data)

def transcript_summarization_page():
    """Transcript summarization page"""
    st.title("📝 Transcript Summarization")
    st.markdown("Convert cleaned transcripts into top 12 interview questions with AI-powered analysis")
    
    # Status indicator
    if get_openai_client():
        st.success("✅ OpenAI GPT-4 Ready")
    else:
        st.error("❌ OpenAI Not Configured")
    
    st.divider()
    
    # Input method selection
    input_method = st.radio(
        "📝 Choose input method:",
        ["Paste Cleaned Text", "Upload Cleaned File"],
        horizontal=True
    )
    
    cleaned_transcript = ""
    filename = "interview_questions"
    
    if input_method == "Paste Cleaned Text":
        cleaned_transcript = st.text_area(
            "📋 Paste your cleaned transcript here:",
            height=200,
            placeholder="Paste your cleaned interview transcript here... This should be the output from the transcript cleaning step."
        )
        filename = "pasted_interview"
    else:
        uploaded_file = st.file_uploader(
            "📁 Upload a cleaned transcript file",
            type=['txt', 'json'],
            help="Supported formats: TXT, JSON (from cleaning results)"
        )
        
        if uploaded_file is not None:
            filename = uploaded_file.name.split('.')[0]
            try:
                if uploaded_file.type == "application/json":
                    # Handle JSON files from cleaning results
                    json_data = json.loads(uploaded_file.read().decode('utf-8'))
                    if 'cleaned_text' in json_data:
                        cleaned_transcript = json_data['cleaned_text']
                    elif 'full_text' in json_data:
                        cleaned_transcript = json_data['full_text']
                    else:
                        cleaned_transcript = str(json_data)
                else:
                    # Handle text files
                    cleaned_transcript = uploaded_file.read().decode('utf-8')
                
                st.success(f"✅ File loaded: {len(cleaned_transcript)} characters")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    if cleaned_transcript.strip():
        st.divider()
        
        # Guidance prompt section - make it more prominent
        st.subheader("🎯 Focus Guidance")
        st.markdown("**Optional:** Provide specific guidance on what aspects to emphasize when extracting questions")
        
        # Quick guidance examples
        st.markdown("**Quick Examples:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔧 Technical Focus", help="Focus on technical skills and problem-solving"):
                st.session_state.guidance_example = "Focus on technical skills, problem-solving approaches, coding challenges, system design decisions, and technical leadership experiences"
        
        with col2:
            if st.button("👥 Leadership Focus", help="Focus on leadership and management"):
                st.session_state.guidance_example = "Emphasize leadership experience, team management, decision-making processes, conflict resolution, and people development strategies"
        
        with col3:
            if st.button("🚀 Career Growth Focus", help="Focus on career development"):
                st.session_state.guidance_example = "Highlight career growth, challenges overcome, lessons learned, professional development, and future aspirations"
        
        # Get initial value from session state if available
        initial_guidance = getattr(st.session_state, 'guidance_example', '')
        
        guidance_prompt = st.text_area(
            "What should the questions focus on?",
            value=initial_guidance,
            height=120,
            placeholder="Examples:\n• Focus on technical skills, leadership experience, and problem-solving approaches\n• Emphasize career growth, challenges faced, and lessons learned\n• Highlight innovation, team collaboration, and strategic thinking\n• Concentrate on industry expertise and future vision",
            help="This guidance will help the AI focus on specific themes when extracting the top 12 questions from your transcript"
        )
        
        # Clear the session state after using it
        if hasattr(st.session_state, 'guidance_example'):
            del st.session_state.guidance_example
        
        # Show guidance preview if provided
        if guidance_prompt.strip():
            st.success(f"✅ Guidance provided: {len(guidance_prompt)} characters")
            with st.expander("👀 Preview Guidance"):
                st.write(guidance_prompt)
        
        # Preview transcript
        with st.expander("👀 Preview Cleaned Transcript"):
            st.text_area("", cleaned_transcript[:1000] + "..." if len(cleaned_transcript) > 1000 else cleaned_transcript, height=150, label_visibility="collapsed")
        
        # Size validation
        is_valid, size_message, estimated_tokens = check_text_size_limits(cleaned_transcript, "summarization")
        
        if size_message:
            if is_valid:
                st.warning(size_message)
            else:
                st.error(size_message)
        
        # Show text statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Characters", f"{len(cleaned_transcript):,}")
        with col2:
            st.metric("Est. Tokens", f"{estimated_tokens:,}")
        with col3:
            st.metric("Words", f"{len(cleaned_transcript.split()):,}")
        
        # Summarize button
        summarize_btn = st.button("📝 Generate Interview Questions", type="primary", use_container_width=True, disabled=not is_valid)
        
        if summarize_btn and is_valid:
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔄 Analyzing transcript and generating questions...")
            progress_bar.progress(25)
            
            # Generate questions
            result_data = summarize_transcript_to_questions(cleaned_transcript, guidance_prompt, filename)
            progress_bar.progress(75)
            
            if 'error' not in result_data:
                progress_bar.progress(100)
                status_text.text("✅ Questions generated successfully!")
                
                st.divider()
                
                # Results section
                st.subheader("❓ Top 12 Interview Questions")
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Questions Generated", len(result_data['questions_list']))
                with col2:
                    st.metric("Source Length", f"{result_data['processing_info']['transcript_length']:,} chars")
                with col3:
                    st.metric("Processing Model", "GPT-4")
                
                # Display questions
                st.subheader("📋 Generated Questions")
                questions_text = ""
                for i, question in enumerate(result_data['questions_list'], 1):
                    st.write(f"**{i}.** {question}")
                    questions_text += f"{i}. {question}\n"
                
                # Editable questions section
                st.subheader("✏️ Edit Questions")
                st.info("💡 You can edit the questions below before downloading")
                
                edited_questions = st.text_area(
                    "Edit the questions as needed:",
                    value=questions_text,
                    height=300,
                    help="Modify, reorder, or rewrite questions as needed"
                )
                
                # Show guidance used
                if guidance_prompt.strip():
                    st.subheader("🎯 Guidance Applied")
                    st.info(f"**Focus Areas Used:** {guidance_prompt}")
                else:
                    st.info("💡 **Tip:** For more targeted questions, try adding focus guidance in your next analysis")
                
                st.divider()
                
                # Download actions
                col1, col2 = st.columns(2)
                
                with col1:
                    # Download questions as text
                    st.download_button(
                        label="📥 Download Questions (TXT)",
                        data=edited_questions,
                        file_name=f"interview_questions_{filename}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                with col2:
                    # Download full results as JSON
                    result_data['final_questions'] = edited_questions
                    result_json = json.dumps(result_data, indent=2)
                    st.download_button(
                        label="📥 Download Full Results (JSON)",
                        data=result_json,
                        file_name=f"questions_analysis_{filename}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
            else:
                progress_bar.progress(0)
                status_text.text("❌ Question generation failed")
                st.error("Failed to generate questions")
                with st.expander("Error Details"):
                    st.json(result_data)

def transcript_cleaning_page():
    """Transcript cleaning page"""
    st.title("✨ Transcript Cleaning with AI")
    st.markdown("Upload or paste raw transcripts and get them professionally cleaned and formatted")
    
    # Status indicator
    if get_openai_client():
        st.success("✅ OpenAI GPT-4 Ready")
    else:
        st.error("❌ OpenAI Not Configured")
    
    st.divider()
    
    # Input method selection
    input_method = st.radio(
        "📝 Choose input method:",
        ["Paste Text", "Upload File"],
        horizontal=True
    )
    
    raw_transcript = ""
    filename = "cleaned_transcript"
    
    if input_method == "Paste Text":
        raw_transcript = st.text_area(
            "📋 Paste your raw transcript here:",
            height=200,
            placeholder="Paste your transcript text here... It can include speaker names, timestamps, filler words, etc."
        )
        filename = "pasted_transcript"
    else:
        uploaded_file = st.file_uploader(
            "📁 Upload a transcript file",
            type=['txt', 'json'],
            help="Supported formats: TXT, JSON"
        )
        
        if uploaded_file is not None:
            filename = uploaded_file.name.split('.')[0]
            try:
                if uploaded_file.type == "application/json":
                    # Handle JSON transcript files
                    json_data = json.loads(uploaded_file.read().decode('utf-8'))
                    if 'full_text' in json_data:
                        raw_transcript = json_data['full_text']
                    elif 'segments' in json_data:
                        # Reconstruct from segments
                        segments_text = []
                        for segment in json_data['segments']:
                            speaker = segment.get('speaker', 'Speaker')
                            text = segment.get('text', '')
                            segments_text.append(f"{speaker}: {text}")
                        raw_transcript = "\n".join(segments_text)
                    else:
                        raw_transcript = str(json_data)
                else:
                    # Handle text files
                    raw_transcript = uploaded_file.read().decode('utf-8')
                
                st.success(f"✅ File loaded: {len(raw_transcript)} characters")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    if raw_transcript.strip():
        # Cleaning options
        st.subheader("🛠️ Cleaning Options")
        
        col1, col2 = st.columns(2)
        with col1:
            grammar = st.checkbox("✏️ Fix Grammar & Sentence Structure", value=True)
            punctuation = st.checkbox("📝 Correct Punctuation & Capitalization", value=True)
            formatting = st.checkbox("📄 Format Paragraphs & Structure", value=True)
        
        with col2:
            business_sanitization = st.checkbox("💼 Business Sanitization (Remove filler words)", value=True)
            remove_repetition = st.checkbox("🔄 Remove Repetitions & Redundancy", value=True)
        
        # Preview original text
        with st.expander("👀 Preview Original Text"):
            st.text_area("", raw_transcript[:1000] + "..." if len(raw_transcript) > 1000 else raw_transcript, height=150, label_visibility="collapsed")
        
        # Size validation
        is_valid, size_message, estimated_tokens = check_text_size_limits(raw_transcript, "cleaning")
        
        if size_message:
            if is_valid:
                st.warning(size_message)
            else:
                st.error(size_message)
        
        # Show text statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Characters", f"{len(raw_transcript):,}")
        with col2:
            st.metric("Est. Tokens", f"{estimated_tokens:,}")
        with col3:
            st.metric("Words", f"{len(raw_transcript.split()):,}")
        
        # Clean button
        clean_btn = st.button("✨ Clean Transcript", type="primary", use_container_width=True, disabled=not is_valid)
        
        if clean_btn and is_valid:
            cleaning_options = {
                'grammar': grammar,
                'punctuation': punctuation,
                'formatting': formatting,
                'business_sanitization': business_sanitization,
                'remove_repetition': remove_repetition
            }
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔄 Cleaning transcript with AI...")
            progress_bar.progress(25)
            
            # Clean transcript
            result_data = clean_transcript_with_llm(raw_transcript, cleaning_options)
            progress_bar.progress(75)
            
            if 'error' not in result_data:
                progress_bar.progress(100)
                status_text.text("✅ Transcript cleaned successfully!")
                
                st.divider()
                
                # Results section
                st.subheader("📋 Cleaning Results")
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Length", f"{result_data['processing_info']['original_length']:,} chars")
                with col2:
                    st.metric("Cleaned Length", f"{result_data['processing_info']['cleaned_length']:,} chars")
                with col3:
                    reduction = result_data['processing_info']['reduction_percentage']
                    st.metric("Reduction", f"{reduction}%", delta=f"-{reduction}%" if reduction > 0 else "No change")
                
                # Cleaned transcript
                st.subheader("✨ Cleaned Transcript")
                cleaned_text = result_data['cleaned_text']
                st.text_area("", cleaned_text, height=300, label_visibility="collapsed")
                
                # Before/After comparison
                with st.expander("🔍 Before/After Comparison"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Before")
                        st.text_area("", raw_transcript[:500] + "..." if len(raw_transcript) > 500 else raw_transcript, height=200, label_visibility="collapsed")
                    with col2:
                        st.subheader("After")
                        st.text_area("", cleaned_text[:500] + "..." if len(cleaned_text) > 500 else cleaned_text, height=200, label_visibility="collapsed")
                
                st.divider()
                
                # Download actions
                col1, col2 = st.columns(2)
                
                with col1:
                    # Download cleaned text
                    st.download_button(
                        label="📥 Download Cleaned Text",
                        data=cleaned_text,
                        file_name=f"cleaned_{filename}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                with col2:
                    # Download full results as JSON
                    result_json = json.dumps(result_data, indent=2)
                    st.download_button(
                        label="📥 Download Full Results (JSON)",
                        data=result_json,
                        file_name=f"cleaning_results_{filename}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
            else:
                progress_bar.progress(0)
                status_text.text("❌ Cleaning failed")
                st.error("Failed to clean transcript")
                with st.expander("Error Details"):
                    st.json(result_data)

def main():
    # Setup credentials in sidebar
    setup_credentials_sidebar()
    
    # Check if credentials are configured
    if not get_credentials():
        st.title("🎙️ Audio Transcription & Processing Suite")
        st.info("👈 Please configure your credentials in the sidebar to get started")
        st.markdown("""
        ### Features Available:
        - **🎙️ Audio Transcription**: Convert audio to text with speaker identification
        - **✨ Transcript Cleaning**: Clean and format raw transcripts
        - **📝 Transcript Summarization**: Extract key questions from interviews
        - **💬 Quote Extraction**: Extract relevant quotes by theme from transcripts
        
        ### Required:
        - **OpenAI API Key**: For all AI-powered text processing
        
        ### Text Size Limits:
        - **Cleaning**: Up to 25,000 characters (~6,250 tokens)
        - **Summarization**: Up to 20,000 characters (~5,000 tokens)  
        - **Quote Extraction**: Up to 23,000 characters (~5,750 tokens)
        
        ### Security Note:
        Your API key is stored only in your browser session and is never saved to our servers.
        """)
        st.stop()
    
    # Sidebar navigation
    st.sidebar.markdown("---")
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.radio(
        "Choose a page:",
        ["🎙️ Audio Transcription", "✨ Transcript Cleaning", "📝 Transcript Summarization", "💬 Quote Extraction"],
        label_visibility="collapsed"
    )
    
    # Page routing
    if page == "🎙️ Audio Transcription":
        audio_transcription_page()
    elif page == "✨ Transcript Cleaning":
        transcript_cleaning_page()
    elif page == "📝 Transcript Summarization":
        transcript_summarization_page()
    elif page == "💬 Quote Extraction":
        quote_extraction_page()


if __name__ == "__main__":
    main()