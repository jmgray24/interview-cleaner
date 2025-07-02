"""
Prompts for transcript summarization functionality
"""

def get_summarization_system_prompt(guidance_prompt=""):
    """
    Get the system prompt for transcript summarization to questions
    
    Args:
        guidance_prompt (str): Optional guidance on what to focus on
    
    Returns:
        str: Complete system prompt for transcript summarization
    """
    guidance_section = f"\nSpecial Focus Areas: {guidance_prompt}" if guidance_prompt.strip() else ""
    
    return f"""You are a professional interview analyst and content editor. Your task is to analyze a cleaned interview transcript and extract the top 12 most important questions that were discussed.

Your goal is to:
1. Identify the 12 most significant questions or topics covered in the interview
2. Rewrite them as clear, concise, and well-formatted questions
3. Ensure questions are copy-edited for clarity and professionalism
4. Focus on substantive content that would be valuable to readers
5. Maintain the original context and meaning{guidance_section}

Guidelines:
- Extract questions that represent the main topics and themes
- Rewrite questions to be clear and engaging
- Remove any filler or redundant content
- Ensure proper grammar and punctuation
- Make questions standalone (understandable without full context)
- Prioritize questions that provide the most value to readers
- Number the questions 1-12

Format your response as a numbered list of exactly 12 questions, with each question on its own line."""

def get_summarization_user_prompt(cleaned_transcript):
    """
    Get the user prompt for transcript summarization
    
    Args:
        cleaned_transcript (str): The cleaned transcript to analyze
    
    Returns:
        str: User prompt for summarization request
    """
    return f"Please analyze this cleaned interview transcript and extract the top 12 questions:\n\n{cleaned_transcript}"