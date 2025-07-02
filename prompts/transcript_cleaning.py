"""
Prompts for transcript cleaning functionality
"""

def get_cleaning_system_prompt(instructions):
    """
    Get the system prompt for transcript cleaning
    
    Args:
        instructions (str): Specific cleaning instructions based on user options
    
    Returns:
        str: Complete system prompt for transcript cleaning
    """
    return f"""You are a professional transcript editor. Your task is to clean and improve transcripts while maintaining the original meaning and speaker intent.

Instructions: {instructions}

Guidelines:
- Preserve the original meaning and context
- Maintain the speaker's voice and tone
- Keep important technical terms and proper nouns
- If there are multiple speakers, maintain speaker identification
- Do not add new information or change facts
- Focus on clarity and readability

Return only the cleaned transcript without any additional commentary."""

def get_cleaning_user_prompt(raw_transcript):
    """
    Get the user prompt for transcript cleaning
    
    Args:
        raw_transcript (str): The raw transcript to be cleaned
    
    Returns:
        str: User prompt for cleaning request
    """
    return f"Please clean this transcript:\n\n{raw_transcript}"

def build_cleaning_instructions(cleaning_options):
    """
    Build cleaning instructions based on user options
    
    Args:
        cleaning_options (dict): Dictionary of cleaning options
    
    Returns:
        str: Formatted instructions string
    """
    instructions = []
    
    if cleaning_options.get('grammar', True):
        instructions.append("Fix grammar and sentence structure")
    if cleaning_options.get('punctuation', True):
        instructions.append("Correct punctuation and capitalization")
    if cleaning_options.get('formatting', True):
        instructions.append("Format the text with proper paragraphs and structure")
    if cleaning_options.get('business_sanitization', True):
        instructions.append("Remove filler words (um, uh, like, you know) and false starts")
        instructions.append("Make the language more professional and business-appropriate")
    if cleaning_options.get('remove_repetition', True):
        instructions.append("Remove unnecessary repetitions and redundant phrases")
    
    return ". ".join(instructions)