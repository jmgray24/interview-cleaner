"""
Prompts for quote extraction functionality
"""

def get_quote_extraction_system_prompt(themes):
    """
    Get the system prompt for quote extraction
    
    Args:
        themes (str): Themes to focus on when extracting quotes
    
    Returns:
        str: Complete system prompt for quote extraction
    """
    return f"""You are a professional content analyst specializing in extracting meaningful quotes from transcripts. Your task is to identify and extract the most relevant and impactful quotes from the provided transcript based on specific themes.

Themes to focus on: {themes}

Guidelines:
- Extract quotes that directly relate to the specified themes
- Select quotes that are complete thoughts and make sense standalone
- Preserve the exact wording from the original transcript
- Include speaker attribution when available
- Choose quotes that are insightful, memorable, or particularly relevant
- Aim for 3-5 quotes per theme (adjust based on content availability)
- Ensure quotes are substantial enough to be meaningful (avoid single words or very short phrases)
- Maintain context and meaning of the original statement

Format your response as a structured list organized by theme:

**Theme: [Theme Name]**
- "[Exact quote from transcript]" - [Speaker if available]
- "[Another quote]" - [Speaker if available]

**Theme: [Next Theme Name]**
- "[Quote]" - [Speaker if available]

If no relevant quotes are found for a theme, note: "No relevant quotes found for this theme."

Return only the organized quotes without additional commentary."""

def get_quote_extraction_user_prompt(cleaned_transcript):
    """
    Get the user prompt for quote extraction
    
    Args:
        cleaned_transcript (str): The cleaned transcript to extract quotes from
    
    Returns:
        str: User prompt for quote extraction request
    """
    return f"Please extract relevant quotes from this transcript based on the specified themes:\n\n{cleaned_transcript}"

def parse_themes_input(themes_input):
    """
    Parse themes input into a clean list
    
    Args:
        themes_input (str): Raw themes input from user
    
    Returns:
        list: List of cleaned theme strings
    """
    if not themes_input.strip():
        return []
    
    # Split by common separators and clean up
    themes = []
    for separator in ['\n', ',', ';']:
        if separator in themes_input:
            themes = [theme.strip() for theme in themes_input.split(separator)]
            break
    
    # If no separators found, treat as single theme
    if not themes:
        themes = [themes_input.strip()]
    
    # Remove empty themes and duplicates
    themes = list(set([theme for theme in themes if theme.strip()]))
    
    return themes