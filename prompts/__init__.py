"""
Prompt library for AI-powered transcript processing

This module contains all the prompts used for:
- Transcript cleaning and formatting
- Transcript summarization to interview questions
- Future prompt templates

Usage:
    from prompts.transcript_cleaning import get_cleaning_system_prompt
    from prompts.transcript_summarization import get_summarization_system_prompt
"""

from .transcript_cleaning import (
    get_cleaning_system_prompt,
    get_cleaning_user_prompt,
    build_cleaning_instructions
)

from .transcript_summarization import (
    get_summarization_system_prompt,
    get_summarization_user_prompt
)

from .quote_extraction import (
    get_quote_extraction_system_prompt,
    get_quote_extraction_user_prompt,
    parse_themes_input
)

__all__ = [
    'get_cleaning_system_prompt',
    'get_cleaning_user_prompt', 
    'build_cleaning_instructions',
    'get_summarization_system_prompt',
    'get_summarization_user_prompt',
    'get_quote_extraction_system_prompt',
    'get_quote_extraction_user_prompt',
    'parse_themes_input'
]