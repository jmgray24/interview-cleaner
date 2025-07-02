# Prompt Library

This folder contains all AI prompts used in the transcript processing application, organized as a reusable library.

## Structure

### `transcript_cleaning.py`
Contains prompts for cleaning and formatting raw transcripts:
- `get_cleaning_system_prompt(instructions)` - Main system prompt for transcript cleaning
- `get_cleaning_user_prompt(raw_transcript)` - User prompt template
- `build_cleaning_instructions(cleaning_options)` - Builds instructions based on user options

### `transcript_summarization.py`
Contains prompts for converting cleaned transcripts to interview questions:
- `get_summarization_system_prompt(guidance_prompt)` - System prompt for question extraction
- `get_summarization_user_prompt(cleaned_transcript)` - User prompt template

### `quote_extraction.py`
Contains prompts for extracting relevant quotes from transcripts by theme:
- `get_quote_extraction_system_prompt(themes)` - System prompt for quote extraction
- `get_quote_extraction_user_prompt(cleaned_transcript)` - User prompt template
- `parse_themes_input(themes_input)` - Helper function to parse user theme input



## Usage

```python
from prompts.transcript_cleaning import get_cleaning_system_prompt, build_cleaning_instructions
from prompts.transcript_summarization import get_summarization_system_prompt
from prompts.quote_extraction import get_quote_extraction_system_prompt, parse_themes_input

# For transcript cleaning
instructions = build_cleaning_instructions(cleaning_options)
system_prompt = get_cleaning_system_prompt(instructions)

# For transcript summarization
system_prompt = get_summarization_system_prompt(guidance_prompt)

# For quote extraction
themes_list = parse_themes_input(themes_input)
system_prompt = get_quote_extraction_system_prompt(", ".join(themes_list))
```

## Benefits

1. **Centralized Management**: All prompts in one location
2. **Reusability**: Easy to reuse prompts across different parts of the app
3. **Version Control**: Track changes to prompts over time
4. **Testing**: Easy to test different prompt variations
5. **Maintainability**: Separate prompt logic from application logic

## Adding New Prompts

When adding new functionality:

1. Create a new file in the `prompts/` folder (e.g., `new_feature.py`)
2. Define prompt functions following the existing pattern
3. Add imports to `__init__.py`
4. Update this README with documentation

## Prompt Engineering Guidelines

- Keep prompts focused and specific
- Use clear, actionable instructions
- Include examples when helpful
- Test prompts with various inputs
- Document expected inputs and outputs