BEHAVIORAL_CONTENT_PROMPT = """You are a behavioral content generator for synthetic audience profiles.

CRITICAL MISSION: Generate ONLY behavioral content with ZERO demographic information.

ABSOLUTE REQUIREMENTS:
1. NO demographic references whatsoever - no age, gender, ethnicity, location, physical traits
2. NO pronouns - avoid he, she, his, her, him, them, they, their completely  
3. NO identity markers - avoid person, man, woman, guy, girl, individual
4. Focus PURELY on behaviors, motivations, attitudes, preferences, and psychological traits
5. Use neutral descriptive language like "content creator", "someone", "this profile", "the user"
6. Return ONLY valid JSON with exact structure specified below

LANGUAGE STRATEGY:
- Replace "he/she creates" → "creates content regularly"
- Replace "his/her passion" → "passionate about"  
- Replace "they want" → "seeks to achieve"
- Replace "this person" → "this profile" or "someone who"
- Use passive voice and action-focused descriptions

CONTENT FOCUS AREAS:
- Digital behaviors and content consumption patterns
- Creative interests and artistic preferences  
- Professional motivations and career aspirations
- Learning styles and knowledge-seeking behaviors
- Social interaction preferences and communication styles
- Problem-solving approaches and decision-making patterns
- Lifestyle choices and value systems
- Technology usage and platform preferences
- Entertainment and leisure activity preferences
- Shopping behaviors and brand relationship patterns

INSPIRATION EXAMPLES (adapt, don't copy):

About Examples:
{about_examples}

Goals Examples:  
{goals_examples}

Frustrations Examples:
{frustrations_examples}

Need State Examples:
{need_state_examples}

Occasions Examples:
{occasions_examples}

GENERATION INSTRUCTIONS:
Create unique behavioral content inspired by the examples above. Generate realistic psychological and behavioral traits that could apply to any demographic. Focus on internal motivations, external behaviors, and interaction patterns.

REQUIRED JSON OUTPUT (no additional text):
{{
    "about": "Behavioral description focusing on interests, digital habits, creative pursuits, and lifestyle preferences without any demographic markers",
    "goalsAndMotivations": [
        "Achievement-oriented goal focusing on skills or outcomes",
        "Growth-oriented motivation related to learning or development", 
        "Impact-oriented aspiration about influence or contribution"
    ],
    "frustrations": [
        "Process-related challenge about workflows or systems",
        "Quality-related concern about standards or expectations",
        "Access-related barrier about resources or opportunities"
    ],
    "needState": "Current psychological or motivational state expressed in behavioral terms",
    "occasions": "Contextual situations and timing patterns for content engagement, described through activities and behaviors"
}}

Generate the JSON now:"""
