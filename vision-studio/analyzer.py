from __future__ import annotations

import base64
import json
import logging
import re
from io import BytesIO
from typing import Any, Dict

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from config import OPENAI_API_KEY, VISION_MODEL

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# LLM Setup
# ─────────────────────────────────────────────
def _get_llm():
    return ChatOpenAI(
        model=VISION_MODEL,   # must be "gpt-4o"
        openai_api_key=OPENAI_API_KEY,
        temperature=0.4,
        max_tokens=1500,
    )

# ─────────────────────────────────────────────
# Image Encoding (FIXED)
# ─────────────────────────────────────────────
def encode_image(image) -> str:
    """Accepts PIL Image OR Streamlit file"""
    
    if not isinstance(image, Image.Image):
        image = Image.open(image)

    # resize
    max_dim = 1024
    if max(image.size) > max_dim:
        image.thumbnail((max_dim, max_dim))

    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)

    return base64.b64encode(buffer.getvalue()).decode()

# ─────────────────────────────────────────────
# CORRECT MULTIMODAL FORMAT (VERY IMPORTANT)
# ─────────────────────────────────────────────
def build_message(image_b64, prompt):
    return [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_b64}"
            },
        },
    ]

# ─────────────────────────────────────────────
# ANALYSIS SYSTEM PROMPT
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert image analyst.
Return ONLY valid JSON:

{
  "caption": "...",
  "description": "...",
  "mood": "...",
  "scene_type": "...",
  "objects": ["..."],
  "dominant_colors": ["..."],
  "time_of_day": "...",
  "style": "..."
}
"""

# ─────────────────────────────────────────────
# MAIN ANALYSIS FUNCTION
# ─────────────────────────────────────────────
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=8))
def analyze_image(image) -> Dict[str, Any]:

    llm = _get_llm()

    image_b64 = encode_image(image)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=build_message(
                image_b64,
                "Analyze this image and return structured JSON."
            )
        ),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()

        # remove ```json ```
        raw = re.sub(r"```json", "", raw)
        raw = re.sub(r"```", "", raw).strip()

        result = json.loads(raw)

        return result

    except Exception as e:
        logger.error("Analysis failed: %s", str(e))

        return {
            "caption": "Image analysis failed",
            "description": "Could not process image",
            "mood": "unknown",
            "scene_type": "other",
            "objects": [],
            "dominant_colors": [],
            "time_of_day": "unknown",
            "style": "unknown",
        }

# ─────────────────────────────────────────────
# STORY GENERATION
# ─────────────────────────────────────────────
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=8))
def generate_story(analysis: Dict[str, Any], image):

    llm = _get_llm()
    image_b64 = encode_image(image)

    context = f"""
Caption: {analysis.get('caption')}
Description: {analysis.get('description')}
Mood: {analysis.get('mood')}
Objects: {analysis.get('objects')}
"""

    messages = [
        SystemMessage(content="Write a short creative story."),
        HumanMessage(
            content=build_message(
                image_b64,
                context + "\nWrite story based on this."
            )
        ),
    ]

    response = llm.invoke(messages)
    return response.content.strip()

# ─────────────────────────────────────────────
# DALLE PROMPT
# ─────────────────────────────────────────────
def build_dalle_prompt(analysis: Dict[str, Any]):

    llm = _get_llm()

    text = f"""
Caption: {analysis.get('caption')}
Description: {analysis.get('description')}
Mood: {analysis.get('mood')}
Create a DALL-E prompt.
"""

    response = llm.invoke([HumanMessage(content=text)])
    return response.content.strip()