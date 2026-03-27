"""
vision_agent.py — fixed version using direct OpenAI SDK (no LangChain for vision).
Fixes: RetryError[ValidationError] caused by LangChain multimodal message format.
"""

from __future__ import annotations

import base64
import logging
import os
from io import BytesIO
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

load_dotenv()
logger = logging.getLogger(__name__)


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
    return OpenAI(api_key=api_key)

def get_model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o")

def get_max_tokens() -> int:
    return int(os.getenv("MAX_TOKENS", "1000"))


def image_to_base64(pil_image: Image.Image) -> str:
    max_size = 1024
    if max(pil_image.size) > max_size:
        pil_image.thumbnail((max_size, max_size), Image.LANCZOS)
    if pil_image.mode in ("RGBA", "P"):
        pil_image = pil_image.convert("RGB")
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def call_vision_api(prompt: str, image_b64: str, temperature: float = 0.7, system: Optional[str] = None) -> str:
    """Core function: sends image + text prompt to GPT-4o, returns response string."""
    client = get_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}", "detail": "high"}},
            {"type": "text", "text": prompt},
        ],
    })
    response = client.chat.completions.create(
        model=get_model(),
        messages=messages,
        max_tokens=get_max_tokens(),
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


ANALYSIS_PROMPT = """
Analyze this image thoroughly and return a structured breakdown with these exact sections:

SCENE TYPE: (e.g. landscape, portrait, urban, nature, indoor, abstract)
SUBJECTS: (list the main subjects/objects you see)
DOMINANT COLORS: (list 3-5 main colors with their mood impact)
MOOD & ATMOSPHERE: (describe the emotional tone in 1-2 sentences)
LIGHTING: (e.g. golden hour, harsh midday, soft diffused, artificial)
ESTIMATED LOCATION: (your best guess at where this might be)
PHOTOGRAPHY STYLE: (e.g. candid, posed, street photography, macro, aerial)
INTERESTING DETAILS: (2-3 small or hidden details worth noticing)

Be specific and insightful. This is for a college project.
"""

def analyze_image(pil_image: Image.Image) -> str:
    image_b64 = image_to_base64(pil_image)
    return call_vision_api(ANALYSIS_PROMPT, image_b64, temperature=0.3)


CAPTION_PROMPTS = {
    "instagram": "Write a single punchy Instagram caption with 3-5 hashtags. Max 2 sentences. Authentic, not corporate.",
    "news": "Write a professional one-sentence news-style photo caption. Factual and concise.",
    "poetic": "Write a short poetic caption — 1-2 literary lines, no hashtags. Pure imagery.",
    "funny": "Write a witty, clever one-liner caption. Reddit/Twitter style. Clean and college-appropriate.",
}

def generate_caption(pil_image: Image.Image, style: str = "instagram") -> str:
    image_b64 = image_to_base64(pil_image)
    prompt = CAPTION_PROMPTS.get(style, CAPTION_PROMPTS["instagram"])
    return call_vision_api(prompt, image_b64, temperature=0.8)


STORY_PROMPTS = {
    "adventure": "Write a gripping short adventure story (250-350 words) inspired by this image. Give the protagonist a name. End with a cliffhanger.",
    "mystery": "Write a mysterious suspenseful story (250-350 words) based on this image. First person, present tense. Build dread slowly.",
    "romance": "Write a tender short love story (250-350 words) inspired by this image. Focus on one emotional moment. Show don't tell. End on hope.",
    "scifi": "Reimagine this image 200 years in the future. Write a sci-fi story (250-350 words). Include a surprising twist at the end.",
    "children": "Write a warm children's story (200-300 words) based on this image. Simple language, fun character names, gentle moral lesson.",
}

def generate_story(pil_image: Image.Image, genre: str = "adventure", custom_prompt: Optional[str] = None) -> str:
    image_b64 = image_to_base64(pil_image)
    prompt = f"Look at this image carefully. {custom_prompt}" if custom_prompt else STORY_PROMPTS.get(genre, STORY_PROMPTS["adventure"])
    return call_vision_api(prompt, image_b64, temperature=0.9)


def ask_question(pil_image: Image.Image, question: str) -> str:
    image_b64 = image_to_base64(pil_image)
    system = "You are a helpful visual AI assistant. Answer questions about the uploaded image accurately and concisely (2-4 sentences unless more detail is needed)."
    return call_vision_api(question, image_b64, temperature=0.4, system=system)


def extract_color_palette(pil_image: Image.Image, num_colors: int = 6) -> list[tuple]:
    small = pil_image.copy().convert("RGB")
    small.thumbnail((150, 150))
    quantized = small.quantize(colors=num_colors, method=Image.Quantize.FASTOCTREE)
    palette_data = quantized.getpalette()
    return [(palette_data[i*3], palette_data[i*3+1], palette_data[i*3+2]) for i in range(num_colors)]