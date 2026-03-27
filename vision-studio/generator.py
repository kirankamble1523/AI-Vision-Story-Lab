"""
generator.py
────────────
DALL-E 3 image generation.

generate_image() takes a text prompt and returns:
  - The URL of the generated image (valid for 60 minutes)
  - The revised prompt that DALL-E actually used

Also includes download_image() to save the generated image locally
so it can be displayed in Streamlit even after the URL expires.
"""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Optional, Tuple

import requests
from openai import OpenAI
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from config import IMAGE_MODEL, IMAGE_QUALITY, IMAGE_SIZE, OPENAI_API_KEY

logger = logging.getLogger(__name__)

# ── OpenAI client ─────────────────────────────────────────────────
_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    return _client


# ═══════════════════════════════════════════════════════════════════
#  Image Generation
# ═══════════════════════════════════════════════════════════════════

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=3, max=15))
def generate_image(prompt: str) -> Tuple[str, str]:
    """
    Call DALL-E 3 to generate an image from a text prompt.

    Parameters
    ----------
    prompt : The text prompt (crafted by build_dalle_prompt() in analyzer.py)

    Returns
    -------
    (image_url, revised_prompt)
      image_url      : Temporary OpenAI CDN URL (valid ~60 min)
      revised_prompt : The prompt DALL-E actually used (may differ from input)
    """
    client = _get_client()

    logger.info("Calling DALL-E 3 with prompt: %s...", prompt[:80])

    response = client.images.generate(
        model=IMAGE_MODEL,
        prompt=prompt,
        size=IMAGE_SIZE,
        quality=IMAGE_QUALITY,
        n=1,                  # DALL-E 3 only supports n=1
        response_format="url",
    )

    image_data = response.data[0]
    url = image_data.url
    revised = image_data.revised_prompt or prompt

    logger.info("DALL-E image generated: %s", url[:60])
    return url, revised


def download_image(url: str) -> Optional[Image.Image]:
    """
    Download the generated image from OpenAI's CDN and return a PIL Image.
    This lets us display it in Streamlit even after the URL expires.
    """
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content))
    except Exception as exc:
        logger.error("Failed to download generated image: %s", exc)
        return None