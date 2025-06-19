# src/tools/writer_tools.py

import logging
import os
from typing import List, Dict, Any
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .reflection_tools import encode_image
from ..config import API_KEY

# Lazily import OpenAI only when needed to keep startup fast
_openai_client = None

def get_openai_client():
    """Lazily initializes and returns the OpenAI client."""
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=API_KEY)
    return _openai_client

class DescribeImagesArgs(BaseModel):
    image_paths: List[str] = Field(description="A list of file paths for ALL the images that need to be described for the final report.")

def describe_images(image_paths: List[str]) -> str:
    """
    Analyzes a list of images and provides a detailed, publication-quality description for each, suitable for figure captions.
    """
    if not image_paths:
        return "No image paths were provided. Cannot generate descriptions."

    descriptions = []
    client = get_openai_client()
    prompt_text = (
        "You are a scientific illustrator creating a figure caption. "
        "For the image provided, describe it in detail for a scientific publication. "
        "Explain what the plot shows, the axes, key trends, and the main takeaway. "
        "Be clear, concise, and objective."
    )

    for i, image_path in enumerate(image_paths):
        if not os.path.exists(image_path):
            logging.warning(f"Image not found at path for description: {image_path}")
            descriptions.append(f"**Figure {i+1} ({os.path.basename(image_path)})**: Error - Image not found at path: {image_path}")
            continue
        try:
            base64_image = encode_image(image_path)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                        ],
                    }
                ],
                max_tokens=500,
            )
            description = response.choices[0].message.content
            descriptions.append(f"**Figure {i+1} ({os.path.basename(image_path)})**:\n{description}")
        except Exception as e:
            logging.error(f"Error describing image {image_path}: {e}")
            descriptions.append(f"**Figure {i+1} ({os.path.basename(image_path)})**: Error analyzing image: {e}")

    return "\n\n---\n\n".join(descriptions)

# --- Tool Definition ---
describe_images_tool = StructuredTool.from_function(
    func=describe_images,
    name="describe_images",
    description="Generates detailed, scientific captions for a list of plot images. This is a mandatory step if any plots were created.",
    args_schema=DescribeImagesArgs,
)