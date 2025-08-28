import os
import io
import re
import requests
import streamlit as st
from PIL import Image
from transformers import ViTImageProcessor, ViTModel
import easyocr
import numpy as np
from pyzbar.pyzbar import decode

from constants import LABELARY_API_URL


@st.cache_resource
def load_vit_model():
    """Loads a ViT model and processor from Hugging Face Hub.

    Returns:
        tuple: A tuple containing the ViT model and processor.
    """
    model_id = "google/vit-base-patch16-224-in21k"
    model = ViTModel.from_pretrained(model_id)
    processor = ViTImageProcessor.from_pretrained(model_id)
    return model, processor


@st.cache_resource
def load_easyocr_model():
    """Loads an easyocr model.

    Returns:
        easyocr.Reader: An easyocr.Reader object.
    """
    return easyocr.Reader(["en"])


def zpl_to_image(zpl_code, dpi=8, width=4, height=6):
    """Converts ZPL code to an image using the Labelary API.

    Args:
        zpl_code (str): The ZPL code to convert.
        dpi (int, optional): The DPI (dots per inch) for the image. Defaults to 8.
        width (int, optional): The width of the label in inches. Defaults to 4.
        height (int, optional): The height of the label in inches. Defaults to 6.

    Returns:
        PIL.Image.Image: A PIL Image object representing the converted ZPL code, or None if an error occurred or the ZPL code is empty.
    """
    if not zpl_code.strip():
        st.warning("ZPL code is empty. Please enter valid ZPL code.")
        return None
    try:
        url = f"{LABELARY_API_URL}/{dpi}dpmm/labels/{width}x{height}/0/"
        headers = {"Accept": "image/png"}
        response = requests.post(url, headers=headers, data=zpl_code.encode("utf-8"))
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        st.error(f"Error from Labelary API: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Error converting ZPL to image: {e}")
    return None


def rotate_image(image, angle):
    """Rotates an image by a specified angle.

    Args:
        image: The image to rotate (PIL Image object).
        angle: The rotation angle in degrees.

    Returns:
        A new PIL Image object representing the rotated image.
    """
    return image.rotate(angle, expand=True)


def extract_text_from_image(image):
    """Extracts text from an image using easyocr.

    Args:
        image: A PIL Image object.

    Returns:
        str: The extracted text from the image.
    """
    reader = load_easyocr_model()
    image_np = np.array(image)
    result = reader.readtext(image_np)
    text = " ".join([res[1] for res in result])
    return text.strip()


def detect_barcodes(image):
    """Detects barcodes in an image using pyzbar.

    Args:
        image: A PIL Image object.

    Returns:
        list: A list of decoded barcodes.
    """
    return decode(image)


def describe_zpl_label_elements(image):
    """Describes the elements of a ZPL label based on image analysis.

    Args:
        image: A PIL Image object representing the ZPL label.

    Returns:
        str: A description of the ZPL label elements, including any text detected.
    """
    description = []
    text = extract_text_from_image(image)
    if text:
        description.append(f"Text detected on the label:\n\n{text}")

    barcodes = detect_barcodes(image)
    if barcodes:
        description.append("Barcodes detected:")
        for barcode in barcodes:
            description.append(f"- {barcode.type}: {barcode.data.decode('utf-8')}")

    return "\n".join(description)


def describe_image(image, model, processor):
    """Describes an image using a vision model and ZPL label element analysis.

    Args:
        image: A PIL Image object.
        model: The vision model.
        processor: The image processor.

    Returns:
        A string containing the vision model output and ZPL label element description.
    """
    description = describe_zpl_label_elements(image)
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        pooled_output = outputs.pooler_output
        output_text = f"Vision Model Output (pooled): {pooled_output.shape}"
        return f"{output_text}\n\n{description}"
    except Exception as e:
        st.error(f"Error describing image: {e}")
        return description


def optimize_zpl(zpl_code: str) -> str:
    """Optimize ZPL code by removing unnecessary whitespace and redundant commands.

    Args:
        zpl_code: The ZPL code to optimize.

    Returns:
        The optimized ZPL code.
    """
    # Remove comments
    zpl_code = re.sub(r"\^FX.*", "", zpl_code)

    # Remove unnecessary whitespace
    lines = filter(None, map(str.strip, zpl_code.split("\n")))
    optimized_lines = [re.sub(r"\s+", " ", line) for line in lines]
    optimized_code = "\n".join(optimized_lines)

    # Remove redundant commands (example: consecutive ^FS)
    optimized_code = re.sub(r"(\^FS\s*){2,}", "^FS", optimized_code)

    return optimized_code


def fix_and_lint_zpl(zpl_code):
    """
    Cleans, fixes, and lints ZPL code.

    It removes invalid characters, ensures the code starts with ^XA and ends with ^XZ,
    and uses Labelary's API to lint the code, displaying any warnings or errors.
    """
    cleaned_code = re.sub(r"[^A-Za-z0-9^\n:,;._ ]", "", zpl_code)
    commands = cleaned_code.split("\n")
    fixed_commands = [cmd.strip() for cmd in commands if cmd.strip().startswith("^")]
    fixed_code = "\n".join(fixed_commands)
    if not fixed_code.startswith("^XA"):
        fixed_code = "^XA\n" + fixed_code
    if not fixed_code.endswith("^XZ"):
        fixed_code += "\n^XZ"
    try:
        url = f"{LABELARY_API_URL}/8dpmm/labels/4x6/0/"
        headers = {"X-Linter": "On", "Accept": "application/json"}
        response = requests.post(url, headers=headers, data=fixed_code.encode("utf-8"))
        if response.status_code == 200:
            warnings = response.headers.get("X-Warnings", "")
            if warnings:
                st.warning(f"Linting Warnings: {warnings}")
        else:
            st.error(f"Linting failed with status: {response.status_code}")
    except Exception as e:
        st.error(f"Error during linting: {e}")
    return fixed_code


def extract_data_from_zpl(zpl_code):
    """
    Extracts data from ZPL code using the Labelary API.

    Args:
        zpl_code: The ZPL code to extract data from.

    Returns:
        A JSON object containing the extracted data, or None if an error occurred.
    """
    try:
        url = f"{LABELARY_API_URL}/8dpmm/labels/4x6/0/"
        headers = {"Accept": "application/json"}
        response = requests.post(url, headers=headers, data=zpl_code.encode("utf-8"))
        if response.status_code == 200:
            return response.json()
        st.error(f"Error from Labelary API: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Error extracting data from ZPL: {e}")
    return None
