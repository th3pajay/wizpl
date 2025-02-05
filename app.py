import os
import io

import re
import requests
import streamlit as st
from PIL import Image
from transformers import ViTImageProcessor, ViTModel
import easyocr
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


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
        url = (
            f"http://api.labelary.com/v1/printers/{dpi}dpmm/labels/{width}x{height}/0/"
        )
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
    reader = easyocr.Reader(["en"])
    image_np = np.array(image)
    result = reader.readtext(image_np)
    text = " ".join([res[1] for res in result])
    return text.strip()


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
    temp_image_path = "temp_image.png"
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(temp_image_path)
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        output_text = (
            f"Vision Model Output (last hidden states): {last_hidden_states.shape}"
        )
        return f"{output_text}\n\n{description}"
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)


def optimize_zpl(zpl_code: str) -> str:
    """Optimize ZPL code by removing unnecessary whitespace.

    Args:
        zpl_code: The ZPL code to optimize.

    Returns:
        The optimized ZPL code.
    """
    lines = filter(None, map(str.strip, zpl_code.split("\n")))
    optimized_lines = [re.sub(r"\s+", " ", line) for line in lines]
    optimized_code = "\n".join(optimized_lines)
    return re.sub(r"(\^FO\d+,\d+)\s+(\^FD.*?)(\^FS)", r"\1 \2 \3", optimized_code)


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
        url = "http://api.labelary.com/v1/printers/8dpmm/labels/4x6/0/"
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
        url = "http://api.labelary.com/v1/printers/8dpmm/labels/4x6/0/"
        headers = {"Accept": "application/json"}
        response = requests.post(url, headers=headers, data=zpl_code.encode("utf-8"))
        if response.status_code == 200:
            return response.json()
        st.error(f"Error from Labelary API: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Error extracting data from ZPL: {e}")
    return None


def main():
    """Main function to run the ZPL to Image application using Streamlit."""
    st.set_page_config(page_title="wizpl")
    st.title("ZPL to Image")

    with st.sidebar:
        st.header("Settings")
        dpi = st.selectbox("DPI (dots per inch)", options=[6, 8, 12, 24], index=1)
        width = st.number_input(
            "Label Width (inches)", min_value=1.0, max_value=24.0, value=4.0, step=0.1
        )
        height = st.number_input(
            "Label Height (inches)", min_value=1.0, max_value=24.0, value=6.0, step=0.1
        )
        rotation_angle = st.selectbox(
            "Rotation Angle (degrees)",
            options=[0, 45, 90, 135, 180, 225, 270, 315],
            index=0,
        )
        st.header("Enter ZPL Code")
        zpl_code = st.text_area("ZPL Code:", value="", height=200)
        use_vit = st.radio(
            "Activate AI Vision Model (vit-base-patch16-224-in21k)?",
            options=["Yes", "No"],
            index=1,
        )

        st.header("ZPL Code Operations")
        optimize_button = st.button("Optimize ZPL Code")
        fix_and_lint_button = st.button("Fix and Lint ZPL Code")
        extract_data_button = st.button("Extract Data from ZPL Code")

        if optimize_button:
            optimized_code = optimize_zpl(zpl_code)
            st.text_area("Optimized ZPL Code:", value=optimized_code, height=200)

        if fix_and_lint_button:
            fixed_code = fix_and_lint_zpl(zpl_code)
            st.text_area("Fixed and Linted ZPL Code:", value=fixed_code, height=200)

        if extract_data_button:
            extracted_data = extract_data_from_zpl(zpl_code)
            if extracted_data:
                st.json(extracted_data)

    st.sidebar.markdown(
        """
            ---
            Created by [th3pajay](https://github.com/th3pajay)
            ![UserGIF](https://user-images.githubusercontent.com/74038190/219925470-37670a3b-c3e2-4af7-b468-673c6dd99d16.png)
        """
    )

    model, processor = None, None
    if use_vit == "Yes":
        model, processor = load_vit_model()

    if "generated_image" not in st.session_state:
        st.session_state.generated_image = None
        st.session_state.image_generated = False

    if st.button("Generate / Analyze Image", key="generate_image_button"):
        image = zpl_to_image(zpl_code, dpi, width, height)
        if image:
            rotated_image = rotate_image(image, rotation_angle)
            st.session_state.generated_image = rotated_image
            st.session_state.image_generated = False
            st.image(
                rotated_image, caption="Generated Label Image", use_container_width=True
            )

            if use_vit == "Yes" and model and processor:
                with st.spinner("Analyzing the image..."):
                    description = describe_image(rotated_image, model, processor)
                st.subheader("Image Analysis")
                st.write(description)
        else:
            st.error("Failed to generate the image. Please check your ZPL code.")


if __name__ == "__main__":
    main()
