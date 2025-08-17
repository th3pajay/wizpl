import streamlit as st

from utils import (
    load_vit_model,
    zpl_to_image,
    rotate_image,
    describe_image,
    optimize_zpl,
    fix_and_lint_zpl,
    extract_data_from_zpl,
)


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

    if st.button("Generate / Analyze Image", key="generate_image_button"):
        image = zpl_to_image(zpl_code, dpi, width, height)
        if image:
            rotated_image = rotate_image(image, rotation_angle)
            st.session_state.generated_image = rotated_image
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
