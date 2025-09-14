import cv2
import streamlit as st
import os
import time
import zipfile
import io
from classes import Preprocessor, Postprocessor, DonutOCR, StreamlitInterface

# Upload multiple files
uploaded_files = st.file_uploader("Upload images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if not uploaded_files:
    st.info("Please upload at least one image")
    st.stop()

preprocessor = Preprocessor()

results = []  # store all results (filename, content)

for uploaded_file in uploaded_files:
    # Save temporarily
    image_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info(f"Processing file: {uploaded_file.name}")

    # PaddleOCR
    paddle_result = preprocessor.predict(image_path)
    print(str(paddle_result) + "\n\n")

    for res in paddle_result:
        print(str(res['parsing_res_list']) + "\n\n")
        res.save_to_json("output")
        res.save_to_markdown("output")

    # Check if file exists
    file_path = "./output/uploaded.md"
    if not os.path.exists(file_path):
        st.warning(f"⚠️ Result file not found for {uploaded_file.name}")
        continue

    with st.spinner(f'Postprocessing: {uploaded_file.name}'):
        try:
            postprocessor = Postprocessor(
                model_name="gemini-2.5-flash",
                system_instruction_file="postprocess_system_instruction.md"
            )
            with open(file_path, "r", encoding="utf-8") as f:
                final_output = postprocessor.postprocess(f.read())
            
            results.append((uploaded_file.name, final_output))
            st.success(f"✅ {uploaded_file.name} processed")

        except Exception as e:
            st.error(f"Error during postprocessing {uploaded_file.name}: {str(e)}")

# Display results and build ZIP
if results:
    st.title("📄 Processing results")

    # Create an in-memory ZIP file
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for fname, output in results:
            md_filename = f"{os.path.splitext(fname)[0]}.md"
            zipf.writestr(md_filename, output)

    zip_buffer.seek(0)

    # Download button for all results in one ZIP
    st.download_button(
        label="⬇️ Download all results as ZIP",
        data=zip_buffer,
        file_name="results.zip",
        mime="application/zip"
    )

    # Also show results inline
    for fname, output in results:
        st.subheader(fname)
        st.markdown("---")
        st.markdown(output)
