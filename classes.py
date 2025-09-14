from paddleocr import PPStructureV3
from transformers import pipeline
from image_preprocessing import preprocess_image
from dotenv import load_dotenv
import google.generativeai as genai
import os
import streamlit as st

load_dotenv()

class Preprocessor:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
         if not hasattr(self, 'initialized'):
            self.paddleocr_pipeline = PPStructureV3(use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                lang="ru")
            self.initialized = True
        
    def preprocess(self, image_path):
        # Now returns 3-channel BGR, suitable for Donut and PaddleOCR
        preprocessed_image = preprocess_image(image_path)
        return preprocessed_image
    
    def predict(self, image_path):
        result = self.paddleocr_pipeline.predict(input=image_path)
        return result
    
class Postprocessor:

    def __init__(self, model_name: str, system_instruction_file: str = ""):

        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY was not found.")
            genai.configure(api_key=api_key)

            with open(system_instruction_file, "r", encoding="utf-8") as f:
                system_instruction = f.read().strip()
            self.model = genai.GenerativeModel(model_name, system_instruction=system_instruction)
            print(f"'{system_instruction}' successfully initialized.")
            print(f"'{model_name}' successfully initialized.")

        except Exception as e:
            print(f"error: {e}")
            self.model = None

    def postprocess(self, input: str) -> str:

        if not self.model:
            return "Postprocess model is not initialized."
            
        try:
            response = self.model.generate_content(input)
            return response.text
        
        except Exception as e:
            return f"error while processing: {e}"



class DonutOCR:
    def __init__(self, model_name="Akajackson/donut_rus"):
        self.pipe = pipeline("document-question-answering", model=model_name)

    def predict(self, image_path, question):
        return self.pipe(image=image_path, question=question)


class StreamlitInterface:
    def __init__(self, save_dir: str = "uploads", fixed_name: str = "uploaded"):
        self.save_dir = save_dir
        self.fixed_name = fixed_name
        os.makedirs(self.save_dir, exist_ok=True)

    def get_image_path(self) -> str | None:
        st.title("Document OCR")
        uploaded_file = st.file_uploader(
            "drag image here or upload using file manager",
            type=["png", "jpg", "jpeg"]
        )

        if uploaded_file is not None:
            _, ext = os.path.splitext(uploaded_file.name)
            if not ext:
                ext = ".jpg"  # fallback, если вдруг без расширения

            save_path = os.path.join(self.save_dir, f"{self.fixed_name}{ext}")

            # сохраняем файл в исходном формате
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.success(f"file saved: {save_path}")
            st.image(save_path, use_column_width=True)

            return save_path

        return None
