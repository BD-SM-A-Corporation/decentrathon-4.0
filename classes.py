from paddleocr import PPStructureV3
from transformers import pipeline
from image_preprocessing import preprocess_image
import google.generativeai as genai
import os
from dotenv import load_dotenv

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