import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

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
            print(f"'{model_name}' successfully initialized.")

        except Exception as e:

            print(f"error: {e}")
            self.model = None

    def postprocess(self, input: str) -> str:

        if not self.model:

            return "Model is not initialized."
            
        try:

            response = self.model.generate_content(input)
            return response.text
        
        except Exception as e:

            return f"error while processing: {e}"

