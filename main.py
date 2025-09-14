from PIL import Image
import cv2
import numpy as np
from deskew import determine_skew
from transformers import pipeline
from paddleocr import PPStructureV3
from image_preprocessing import preprocess_image

SAMPLES_FOLDER_PATH = "./decentrathon_samples"

# Path to your image
image = f"{SAMPLES_FOLDER_PATH}/beta2.jpg" 

class Preprocessor:
    def __init__(self):
        self.paddleocr_pipeline = PPStructureV3(use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang="ru")
    
    def preprocess(self, image_path):
        preprocessed_image = preprocess_image(image_path)
        return preprocessed_image
    
    def predict(self, image_path):
        result = self.paddleocr_pipeline.predict(input=image_path)
        return result
    
    def postprocess(self, result):
        # Put here Gemini postprocessing
        return result
    
    def run(self, image_path):
        preprocessed_image = self.preprocess(image_path)
        result = self.predict(preprocessed_image)
        return self.postprocess(result)

# Preprocess the image
preprocessed_image = Preprocessor().run(image)

# Save the preprocessed image to a file (Optional, for inspection)
output_image_path = "preprocessed_image.jpg"
cv2.imwrite(output_image_path, preprocessed_image)

# Donut (pre-trained)
# https://huggingface.co/Akajackson/donut_rus/blob/1f359e48057fb5ead3be17da5d92eae879d3f637/README.md

# donut_pipe = pipeline("image-to-text", model="naver-clova-ix/donut-base-finetuned-cord-v2")

# PaddleOCR  !!! PPStructureV3 !!!
paddleocr_pipeline = PPStructureV3(use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang="ru")

# Use the preprocessed image directly
result = paddleocr_pipeline.predict(input=output_image_path)

for res in result:
    res.print()
    res.save_to_json(save_path="result")

# Gemini (Post-processing)
# gemini = Gemini(model_name="gemini-2.0-flash")