from paddleocr import PPStructureV3
from transformers import pipeline
from image_preprocessing import preprocess_image

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
    
    def postprocess(self, result):
        # Put here Gemini postprocessing
        return result
    
    
class Postprocessor:
    def __init__(self, model_name="gemini-2.0-flash"):
        self.gemini = pipeline("text-generation", model=model_name)
    
    def postprocess(self, result):
        # Put here Gemini postprocessing
        return result

class DonutOCR:
    def __init__(self, model_name="Akajackson/donut_rus"):
        self.pipe = pipeline("document-question-answering", model=model_name)

    def predict(self, image_path, question):
        return self.pipe(image=image_path, question=question)