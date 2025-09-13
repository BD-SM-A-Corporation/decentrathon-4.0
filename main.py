from PIL import Image
from transformers import pipeline
from paddleocr import PPStructureV3

# Donut (pre-trained)
# https://huggingface.co/naver-clova-ix/donut-base-finetuned-cord-v2

image = Image.open("image.png") # Change to the input

donut_pipe = pipeline("image-to-text", model="naver-clova-ix/donut-base-finetuned-cord-v2")

# PaddleOCR  !!! PPStructureV3 !!!
paddleocr_pipeline = PPStructureV3(use_angle_cls=True, lang="ru")

result = paddleocr_pipeline.predict(input=image)

for res in result:
    res.print()
    res.save_to_json(save_path="result")
    res.save_to_markdown(save_path="result")
    


# Gemini (Post-processing)
# gemini = Gemini(model_name="gemini-2.0-flash")



