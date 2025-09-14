import cv2
from classes import Preprocessor, Postprocessor, DonutOCR

SAMPLES_FOLDER_PATH = "./decentrathon_samples"
image = f"{SAMPLES_FOLDER_PATH}/8A16.pdf" 



# Preprocessing
preprocessor = Preprocessor()
# preprocessed_image = preprocessor.preprocess(image)

# Save for inspection
# output_image_path = "preprocessed_image.jpg"
# cv2.imwrite(output_image_path, preprocessed_image)

# PaddleOCR: extract text and structure
paddle_result = preprocessor.predict(image)

for paddle_res in paddle_result:
    print(paddle_res)
    paddle_res.save_to_markdown("result")

# Donut: ask a question for understanding
# donut = DonutOCR()
# question = "Какая дата указана в документе?"
# donut_result = donut.predict(output_image_path, question)

# with open("log_gemini_input.txt", "w") as f:
#     f.write(str(donut_result))

# Gemini (Post-processing)
# postprocessor = Postprocessor(model_name="gemini-2.0-flash")
# final_output = postprocessor.postprocess(combined_input)

# print("Final processed output:", donut_result)