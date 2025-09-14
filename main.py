import cv2
from classes import Preprocessor, Postprocessor, DonutOCR

SAMPLES_FOLDER_PATH = "./decentrathon_samples"
image = f"{SAMPLES_FOLDER_PATH}/beta.jpg" 



# Preprocessing
preprocessor = Preprocessor()
# preprocessed_image = preprocessor.preprocess(image)

# Save for inspection
# output_image_path = "preprocessed_image.jpg"
# cv2.imwrite(output_image_path, preprocessed_image)

# PaddleOCR: extract text and structure
# The code snippet `paddle_result = preprocessor.predict(image)` is using a preprocessor object to
# predict and extract text and structure from the given image. The result is stored in the
# `paddle_result` variable, which likely contains information about the text and structure found in
# the image.
paddle_result = preprocessor.predict(image)

print(str(paddle_result) + "\n\n")

for res in paddle_result:
    # idk how to extract text only
    print(str(res['parsing_res_list']) + "\n\n")
    res.save_to_json("output")
    res.save_to_markdown("output")
            
# Does not work well for now
# Donut: ask a question for understanding
# donut = DonutOCR()
# question = "Какая дата указана в документе?"
# donut_result = donut.predict(output_image_path, question)

# with open("log_gemini_input.txt", "w") as f:
#     f.write(str(donut_result))

# Imlpement a post-processing
# Gemini (Post-processing)
# postprocessor = Postprocessor(model_name="gemini-2.0-flash")
# final_output = postprocessor.postprocess(combined_input)

# print("Final processed output:", donut_result)

#Postprocessing
postprocessor = Postprocessor(model_name="gemini-2.5-flash", system_instruction_file="postprocess_system_instruction.md")
with open("./output/beta.md", "r") as f:
    final_output = postprocessor.postprocess(f.read())
    print('===== FINAL OUTPUT =====')
    print(final_output)