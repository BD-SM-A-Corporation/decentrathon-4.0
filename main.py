import cv2
import streamlit as st
import os
import time
from classes import Preprocessor, Postprocessor, DonutOCR, StreamlitInterface
# Loading image
streamlit = StreamlitInterface()
image_path = streamlit.get_image_path()

if image_path is None:
        st.info("Пожалуйста, загрузите изображение для обработки")
        st.stop() 
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
st.info("Extraction...")
paddle_result = preprocessor.predict(image_path)

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

#Postprocessing and result
st.title("Results:")

file_path = "./output/uploaded.md"

if not os.path.exists(file_path):
    st.info("Working...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(30):  
        if os.path.exists(file_path):
            progress_bar.progress(100)
            status_text.success("Preprocessing finished...")
            time.sleep(1)
            st.rerun()  
            break
        
        progress_bar.progress((i + 1) * 100 // 30)
        status_text.text(f"Проверка {i + 1}/30... ({(i + 1) * 2} сек)")
        time.sleep(2)
    else:
        st.error("❌ Файл не был создан в течение 60 секунд")
        if st.button("🔄 Попробовать еще раз"):
            st.rerun()
    
    st.stop()

# Файл существует - обрабатываем
with st.spinner('postprocessing...'):
    try:
        postprocessor = Postprocessor(
            model_name="gemini-2.5-flash", 
            system_instruction_file="postprocess_system_instruction.md"
        )
        
        with open(file_path, "r", encoding="utf-8") as f:
            final_output = postprocessor.postprocess(f.read())
        
        st.success("finished!")
        
    except Exception as e:
        st.error(f"error while postprocessing: {str(e)}")
        st.stop()

# Вывод результата
st.subheader("📄 Result:")
st.markdown("---")
st.markdown(final_output)
st.markdown("---")

# Кнопка скачивания
st.download_button(
    label="Download",
    data=final_output,
    file_name="result.md",
    mime="text/markdown"
)