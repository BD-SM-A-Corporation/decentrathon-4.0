from classes import Postprocessor
postprocessor = Postprocessor(model_name="gemini-2.5-flash", system_instruction_file="postprocess_system_instruction.md")
with open("./output/beta.md", "r") as f:
    final_output = postprocessor.postprocess(f.read())
    print('===== FINAL OUTPUT =====')
    print(final_output)