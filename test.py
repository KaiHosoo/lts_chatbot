import google.generativeai as genai

genai.configure(api_key="AIzaSyCglrNo0xkQjfKwSCuAaB0bfA79qDiCewY")

for m in genai.list_models():
    print(m.name, m.supported_generation_methods)