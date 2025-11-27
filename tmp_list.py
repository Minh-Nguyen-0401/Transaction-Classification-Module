import google.generativeai as genai
from pprint import pprint

genai.configure(api_key="YOUR_API_KEY_HERE")
models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
print('models with generateContent:')
for m in models:
    print(m)
