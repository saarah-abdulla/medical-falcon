import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open

# Model repository
model_repo = "saarah-a/falcon-finetuned"

# Load model and tokenizer from Hugging Face
try:
    # Load the tokenizer from the Hugging Face model repo
    tokenizer = AutoTokenizer.from_pretrained(model_repo)
    
    # Load the model from the Hugging Face model repo
    model = AutoModelForCausalLM.from_pretrained(model_repo)

except Exception as e:
    st.error(f"Error loading model: {e}")

st.title("Falcon Fine-Tuned Model Inference")

# Example inference
user_input = st.text_input("Enter your prompt:")
if user_input:
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model.generate(**inputs)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.write(f"Generated Text: {result}")
