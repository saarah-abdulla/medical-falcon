import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
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

# ✅ Create a text-generation pipeline
text_gen = pipeline("text-generation", model=model, tokenizer=tokenizer)

st.title("🦅 Falcon-7B Medical Chatbot")

user_input = st.text_area("Enter your prompt:")

if st.button("Generate"):
    if user_input:
        with st.spinner("Generating response..."):
            response = text_gen(user_input, max_length=150, do_sample=True, top_k=50, top_p=0.95,
                                return_full_text=False,eos_token_id=tokenizer.eos_token_id, early_stopping=True)
            st.success(response[0]["generated_text"])
    else:
        st.warning("Please enter a prompt!")
