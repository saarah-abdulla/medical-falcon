import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from transformers import BitsAndBytesConfig


# ✅ Load fine-tuned model & tokenizer
MODEL_NAME = "saarah-a/falcon-finetuned"


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

@st.cache_resource  # Cache model to avoid reloading on every run
def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="cpu"
        torch_dtype=torch.float16, offload_folder = 'offload_dir',
        quantization_config=bnb_config
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer

model, tokenizer = load_model()

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
