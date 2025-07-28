import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from transformers import BitsAndBytesConfig
from peft import PeftModel, PeftConfig

BASE_MODEL = "tiiuae/falcon-7b-instruct"
ADAPTER_REPO = "saarah-a/falcon-finetuned"

# Cache model to avoid reloading on every run
@st.cache_resource
def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_REPO)
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_REPO)

    return model, tokenizer

# Avoid eager loading at the module level
if "model" not in st.session_state:
    model, tokenizer = load_model()
    st.session_state.model = model
    st.session_state.tokenizer = tokenizer
    st.session_state.text_gen = pipeline("text-generation", model=model, tokenizer=tokenizer)

# UI and interaction logic
st.title("🦅 Falcon3-7B Medical Chatbot")

user_input = st.text_area("Enter your prompt:")

if st.button("Generate"):
    if user_input:
        with st.spinner("Generating response..."):
            response = st.session_state.text_gen(
                user_input,
                max_length=150,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                return_full_text=False,
                eos_token_id=st.session_state.tokenizer.eos_token_id,
                early_stopping=True
            )
            st.success(response[0]["generated_text"])
    else:
        st.warning("Please enter a prompt!")
