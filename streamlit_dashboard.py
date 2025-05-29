import streamlit as st
import torch
from test import TrigramLanguageModel, decode, stoi, device, block_size

@st.cache_resource
def load_model():
    model = TrigramLanguageModel()
    model.load_state_dict(torch.load('trigram_model.pth', map_location=device))
    model.to(device)
    model.eval()
    return model

def generate_text(model, prompt, max_new_tokens=100):
    # Encode prompt
    idx = torch.tensor([stoi.get(c, 0) for c in prompt], dtype=torch.long, device=device).unsqueeze(0)
    # Generate tokens
    with torch.no_grad():
        generated_idx = model.generate(idx, max_new_tokens=max_new_tokens)[0].tolist()
    # Decode generated tokens
    return decode(generated_idx)

st.title("Trigram Language Model Text Generator")

model = load_model()

prompt = st.text_input("Enter prompt text:", value="")

if st.button("Generate"):
    if prompt.strip() == "":
        st.warning("Please enter a prompt to generate text.")
    else:
        generated_text = generate_text(model, prompt)
        st.subheader("Generated Text")
        st.text_area("", generated_text, height=300)
