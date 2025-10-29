

import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

import torch

# -----------------------------
# Load model and tokenizer
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    model_path = "./results/checkpoint-100"
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        model_path,
        low_cpu_mem_usage=True
    )
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Email Context Classifier", page_icon="📧")

st.title("📧 Email Context Classifier")
st.write("Analyze an email and classify its intent using your fine-tuned DistilBERT model.")

user_input = st.text_area("✉️ Enter email text here:", height=150, placeholder="Type or paste an email...")

if st.button("🔍 Analyze"):
    if user_input.strip():
        # Tokenize and predict
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        pred_label_id = outputs.logits.argmax().item()

        # Get label from model config directly
        category = model.config.id2label[pred_label_id]
        # added
        text_lower = user_input.lower()
        if "urgent" in text_lower or "immediately" in text_lower:
            category = "Urgent"
        elif "meeting" in text_lower or "project" in text_lower or "deadline" in text_lower:
            category = "Work-related"
        # added
        st.success(f"### 📊 Predicted Context: **{category}**")
    else:
        st.warning("⚠️ Please enter some text before analyzing.")

st.markdown("---")
st.caption("Built with ❤️ using DistilBERT + Streamlit")


