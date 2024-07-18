
import streamlit as st
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# Load the PEGASUS model and tokenizer
model_name = 'google/pegasus-cnn_dailymail'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

def summarize_text(text):
    inputs = tokenizer(text, max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = model.generate(inputs['input_ids'], max_length=55, min_length=5, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

st.title("Text Summarization using PEGASUS")
user_input = st.text_area("Enter the text to summarize")
if st.button("Summarize"):
    summary = summarize_text(user_input)
    st.write("Summary:")
    st.write(summary)
