
import streamlit as st
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# Load the PEGASUS model and tokenizer
model_name = 'google/pegasus-xsum'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

def summarize_text(text, num_sentences):
    # Set the maximum and minimum length based on the number of sentences
    max_length = num_sentences * 15  # Rough estimate: 15 tokens per sentence
    min_length = num_sentences * 10  # Rough estimate: 10 tokens per sentence
    inputs = tokenizer(text, max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = model.generate(inputs['input_ids'], max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

st.title("Text Summarization using PEGASUS")
user_input = st.text_area("Enter the text to summarize")
num_sentences = st.number_input("Enter the number of sentences for the summary", min_value=1, max_value=10, value=3, step=1)

if st.button("Summarize"):
    summary = summarize_text(user_input, num_sentences)
    st.write("Summary:")
    st.write(summary)

