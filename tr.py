import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=150)
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

# Set maximum input length
MAX_INPUT_LENGTH = 150

# Prompt user for target language
target_lang = st.selectbox("Select target language", ["en", "fr", "de", "es"])  # Add more languages if needed

# Prompt user for input text
user_input = st.text_area("Enter text to translate", max_chars=MAX_INPUT_LENGTH)

# Perform translation
if st.button("Translate"):
    if user_input:
        # Detect input language
        input_lang = detect(user_input)

        # Tokenize and encode the input text
        input_text = f"{input_lang} to {target_lang}: {user_input}"
        encoded_input = tokenizer.encode(input_text, return_tensors="pt", truncation=True, padding=True, max_length=150)

        # Generate translation
        translation = model.generate(encoded_input, max_length=100)

        # Decode the translated text
        translated_text = tokenizer.decode(translation[0], skip_special_tokens=True)

        # Display the translated text
        st.text_area("Translated Text", value=translated_text, height=200)
    else:
        st.warning("Please enter text to translate.")
