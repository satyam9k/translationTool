import streamlit as st
from transformers import pipeline

# Load the translation pipeline
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-auto", tokenizer="Helsinki-NLP/opus-mt-auto")

# Streamlit app
def main():
    st.title("Translator App")
    st.write("Translate text to different languages")

    # User input
    user_input = st.text_area("Enter the text to translate", "", max_chars=150)

    # Detect language and display
    if user_input:
        detected_language = translator(user_input[:100], max_length=100)[0]["language"]
        st.write(f"Detected Language: {detected_language}")

    # Language selection
    target_language = st.selectbox("Select the target language", ["English", "Spanish", "French", "German"])

    # Translate button
    if st.button("Translate"):
        if user_input:
            # Translate the text
            if target_language == "English":
                translation = translator(user_input, max_length=150)[0]["translation_text"]
            else:
                target_language_code = {"Spanish": "es", "French": "fr", "German": "de"}[target_language]
                translation = translator(user_input, max_length=150, target_lang=target_language_code)[0]["translation_text"]

            # Display the translation
            st.success(f"Translation: {translation}")
        else:
            st.warning("Please enter some text to translate.")

if __name__ == "__main__":
    main()
