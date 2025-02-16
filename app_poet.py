import streamlit as st
from transformers import TFAutoModelForCausalLM, AutoTokenizer
import tensorflow as tf
from PIL import Image
import re  # For text formatting

# Set the page layout
st.set_page_config(layout="wide")

# Load the model and tokenizer
@st.cache_resource
def load_model():
    model_path = "C:/Users/AVNEET/Desktop/saved_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = TFAutoModelForCausalLM.from_pretrained(model_path)
    return tokenizer, model

tokenizer, model = load_model()

# Function to format text as a poem
def format_poem(text):
    formatted_poem = re.sub(r'([,.;!?])\s*', r'\1\n', text)  # Line break after punctuation
    return formatted_poem

# Create columns: Image on the left, text on the right
col1, col2 = st.columns([1, 2])  # Adjust the ratio as needed

# Load and display the image in the left column
with col1:
    image_path = "C:/Users/AVNEET/Desktop/transparent-mustache-medieval-man-reading-book-as-you-like-it6616de1c11cfd7.65241397.png"
    image = Image.open(image_path)
    st.image(image, use_column_width=True)

# Right column: Title, input, and generated poem
with col2:
    st.title("📝 AI Poem Generator")
    st.write("Enter a prompt, and let the AI create a beautiful poem for you!")

    # User input
    user_input = st.text_area("Enter a prompt:", "Ba Ba the black sheep!")

    if st.button("Generate Poem"):
        if user_input.strip():
            input_ids = tokenizer.encode(user_input, return_tensors='tf')

            sample_outputs = model.generate(
                input_ids,
                do_sample=True,
                max_length=100,
                top_k=50,
                top_p=0.9,
                temperature=1.0,
                num_return_sequences=1
            )

            poem = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)

            # Format the poem with proper line breaks
            formatted_poem = format_poem(poem)

            st.subheader("✨ Generated Poem:")
            st.text_area("", formatted_poem, height=300)  # Display as multi-line text area
        else:
            st.warning("Please enter a prompt!")

# Footer
st.markdown("---")
st.caption("🚀 Built by famous Poet -- The Navneet Jha")
