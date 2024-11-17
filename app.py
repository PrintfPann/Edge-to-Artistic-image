import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

def preprocess_image(image):
    image = tf.image.resize(image, (256, 256))  # Resize เป็น 256x256
    image = image / 127.5 - 1  # Normalize [-1, 1]
    return tf.expand_dims(image, axis=0)  # เพิ่มมิติ batch

def generate_image(generator, input_image):
    input_image = preprocess_image(input_image)
    generated_image = generator(input_image, training=True)
    generated_image = (generated_image + 1) * 127.5  # Unnormalize [0, 255]
    return tf.squeeze(generated_image).numpy().astype("uint8")

generator_black = load_model("model/generator_model_black_100.h5")
generator_white = load_model("model/generator_model_white_100.h5")

st.title("Pix2Pix: Edge to Artistic Image")
st.subheader("Choose a version to generate your artistic image.")

option = st.selectbox("Select a version:", ["Black Version", "White Version"])

if option == "Black Version":
    st.subheader("Upload an edge image for the Black version")
    uploaded_file = st.file_uploader("Upload an edge image", type=["jpg", "png"], key="black_version")
    generator = generator_black
elif option == "White Version":
    st.subheader("Upload an edge image for the White version")
    uploaded_file = st.file_uploader("Upload an edge image", type=["jpg", "png"], key="white_version")
    generator = generator_white

if uploaded_file:
    input_image = Image.open(uploaded_file).convert("RGB")
    input_image_np = np.array(input_image)

    st.image(input_image, caption="Input Image", use_container_width=True)

    with st.spinner("Generating image..."):
        output_image = generate_image(generator, input_image_np)
        st.image(output_image, caption="Generated Image", use_container_width=True)
