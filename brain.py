
import streamlit as st

from tensorflow.keras.models import load_model import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array from PIL import Image
st.header("Brain Tumor Detection") st.caption("Upload an image. ")
st.caption("The application will infer the one label out of 4 labels: 'no_tumor', 'pituitary_tumor', 'meningioma_tumor', 'glioma_tumor'.")

st.caption("Warning: Do not click Recognize button before uploading image. It will result in error."

model = load_model("Brain_Tumor_Image_Classification_Model(2).h5",compile=False) model.compile(
loss = 'categorical_crossentropy', optimizer = 'Adam',
metrics = ['accuracy']
)

class_names = ['no_tumor', 'pituitary_tumor', 'meningioma_tumor', 'glioma_tumor']


def load_image(image_file):
img = Image.open(image_file) return img
imgpath = st.file_uploader("Choose a file", type =['png', 'jpeg', 'jpg']) if imgpath is not None:
img = load_image(imgpath ) st.image(img, width=224)
 
def predict_label(image2):

imgLoaded = load_img(image2, target_size=(224, 224))

