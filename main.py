import streamlit as st
import tensorflow as tf
import numpy as np
import google.generativeai as genai

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

def fetch_API(disease_name, plant_name):
    genai.configure(api_key="AIzaSyDCOn1pf2ail04yipozTV9Vt1JNaRcct6g")
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content("Explain how to cure and prevent " + disease_name + " in " + plant_name + " plants in two parts first cure and second prevention in points and less than 250 words and no extra text in the starting and the ending.")
    return response.text

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    st.header("AI and Image based Crop Health Detection System")
    image_path = "home_page.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown(
        """
    Welcome to our project! 🌿🔍
    
    Our goal is to effectively assist in the identification of plant diseases. Our technology will examine a plant image you upload to look for any indications of disease. Let's work together to safeguard our crops and guarantee a more nutritious harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
"""
    )

# About Page
elif app_mode == "About":
    st.header("About")
    st.markdown(
        """
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this github repo. This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure. A new directory containing 33 test images is created later for prediction purpose.
    #### Content
    1. Train (70295 images)
    2. Valid (17572 image)
    3. Test (33 images)
"""
    )

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if st.button("Show Image"):
        if test_image is not None:
            st.image(test_image, use_column_width=True)
    # Predict Button
    if st.button("Predict"):
        if test_image is not None:
            with st.spinner("Please Wait.."):
                st.write("Our Prediction")
                result_index = model_prediction(test_image)
                # Define Class
                class_name = [
                    "Apple___Apple_scab",
                    "Apple___Black_rot",
                    "Apple___Cedar_apple_rust",
                    "Apple___healthy",
                    "Blueberry___healthy",
                    "Cherry_(including_sour)___Powdery_mildew",
                    "Cherry_(including_sour)___healthy",
                    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
                    "Corn_(maize)___Common_rust_",
                    "Corn_(maize)___Northern_Leaf_Blight",
                    "Corn_(maize)___healthy",
                    "Grape___Black_rot",
                    "Grape___Esca_(Black_Measles)",
                    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
                    "Grape___healthy",
                    "Orange___Haunglongbing_(Citrus_greening)",
                    "Peach___Bacterial_spot",
                    "Peach___healthy",
                    "Pepper,_bell___Bacterial_spot",
                    "Pepper,_bell___healthy",
                    "Potato___Early_blight",
                    "Potato___Late_blight",
                    "Potato___healthy",
                    "Raspberry___healthy",
                    "Soybean___healthy",
                    "Squash___Powdery_mildew",
                    "Strawberry___Leaf_scorch",
                    "Strawberry___healthy",
                    "Tomato___Bacterial_spot",
                    "Tomato___Early_blight",
                    "Tomato___Late_blight",
                    "Tomato___Leaf_Mold",
                    "Tomato___Septoria_leaf_spot",
                    "Tomato___Spider_mites Two-spotted_spider_mite",
                    "Tomato___Target_Spot",
                    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
                    "Tomato___Tomato_mosaic_virus",
                    "Tomato___healthy",
                ]
                flag = 0
                plant_name = ""
                disease_name = ""
                if "healthy" in class_name[result_index]:
                    item = class_name[result_index].split("___")
                    st.success("It's a healthy {}".format(item[0]) + " plant.")
                else:
                    item = class_name[result_index].split("___")
                    # flag = 1
                    st.error(
                        "The plant is not healthy. It's a {}".format(item[0])
                        + " plant. It's suffering from {}".format(item[1])
                        + " disease."
                    )
                    plant_name = item[0]
                    disease_name = item[1]
                    st.write("See the recommendations below:")
                    with st.spinner("Loading..."):
                        st.write(fetch_API(disease_name, plant_name))