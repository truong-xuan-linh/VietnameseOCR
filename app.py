import streamlit as st
from PIL import Image
st.set_page_config(page_title="Vietnamese OCR", layout="wide", page_icon = "./storage/linhai.jpeg")

#Trick to not init function multitime
option = st.selectbox(
    "Choose the detection model",
    ("PaddleOCR", "ESTextSpotter"),
)

if "ocr_detector" not in st.session_state or option != st.session_state.get('option'):
    print(f"INIT MODEL {option}")
    if "ocr_detector" not in st.session_state:
        from src.setup import Setup
        Setup().ocr_model_downloader()
    
    from src.OCR import OCRDetector
    st.session_state.option = option
    is_paddle_ocr = True if option == "PaddleOCR" else False
    st.session_state.ocr_detector = OCRDetector(is_paddle_ocr=is_paddle_ocr)
    print(f"DONE INIT MODEL {option}")

hide_menu_style = """
<style>
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_menu_style, unsafe_allow_html= True)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        margin-left: -400px;
    }
     
    """,
    unsafe_allow_html=True,
)

st.markdown("<h2 style='text-align: center; color: grey;'>Input: Image </h2>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: grey;'>Output: The Vietnamese or English text in the image (if any).</h2>", unsafe_allow_html=True)
left_col, right_col = st.columns(2)

#LEFT COLUMN
upload_image = left_col.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "webp", ])

if left_col.button("OCR Detect"):
    image, texts, boxes = st.session_state.ocr_detector.text_detector(upload_image, is_local=True)
    left_col.write("**RESULTS:** ")
    left_col.write(texts)
    
    #RIGHT COLUMN
    visualize_image = st.session_state.ocr_detector.visualize_ocr(image, texts, boxes)
    right_col.write("**ORIGIN IMAGE:** ")
    right_col.image(image)
    right_col.write("**OCR IMAGE:** ")
    right_col.image(visualize_image)