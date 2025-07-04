import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageEnhance
import pytesseract
import easyocr
import numpy as np
from google.cloud import vision
import io
import json
import os
import tempfile

# ========== Multilingual Support ==========
LANGUAGES = {
    "English": "en",
    "Русский": "ru"
}

lang = st.sidebar.radio("🌐 Language / Язык", list(LANGUAGES.keys()))
locale = LANGUAGES[lang]

# Translations dictionary
translations = {
    "en": {
        "title": "🖊️ SROIE Annotator (by categories)",
        "upload_key_title": "🔐 Google Vision API Key",
        "upload_key": "📁 Upload Google Cloud JSON key",
        "key_success": "✅ Key uploaded and applied!",
        "key_warning": "⚠️ Key not uploaded. Google Vision API won't work.",
        "ocr_select": "🧠 Choose OCR engine",
        "upload_image": "📎 Upload receipt image",
        "draw_prompt": "🖱️ Mark region on image",
        "recognized_text": "✏️ Recognized text",
        "new_category": "➕ New category",
        "add_category_btn": "➕ Add category",
        "category_exists": "⚠️ Category already exists.",
        "category_empty": "⚠️ Enter category name.",
        "category_added": "✅ Category added:",
        "select_category": "📂 Select category",
        "add_annotation_btn": "➕ Add annotation to",
        "added_to": "✅ Added:",
        "all_annotations": "📋 All annotations",
        "download_txt": "💾 Download .txt",
        "download_json": "📤 Download JSON",
        "download_original": "🖼️ Download original",
        "category": "🗂️",
        "add_annotation": "📌 Add annotation by category",
        "upload_key_label": "📁 Upload Google Cloud JSON key"
    },
    "ru": {
        "title": "🖊️ SROIE Аннотатор (по категориям)",
        "upload_key_title": "🔐 Ключ Google Vision API",
        "upload_key": "📁 Загрузите JSON-ключ Google Cloud",
        "key_success": "✅ Ключ загружен и применён!",
        "key_warning": "⚠️ Ключ не загружен. Google Vision API не будет работать.",
        "ocr_select": "🧠 Выберите OCR-движок",
        "upload_image": "📎 Загрузите изображение чека",
        "draw_prompt": "🖱️ Отметьте регион на изображении",
        "recognized_text": "✏️ Распознанный текст",
        "new_category": "➕ Новая категория",
        "add_category_btn": "➕ Добавить категорию",
        "category_exists": "⚠️ Такая категория уже существует.",
        "category_empty": "⚠️ Введите название категории.",
        "category_added": "✅ Категория добавлена:",
        "select_category": "📂 Выберите категорию",
        "add_annotation_btn": "➕ Добавить аннотацию в",
        "added_to": "✅ Добавлено:",
        "all_annotations": "📋 Все аннотации",
        "download_txt": "💾 Скачать .txt",
        "download_json": "📤 Скачать JSON",
        "download_original": "🖼️ Скачать оригинал",
        "category": "🗂️",
        "add_annotation": "📌 Добавить аннотацию по категориям",
        "upload_key_label": "📁 Загрузите JSON-ключ Google Cloud"
    }
}[locale]

# ========== Page Config ==========
st.set_page_config(page_title=translations["title"], layout="wide")
st.title(translations["title"])

# ========== Upload API Key ==========
st.sidebar.markdown(translations["upload_key_title"])
uploaded_key = st.sidebar.file_uploader(translations["upload_key_label"], type=["json"])

if uploaded_key:
    temp_key_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    temp_key_file.write(uploaded_key.read())
    temp_key_file.flush()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_key_file.name
    st.sidebar.success(translations["key_success"])
elif "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    st.sidebar.warning(translations["key_warning"])

# ========== OCR Engine Selection ==========
ocr_option = st.radio(translations["ocr_select"], [
    "Tesseract (по умолчанию)",
    "EasyOCR",
    "Google Cloud Vision"
])

# EasyOCR init
if "easyocr_reader" not in st.session_state:
    st.session_state.easyocr_reader = easyocr.Reader(["en", "ru"])

def run_easyocr(image: Image.Image) -> str:
    img_np = np.array(image)
    result = st.session_state.easyocr_reader.readtext(img_np, detail=0)
    return "\n".join(result)

def run_google_vision(image: Image.Image) -> str:
    client = vision.ImageAnnotatorClient()
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    content = buffered.getvalue()
    image_gcv = vision.Image(content=content)
    response = client.text_detection(image=image_gcv)
    return response.text_annotations[0].description if response.text_annotations else ""

# ========== Default Categories ==========
if "CATEGORIES" not in st.session_state:
    st.session_state.CATEGORIES = ["Company", "Date", "Total", "Address"]

if "annotations" not in st.session_state:
    st.session_state.annotations = {cat: [] for cat in st.session_state.CATEGORIES}

# ========== Image Upload ==========
uploaded_file = st.file_uploader(translations["upload_image"], type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    filename = uploaded_file.name.rsplit(".", 1)[0]

    st.subheader(translations["draw_prompt"])
    canvas_result = st_canvas(
        fill_color="rgba(0, 255, 0, 0.3)",
        stroke_width=2,
        stroke_color="green",
        background_image=image,
        update_streamlit=True,
        height=image.height,
        width=image.width,
        drawing_mode="polygon",
        key="canvas",
    )

    if canvas_result.json_data:
        objects = canvas_result.json_data.get("objects", [])
        if objects:
            last = objects[-1]
            if "path" in last:
                path = last["path"]
                x_coords = [p[1] for p in path]
                y_coords = [p[2] for p in path]
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))

                padding = 5
                x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
                x_max, y_max = min(image.width, x_max + padding), min(image.height, y_max + padding)

                cropped = image.crop((x_min, y_min, x_max, y_max))
                gray = cropped.convert("L")
                enhanced = ImageEnhance.Contrast(gray).enhance(2.0)

                # Run selected OCR
                if ocr_option == "Tesseract (по умолчанию)":
                    text = pytesseract.image_to_string(enhanced).strip()
                elif ocr_option == "EasyOCR":
                    text = run_easyocr(enhanced)
                elif ocr_option == "Google Cloud Vision":
                    if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
                        text = run_google_vision(enhanced)
                    else:
                        st.warning(translations["key_warning"])
                        text = ""
                else:
                    text = ""

                st.image(cropped, caption="📸", width=300)

                with st.expander(translations["add_annotation"], expanded=True):
                    corrected_text = st.text_input(translations["recognized_text"], value=text, key="corrected_text")

                    new_cat = st.text_input(translations["new_category"], value="", key="new_cat")
                    if st.button(translations["add_category_btn"]):
                        new_cat = new_cat.strip()
                        if new_cat and new_cat not in st.session_state.CATEGORIES:
                            st.session_state.CATEGORIES.append(new_cat)
                            st.session_state.annotations[new_cat] = []
                            st.success(f"{translations['category_added']} {new_cat}")
                        elif new_cat in st.session_state.CATEGORIES:
                            st.warning(translations["category_exists"])
                        else:
                            st.warning(translations["category_empty"])

                    selected_cat = st.selectbox(translations["select_category"], st.session_state.CATEGORIES)
                    if st.button(f"{translations['add_annotation_btn']} «{selected_cat}»"):
                        st.session_state.annotations[selected_cat].append({
                            "coords": [(x_min, y_min), (x_max, y_max)],
                            "text": corrected_text
                        })
                        st.success(f"{translations['added_to']} [{selected_cat}] — {corrected_text}")

    # ========== Display Annotations ==========
    st.subheader(translations["all_annotations"])
    for cat in st.session_state.CATEGORIES:
        st.markdown(f"### {translations['category']} {cat}")
        for i, ann in enumerate(st.session_state.annotations.get(cat, [])):
            st.markdown(f"{i+1}. `{ann['coords']}` → _{ann['text']}_")

    # ========== Export ==========
    txt_data = ""
    structured = {}
    for cat in st.session_state.CATEGORIES:
        structured[cat.lower()] = []
        for ann in st.session_state.annotations[cat]:
            coords_flat = ",".join(str(p) for xy in ann["coords"] for p in xy)
            txt_data += f"{coords_flat},{cat.upper()}: {ann['text']}\n"
            structured[cat.lower()].append(ann["text"])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(label=translations["download_txt"], data=txt_data.encode("utf-8"),
                           file_name=f"{filename}.txt", mime="text/plain")
    with col2:
        st.download_button(label=translations["download_json"], data=json.dumps(structured, indent=4, ensure_ascii=False).encode("utf-8"),
                           file_name=f"{filename}_structured.json", mime="application/json")
    with col3:
        buffered_img = io.BytesIO()
        image.save(buffered_img, format="JPEG")
        st.download_button(label=translations["download_original"], data=buffered_img.getvalue(),
                           file_name=f"{filename}_original.jpg", mime="image/jpeg")
