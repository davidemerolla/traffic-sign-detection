import os
import gdown
import streamlit as st
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import random

# ID dei file individuali su Google Drive
MODEL_FILES = {
    "best_93.pt": "1iDtNbaxcFi6e-jBfNTWObfi5OvVg2jmw",
    "road_sign_classifier.h5": "1kfXHwsl_rCMOGORwsobYycupmLGv9uSb",
    "best (1).pt": "1D8noW2biYWbnBTacfOXXzYZ88zXXfdcV",
    "best (3).pt": "13Nl6pw6ntLmLfoR42tfUlvxRqVXCO453",
    "best.pt": "1vr6Dmuo88hYqBH5r182aZ1RuHVcAAaNy",
    "model_- 14 march 2024 21_34.pt": "1WkDFeccSuitnIeJJT-mBqPA3B7vI3-Hc"
}

# Percorso di destinazione per i file scaricati
TARGET_DIR = "models"
os.makedirs(TARGET_DIR, exist_ok=True)

# Funzione per scaricare singoli file da Google Drive
def download_model_files():
    for filename, file_id in MODEL_FILES.items():
        filepath = os.path.join(TARGET_DIR, filename)
        if not os.path.exists(filepath):
            print(f"Scaricando {filename} da Google Drive...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, filepath, quiet=False)
        else:
            print(f"{filename} è già presente.")

# Scarica i file di modello (se non presenti)
download_model_files()

# Percorsi dei modelli
yolo_model_path = os.path.join(TARGET_DIR, "best_93.pt")
cnn_model_path = os.path.join(TARGET_DIR, "road_sign_classifier.h5")
yolo_damage_model_path = os.path.join(TARGET_DIR, "model_- 14 march 2024 21_34.pt")

# Categorie desiderate per YOLOv5
wanted_classes = [11, 12, 21]

# Carica i modelli YOLO e il modello di classificazione CNN
yolo_v5_weights = [yolo_model_path, 'yolov5s.pt']
yolo_v5_models = [torch.hub.load('ultralytics/yolov5', 'custom', path=weight) for weight in yolo_v5_weights]
yolo_v8_weights = [os.path.join(TARGET_DIR, 'best (1).pt'), os.path.join(TARGET_DIR, 'best (3).pt'), os.path.join(TARGET_DIR, 'best.pt')]
yolo_v8_models = [YOLO(weight) for weight in yolo_v8_weights]
cnn_model = load_model(cnn_model_path, compile=False)
cnn_model.compile(optimizer="adam", loss="binary_crossentropy")
input_shape = cnn_model.input_shape[1:3]

# Carica il modello YOLOv8 per le anomalie sul manto stradale
yolo_damage_model = YOLO(yolo_damage_model_path)

# Genera una palette di colori unici
def generate_unique_colors(num_colors):
    colors = []
    for _ in range(num_colors):
        while True:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            if color not in colors:
                colors.append(color)
                break
    return colors

# Funzione per disegnare forme geometriche attorno al bounding box
def draw_shape(image, shape, x1, y1, x2, y2, color):
    if shape == "circle":
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        radius = max((x2 - x1) // 2, (y2 - y1) // 2)
        cv2.circle(image, center, radius, color, 2)
    elif shape == "hexagon":
        width, height = x2 - x1, y2 - y1
        hexagon_points = np.array([
            [x1 + width // 2, y1],
            [x2, y1 + height // 3],
            [x2, y1 + 2 * height // 3],
            [x1 + width // 2, y2],
            [x1, y1 + 2 * height // 3],
            [x1, y1 + height // 3]
        ])
        cv2.polylines(image, [hexagon_points], isClosed=True, color=color, thickness=2)
    elif shape == "triangle":
        triangle_points = np.array([
            [int((x1 + x2) / 2), y1],
            [x2, y2],
            [x1, y2]
        ])
        cv2.polylines(image, [triangle_points], isClosed=True, color=color, thickness=2)
    elif shape == "rectangle":
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    elif shape == "ellipse":
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        axes = (max((x2 - x1) // 2, 1), max((y2 - y1) // 2, 1))
        cv2.ellipse(image, center, axes, 0, 0, 360, color, 2)

# Funzione per aggiungere una legenda
def add_legend(image, model_names, colors, position="top-right"):
    legend_img = image.copy()
    offset_x = 10 if position == "top-left" else legend_img.shape[1] - 150
    offset_y = 10
    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        cv2.putText(legend_img, model_name, (offset_x, offset_y + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return legend_img

# Funzione di rilevamento e classificazione
def detect_and_classify(image):
    cartelli_trovati = False
    messages = []
    color_palette = generate_unique_colors(len(yolo_v5_weights) + len(yolo_v8_weights) + 1)  # +1 per il nuovo modello
    shapes = ["circle", "hexagon", "triangle", "rectangle", "ellipse", "ellipse"]  # Forma aggiuntiva per il modello stradale
    model_shapes = shapes[:len(yolo_v5_weights) + len(yolo_v8_weights) + 1]
    model_names = []

    def resize_and_pad(img, target_size=(640, 640)):
        h, w, _ = img.shape
        scale = min(target_size[0] / h, target_size[1] / w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (nw, nh))
        padded = np.full((target_size[0], target_size[1], 3), (0, 0, 0), dtype=np.uint8)
        pad_top, pad_left = (target_size[0] - nh) // 2, (target_size[1] - nw) // 2
        padded[pad_top:pad_top + nh, pad_left:pad_left + nw] = resized
        return padded

    image_resized = resize_and_pad(image)
    annotated_image = image_resized.copy()

    # Rilevamento con YOLOv5 per i cartelli
    for idx, model in enumerate(yolo_v5_models):
        model_name = f"YOLOv5_{idx + 1}"
        model_names.append(model_name)
        results = model(image_resized)
        df = results.pandas().xyxy[0]  # YOLOv5 usa .pandas() per ottenere i dati

        if 'yolov5s.pt' in yolo_v5_weights[idx]:
            df = df[df['class'].isin(wanted_classes)]

        color = color_palette[idx]
        shape = model_shapes[idx]

        for _, row in df.iterrows():
            cartelli_trovati = True
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            draw_shape(annotated_image, shape, x1, y1, x2, y2, color)
            cropped_sign = image_resized[y1:y2, x1:x2]
            cropped_resized = cv2.resize(cropped_sign, input_shape)
            cropped_resized = np.expand_dims(cropped_resized, axis=0) / 255.0
            damage_prediction = cnn_model.predict(cropped_resized)
            status = "Intatto" if damage_prediction[0][0] >= 0.5 else "Danneggiato"
            messages.append(f"{model_name} - Stato: {status}")

    # Rilevamento con YOLOv8 per i cartelli
    for idx, model in enumerate(yolo_v8_models):
        model_name = f"YOLOv8_{idx + 1}"
        model_names.append(model_name)
        results = model.predict(image_resized, conf=0.3)
        color = color_palette[len(yolo_v5_weights) + idx]
        shape = model_shapes[len(yolo_v5_weights) + idx]

        for box in results[0].boxes:
            cartelli_trovati = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            draw_shape(annotated_image, shape, x1, y1, x2, y2, color)
            cropped_sign = image_resized[y1:y2, x1:x2]
            cropped_resized = cv2.resize(cropped_sign, input_shape)
            cropped_resized = np.expand_dims(cropped_resized, axis=0) / 255.0
            damage_prediction = cnn_model.predict(cropped_resized)
            status = "Intatto" if damage_prediction[0][0] >= 0.5 else "Danneggiato"
            messages.append(f"{model_name} - Stato: {status}")

    # Rilevamento con il modello YOLOv8 per anomalie sul manto stradale
    model_name = "YOLOv8_Strada"
    model_names.append(model_name)
    results = yolo_damage_model.predict(image_resized, conf=0.3)
    color = color_palette[-1]
    shape = model_shapes[-1]

    for box in results[0].boxes:
        cartelli_trovati = True
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        draw_shape(annotated_image, shape, x1, y1, x2, y2, color)
        messages.append(f"{model_name} - Anomalia sul manto stradale rilevata")

    if not cartelli_trovati:
        messages.append("Nessun cartello o anomalia rilevato in questa immagine.")

    # Aggiungi la legenda
    annotated_image = add_legend(annotated_image, model_names, color_palette, position="top-right")
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    return annotated_image_rgb, messages

# Configurazione interfaccia utente Streamlit
st.title("Rilevamento e Classificazione Cartelli Stradali e Anomalie Stradali")
st.write("Carica una o più immagini per identificare e classificare i cartelli stradali e rilevare anomalie sul manto stradale.")

uploaded_files = st.file_uploader("Carica le immagini", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = np.frombuffer(uploaded_file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        st.image(image, caption="Immagine caricata", use_column_width=True)
        st.write("Analisi in corso...")

        # Esegui rilevamento e classificazione
        annotated_image, messages = detect_and_classify(image)

        # Mostra l'immagine con la legenda e i bounding box
        st.image(annotated_image, caption="Immagine con cartelli e anomalie rilevate", use_column_width=True)

        # Mostra i risultati
        for message in messages:
            st.write(message)
