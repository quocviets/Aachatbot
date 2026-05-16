import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

from model import build_mobilenetv3_small


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ========================
# PATH MODELS
# ========================

STAGE1_MODEL = r"C:\Users\lequo\Downloads\Đồ Án\Model\Stage 1\stage1_plant_classifier.pth"

STAGE2_MODELS = {
    "apple": r"C:\Users\lequo\Downloads\Đồ Án\Model\Apple\Apple_classifier.pth",
    "Corn": r"C:\Users\lequo\Downloads\Đồ Án\Model\Corn\Corn_classifier.pth",
    "grape": r"C:\Users\lequo\Downloads\Đồ Án\Model\Grape\Grape_classifier.pth",
    "Rice_Leaf": r"C:\Users\lequo\Downloads\Đồ Án\Model\Rice_leaf\Rice_leaf_classifier.pth",
    "tomato": r"C:\Users\lequo\Downloads\Đồ Án\Model\Tomato\Tomato_classifier.pth"
}


# ========================
# CLASS NAMES
# ========================

STAGE1_CLASSES = [
    "Corn",
    "Other",
    "Rice_Leaf",
    "apple",
    "grape",
    "tomato"
]


STAGE2_CLASSES = {

    "apple": [
        "Apple_scab",
        "Black_rot",
        "Cedar_apple_rust",
        "Healthy"
    ],

    "Corn": [
        "Blight",
        "Common_Rust",
        "Gray_Leaf_Spot",
        "Healthy"
    ],

    "grape": [
        "Black_rot",
        "Esca",
        "Leaf_blight",
        "Healthy"
    ],

    "Rice_Leaf": [
        "Bacterial_Leaf_Blight",
        "Healthy",
        "Narrow_Brown_Spot",
        "Neck_Blast"
    ],

    "tomato": [
        "Early_blight",
        "Late_blight",
        "Leaf_Mold",
        "Healthy"
    ]
}


# ========================
# IMAGE TRANSFORM
# ========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ========================
# LOAD MODEL
# ========================

def load_model(model_path, num_classes):

    model = build_mobilenetv3_small(num_classes)

    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE)
    )

    model.to(DEVICE)
    model.eval()

    return model


# ========================
# STAGE1 PREDICT
# ========================

def predict_stage1(image):

    model = load_model(STAGE1_MODEL, len(STAGE1_CLASSES))

    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(image)
        print("OUTPUT SHAPE:", outputs.shape)
        probs = F.softmax(outputs, dim=1)
        print("PROBS:", probs)
        pred = torch.argmax(probs).item()
        print("PRED INDEX:", pred)
        plant = STAGE1_CLASSES[pred]
        print("PLANT:", plant)
        confidence = probs[0][pred].item()
        print("CONFIDENCE:", confidence)
    return plant, confidence


# ========================
# STAGE2 PREDICT
# ========================

def predict_stage2(image, plant):

    model_path = STAGE2_MODELS[plant]

    classes = STAGE2_CLASSES[plant]

    model = load_model(model_path, len(classes))

    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(image)

        probs = F.softmax(outputs, dim=1)

        pred = torch.argmax(probs).item()

    disease = classes[pred]

    confidence = probs[0][pred].item()

    return disease, confidence


# ========================
# MAIN PIPELINE
# ========================

def predict(image_path):

    image = Image.open(image_path).convert("RGB")

    plant, plant_conf = predict_stage1(image)

    print("Plant:", plant, f"({plant_conf:.2f})")

    if plant == "Other":
        print("Plant not supported.")
        return

    disease, dis_conf = predict_stage2(image, plant)

    print("Disease:", disease, f"({dis_conf:.2f})")


# ========================
# RUN TEST
# ========================

if __name__ == "__main__":

    image_path = r"C:\Users\lequo\Downloads\images (2).jpg"

    predict(image_path)