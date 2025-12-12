import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import json
import os
from product_tagger.modeling.vit import create_vit_model
from product_tagger.config import MODELS_DIR

# Paths (adapt if project structure changes)
MODEL_PATH = os.path.join(MODELS_DIR, "vit_articleType.pt")

# Load model/config exactly as in predict.py
def load_model_and_metadata():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    meta = checkpoint.get("meta", {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "image_size": 224,
    })
    num_classes = len(class_to_idx)
    # predict.py uses create_vit_model(num_classes, pretrained=False), returns (model, meta_d)
    model, _ = create_vit_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, device, idx_to_class, meta

model, device, idx_to_class, meta = load_model_and_metadata()

# Preprocessing (match values used in training/inference exactly)
mean = meta["mean"]
std = meta["std"]
image_size = meta.get("image_size", 224)
if isinstance(image_size, (tuple, list)):
    image_size = image_size[0]  # for (h, w) tuples
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

def predict(image):
    img = image.convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        _, pred_idx = outputs.max(1)
        pred_idx = pred_idx.item()
        pred_label = idx_to_class[pred_idx]
        prob = torch.softmax(outputs, dim=1)[0, pred_idx].item()
    return {"Predicted label": pred_label, "Confidence": f"{prob:.2%}"}

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Label(num_top_classes=1)],
    title="Product Type Prediction (ViT Model)",
    description="Upload a clothing product image to get a model prediction for its type."
)

if __name__ == "__main__":
    iface.launch()
