import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
from product_tagger.config import MLFLOW_TRACKING_URI
import mlflow
import mlflow.pytorch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model_from_mlflow(run_id: str, artifact_path: str = "model"):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"runs:/{run_id}/{artifact_path}"
    model = mlflow.pytorch.load_model(model_uri, map_location=device)
    model.eval().to(device)
    return model


def register_attn_hooks(model, attn_modules):
    """
    attn_modules: lista de módulos de atención (por ejemplo, los bloques del encoder)
    """
    attn_maps = []

    def hook_fn(module, input, output):
        # output esperado: [B, num_heads, tokens, tokens]
        attn_maps.append(output.detach().cpu())

    hooks = [m.register_forward_hook(hook_fn) for m in attn_modules]
    return hooks, attn_maps


def compute_cls_attention_map(attn_maps, layer_idx=-1, head_fusion="mean"):
    """
    attn_maps: lista de tensores [B, num_heads, tokens, tokens]
    layer_idx: qué capa usar ( -1 = última capa )
    head_fusion: 'mean' (promedio de heads) o 'max'
    """
    attn = attn_maps[layer_idx]  # [B, H, T, T]
    attn = attn[0]               # batch 0 → [H, T, T]

    if head_fusion == "mean":
        attn = attn.mean(dim=0)  # [T, T]
    elif head_fusion == "max":
        attn, _ = attn.max(dim=0)
    else:
        raise ValueError("head_fusion must be 'mean' or 'max'")

    # Atención del token CLS (índice 0) hacia los demás tokens
    cls_attn = attn[0]          
    cls_attn = cls_attn[1:]     

    return cls_attn  # vector de tamaño (num_patches,)


def cls_attention_to_map(cls_attn, img_size=(224, 224)):
    """
    cls_attn: tensor 1D con atención a cada patch (longitud = num_patches)
    img_size: tamaño original de la imagen (H, W)
    """
    num_patches = cls_attn.shape[0]
    grid_size = int(num_patches ** 0.5)  # asumimos grid cuadrado

    attn_map = cls_attn.reshape(1, 1, grid_size, grid_size)  # [1,1,h,w]

    # Normalizar a 0–1
    attn_map = attn_map - attn_map.min()
    if attn_map.max() > 0:
        attn_map = attn_map / attn_map.max()

    # Re-escalar al tamaño de imagen
    attn_map = F.interpolate(attn_map, size=img_size, mode="bilinear", align_corners=False)
    attn_map = attn_map.squeeze().numpy()  # [H,W]

    return attn_map


def load_image_as_tensor(path, size=(224, 224)):
    img = Image.open(path).convert("RGB")
    img = img.resize(size)
    # [H, W, C] → [C, H, W], normalizado a 0–1
    arr = np.array(img).astype("float32") / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    tensor = torch.from_numpy(arr).unsqueeze(0)  # [1,3,H,W]
    return tensor, img  # tensor, imagen PIL original


def show_attention_overlay(pil_img, attn_map, alpha=0.5, cmap="jet"):
    """
    pil_img: imagen original (PIL.Image)
    attn_map: numpy [H,W] con valores 0–1
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(pil_img)
    ax.imshow(attn_map, cmap=cmap, alpha=alpha)
    ax.axis("off")
    plt.tight_layout()
    plt.show()
