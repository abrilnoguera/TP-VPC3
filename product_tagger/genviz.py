def main():
    # 1) Cargar modelo desde mlflow
    RUN_ID = "PONE_ACÁ_TU_RUN_ID"
    ARTIFACT_PATH = "model"  # o el que hayas usado
    model = get_model_from_mlflow(RUN_ID, ARTIFACT_PATH)

    # 2) Elegir las capas de atención a enganchar
    # Esto depende de tu implementación.
    # Ejemplos posibles:
    #   - model.encoder.layers[i].attn
    #   - model.blocks[i].attn
    #   - model.transformer.layers[i].self_attn
    #
    # Ajusta esta línea a tu modelo real:
    attn_modules = [blk.attn for blk in model.encoder.layers]

    hooks, attn_maps = register_attn_hooks(model, attn_modules)

    # 3) Preparar imagen
    img_path = "ruta/a/una/imagen.jpg"
    x, pil_img = load_image_as_tensor(img_path)
    x = x.to(device)

    # 4) Forward para llenar attn_maps
    with torch.no_grad():
        _ = model(x)

    # Soltar hooks
    for h in hooks:
        h.remove()

    # 5) Atencion de la última capa
    cls_attn = compute_cls_attention_map(attn_maps, layer_idx=-1, head_fusion="mean")
    attn_map = cls_attention_to_map(cls_attn, img_size=pil_img.size[::-1])  # PIL: (W,H)

    # 6) Visualizar
    show_attention_overlay(pil_img, attn_map, alpha=0.5, cmap="jet")


if __name__ == "__main__":
    main()
