import torch


def apply_probe_trigger(inputs, trigger_type):
    patched = inputs.clone()
    _, _, h, w = patched.shape
    patch = max(4, min(h, w) // 8)

    if trigger_type == "checkerboard":
        for i in range(patch):
            for j in range(patch):
                value = 1.0 if (i + j) % 2 == 0 else 0.0
                patched[:, :, h - patch + i, w - patch + j] = value
    elif trigger_type == "square":
        patched[:, :, h - patch:, w - patch:] = 1.0
    elif trigger_type == "blending":
        noise = torch.rand_like(patched)
        patched = torch.clamp(0.82 * patched + 0.18 * noise, 0.0, 1.0)
    elif trigger_type == "spatial_conditional":
        y2 = min(h, patch + 2)
        x2 = min(w, patch + 2)
        patched[:, 0, 2:y2, 2:x2] = 1.0
        patched[:, 1:, 2:y2, 2:x2] = 0.0
    elif trigger_type == "instagram_filter":
        sepia = torch.tensor(
            [[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]],
            device=patched.device,
            dtype=patched.dtype,
        )
        flat = patched.permute(0, 2, 3, 1).reshape(-1, 3)
        patched = torch.matmul(flat, sepia.T).reshape(inputs.size(0), h, w, 3).permute(0, 3, 1, 2)
        patched = torch.clamp(patched, 0.0, 1.0)

    return patched


def black_box_trigger_sweep(model, clean_loader, device, num_batches, trigger_types=None):
    trigger_types = trigger_types or ["checkerboard", "square", "blending", "spatial_conditional", "instagram_filter"]
    model.eval()
    clean_counts = None
    clean_total = 0
    sweep = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(clean_loader):
            if batch_idx >= num_batches:
                break
            inputs = batch[0].to(device)
            clean_logits = model(inputs)
            num_classes = clean_logits.shape[1]
            if clean_counts is None:
                clean_counts = torch.zeros(num_classes, device=device)
                for name in trigger_types:
                    sweep[name] = torch.zeros(num_classes, device=device)

            clean_preds = torch.argmax(clean_logits, dim=1)
            clean_counts += torch.bincount(clean_preds, minlength=num_classes)
            clean_total += inputs.size(0)

            for name in trigger_types:
                patched = apply_probe_trigger(inputs, name)
                preds = torch.argmax(model(patched), dim=1)
                sweep[name] += torch.bincount(preds, minlength=num_classes)

    if clean_counts is None or clean_total == 0:
        return {
            "blackbox_sweep_risk": 0.0,
            "blackbox_sweep_target": None,
            "blackbox_sweep_trigger": None,
            "blackbox_sweep_lift": 0.0,
            "blackbox_sweep_target_rate": 0.0,
            "blackbox_sweep_clean_target_rate": 0.0,
        }

    clean_rates = clean_counts / clean_total
    best = {
        "risk": 0.0,
        "target": None,
        "trigger": None,
        "lift": 0.0,
        "target_rate": 0.0,
        "clean_target_rate": 0.0,
    }

    for name, counts in sweep.items():
        rates = counts / clean_total
        target_rate, target = torch.max(rates, dim=0)
        target = int(target.item())
        target_rate = float(target_rate.item())
        clean_target_rate = float(clean_rates[target].item())
        lift = max(0.0, target_rate - clean_target_rate)
        lift_risk = min(max((lift - 0.15) / 0.55, 0.0), 1.0)
        concentration_risk = min(max((target_rate - 0.45) / 0.50, 0.0), 1.0) if lift > 0.08 else 0.0
        risk = max(lift_risk, 0.85 * concentration_risk)
        if risk > best["risk"]:
            best = {
                "risk": float(risk),
                "target": target,
                "trigger": name,
                "lift": float(lift),
                "target_rate": float(target_rate),
                "clean_target_rate": float(clean_target_rate),
            }

    return {
        "blackbox_sweep_risk": best["risk"],
        "blackbox_sweep_target": best["target"],
        "blackbox_sweep_trigger": best["trigger"],
        "blackbox_sweep_lift": best["lift"],
        "blackbox_sweep_target_rate": best["target_rate"],
        "blackbox_sweep_clean_target_rate": best["clean_target_rate"],
    }
