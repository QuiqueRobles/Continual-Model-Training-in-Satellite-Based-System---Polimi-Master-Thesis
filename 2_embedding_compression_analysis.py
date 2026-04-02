# =============================================================
# STEP 19 - Embedding Extraction and Analysis (Corrected)
# Objective: Core of the TFM. We simulate the satellite by extracting
# embeddings from the frozen backbone and compressing them with SVD.
# We now also capture the skip connections (conv1,2,3) that the UNet
# decoder needs, together with the embedding from dconv_down4.
# The complete system simulates:
#   SATELLITE: image → backbone → embedding + skips → compress → downlink
#   GROUND:    compressed embedding → decoder → prediction
# We measure original IoU vs compressed IoU and the compression ratio.
# =============================================================

import torch
import torch.nn.functional as F
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download, list_repo_files
from ml4floods.models.architectures.unets import UNet
from ml4floods.preprocess.worldfloods.normalize import get_normalisation
from ml4floods.data.worldfloods.configs import CHANNELS_CONFIGURATIONS

# --- 1. Load and freeze the model ---
model_path = hf_hub_download(
    repo_id="isp-uv-es/ml4floods",
    filename="models/WF2_unetv2_bgriswirs/model.pt"
)
weights = torch.load(model_path, map_location="cpu", weights_only=False)
model   = UNet(n_channels=6, n_class=2)
model.load_state_dict({k.replace("network.", ""): v for k, v in weights.items()})
model.eval()

# Freeze all parameters (simulating satellite-side backbone)
for param in model.parameters():
    param.requires_grad = False
print("Backbone frozen (simulating satellite)\n")

# --- 2. Hooks to capture embedding and skip connections ---
captures = {}
hooks = [
    model.dconv_down1.register_forward_hook(
        lambda m, i, o: captures.update({'conv1': o.detach()})),
    model.dconv_down2.register_forward_hook(
        lambda m, i, o: captures.update({'conv2': o.detach()})),
    model.dconv_down3.register_forward_hook(
        lambda m, i, o: captures.update({'conv3': o.detach()})),
    model.dconv_down4.register_forward_hook(
        lambda m, i, o: captures.update({'emb':   o.detach()})),
]

# --- 3. Helper functions ---
indices    = CHANNELS_CONFIGURATIONS["bgriswirs"]
mean, std  = get_normalisation("bgriswirs", channels_first=True)
TILE_SIZE  = 256


def preprocess(image_full):
    """Normalize the 6 selected bands"""
    return (image_full[indices, :, :] - mean) / std


def find_best_tile(image_norm, gt_water, tile_size=256):
    """Find the best 256x256 tile with sufficient water content"""
    H, W = image_norm.shape[1], image_norm.shape[2]
    best_score, result = -1, (None, None, None)
    for y in range(0, H - tile_size, tile_size):
        for x in range(0, W - tile_size, tile_size):
            tile    = image_norm[:, y:y+tile_size, x:x+tile_size]
            gt_tile = gt_water[y:y+tile_size, x:x+tile_size]
            valid   = gt_tile != 0
            if valid.sum() < tile_size * tile_size * 0.3:
                continue
            pct_water = np.sum(gt_tile == 2) / valid.sum() if valid.sum() > 0 else 0
            score     = np.var(tile) * pct_water - np.percentile(tile, 95) * 0.1
            if score > best_score and pct_water > 0.05:
                best_score = score
                result = (tile, gt_tile, (y, x))
    return result


def compress_svd(tensor, k_ratio=0.1):
    """
    Compress a tensor [1, C, H, W] using truncated SVD.
    Keeps only the k most important components.
    Returns the reconstructed tensor + original and compressed sizes in bytes.
    """
    arr = tensor.squeeze(0).numpy()          # [C, H, W]
    C, H, W = arr.shape
    mat = arr.reshape(C, H * W)              # [C, H*W]

    U, S, Vt = np.linalg.svd(mat, full_matrices=False)

    k = max(1, int(len(S) * k_ratio))
    Uk = U[:, :k]
    Sk = S[:k]
    Vtk = Vt[:k, :]

    recon = (Uk @ np.diag(Sk) @ Vtk).reshape(C, H, W)

    bytes_orig = arr.nbytes
    bytes_comp = Uk.nbytes + Sk.nbytes + Vtk.nbytes

    return torch.tensor(recon).unsqueeze(0).float(), bytes_orig, bytes_comp


def decoder_from_features(model, emb, conv1, conv2, conv3):
    """
    Exactly reproduces the UNet decoder using the embedding
    and the captured skip connections.
    """
    x = F.interpolate(emb, scale_factor=2, mode='bilinear', align_corners=True)
    x = torch.cat([x, conv3], dim=1)
    x = model.dconv_up3(x)

    x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
    x = torch.cat([x, conv2], dim=1)
    x = model.dconv_up2(x)

    x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
    x = torch.cat([x, conv1], dim=1)
    x = model.dconv_up1(x)

    return model.conv_last(x)


def map_from_logits(logits):
    """Convert logits to class map: 0=land, 1=water, 2=cloud"""
    probs = torch.sigmoid(logits).squeeze()
    map_pred = np.zeros(probs[0].shape, dtype=np.int32)
    map_pred[probs[1].numpy() > 0.5] = 1   # water
    map_pred[probs[0].numpy() > 0.5] = 2   # cloud
    return map_pred


def water_iou(map_pred, gt_water):
    """Calculate IoU only for the water class"""
    valid   = gt_water != 0
    gt_w    = (gt_water == 2) & valid
    pred_w  = (map_pred == 1) & valid
    inter   = np.sum(pred_w & gt_w)
    union   = np.sum(pred_w | gt_w)
    return inter / union if union > 0 else 0.0


# --- 4. Process all test images ---
files = list(list_repo_files(
    repo_id="isp-uv-es/WorldFloodsv2", repo_type="dataset"
))
test_s2 = sorted(set([f for f in files if f.startswith("test/S2/")]))

results = []
kb_image_fixed = 6 * TILE_SIZE * TILE_SIZE * 4 / 1024   # KB for float32 image

print(f"{'Image':<40} {'KB img':>7} {'KB emb':>7} {'KB comp':>8} "
      f"{'Comp%':>7} {'IoU orig':>9} {'IoU SVD':>9}")
print("-" * 100)

for s2_path in test_s2:
    name   = s2_path.replace("test/S2/", "").replace(".tif", "")[:38]
    gt_path = s2_path.replace("test/S2/", "test/gt/")
    try:
        img_path = hf_hub_download(
            repo_id="isp-uv-es/WorldFloodsv2",
            filename=s2_path, repo_type="dataset"
        )
        gt_path_i = hf_hub_download(
            repo_id="isp-uv-es/WorldFloodsv2",
            filename=gt_path, repo_type="dataset"
        )

        with rasterio.open(img_path) as src:
            image_full = src.read().astype(np.float32)
        with rasterio.open(gt_path_i) as src:
            gt_water = src.read(2)

        image_norm = preprocess(image_full)
        tile_img, tile_gt, pos = find_best_tile(image_norm, gt_water)

        if tile_img is None:
            continue

        tensor_input = torch.tensor(tile_img).unsqueeze(0)

        # Full forward pass → captures hooks + original prediction
        with torch.no_grad():
            logits_orig = model(tensor_input)

        map_orig = map_from_logits(logits_orig)
        iou_orig = water_iou(map_orig, tile_gt)

        # Retrieve captured features
        emb   = captures['emb'].clone()
        conv1 = captures['conv1'].clone()
        conv2 = captures['conv2'].clone()
        conv3 = captures['conv3'].clone()

        # Compress only the embedding (skip connections can be recalculated on ground)
        emb_comp, bytes_orig, bytes_comp = compress_svd(emb, k_ratio=0.1)

        # Reconstruct prediction on ground using decoder
        with torch.no_grad():
            logits_comp = decoder_from_features(model, emb_comp, conv1, conv2, conv3)

        map_comp = map_from_logits(logits_comp)
        iou_comp = water_iou(map_comp, tile_gt)

        kb_emb   = bytes_orig / 1024
        kb_comp  = bytes_comp / 1024
        comp_pct = (1 - bytes_comp / bytes_orig) * 100

        print(f"{name:<40} {kb_image_fixed:>7.1f} {kb_emb:>7.1f} {kb_comp:>8.1f} "
              f"{comp_pct:>6.1f}% {iou_orig:>9.3f} {iou_comp:>9.3f}")

        results.append({
            "name": name,
            "iou_orig": iou_orig,
            "iou_comp": iou_comp,
            "kb_img": kb_image_fixed,
            "kb_emb": kb_emb,
            "kb_comp": kb_comp,
            "comp_pct": comp_pct
        })

    except Exception as e:
        print(f"{name:<40} ERROR: {e}")

# Remove hooks
for h in hooks:
    h.remove()

# --- 5. Final Summary ---
if results:
    print("\n" + "=" * 100)
    print(f"FINAL SUMMARY ({len(results)} test images)")
    print(f"  Original image (6 bands, 256x256):     {np.mean([r['kb_img']  for r in results]):>8.1f} KB")
    print(f"  Embedding without compression [512,32,32]: {np.mean([r['kb_emb']  for r in results]):>8.1f} KB")
    print(f"  Compressed embedding SVD (k=10%):       {np.mean([r['kb_comp'] for r in results]):>8.1f} KB")
    print(f"  Reduction vs original image:            "
          f"{(1 - np.mean([r['kb_comp'] for r in results])/kb_image_fixed)*100:>7.1f}%")
    print(f"  Mean Water IoU (full backbone):         {np.mean([r['iou_orig'] for r in results]):>8.3f}")
    print(f"  Mean Water IoU (compressed embedding):  {np.mean([r['iou_comp'] for r in results]):>8.3f}")
    delta = np.mean([r['iou_orig'] - r['iou_comp'] for r in results])
    print(f"  Average IoU degradation:                {delta:>8.3f}")

# --- 6. Plots ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: IoU comparison
names_short = [r['name'][-22:] for r in results]
x = np.arange(len(results))
w = 0.35
axes[0].bar(x - w/2, [r['iou_orig'] for r in results], w,
            label='Full backbone', color='steelblue', alpha=0.85)
axes[0].bar(x + w/2, [r['iou_comp'] for r in results], w,
            label='SVD 10% embedding', color='coral', alpha=0.85)

axes[0].axhline(np.mean([r['iou_orig'] for r in results]),
                color='steelblue', ls='--', lw=1.5,
                label=f"Mean orig: {np.mean([r['iou_orig'] for r in results]):.3f}")
axes[0].axhline(np.mean([r['iou_comp'] for r in results]),
                color='coral', ls='--', lw=1.5,
                label=f"Mean comp: {np.mean([r['iou_comp'] for r in results]):.3f}")

axes[0].set_xticks(x)
axes[0].set_xticklabels(names_short, rotation=45, ha='right', fontsize=7)
axes[0].set_ylabel('Water IoU')
axes[0].set_title('IoU: Full Backbone vs Compressed Embedding (SVD 10%)')
axes[0].legend(fontsize=8)
axes[0].set_ylim(0, 1.15)

# Plot 2: Size comparison
labels = ['Original\nimage\n(6 bands)', 'Embedding\n(uncompressed)\n[512,32,32]',
          'Embedding\ncompressed\n(SVD 10%)']
values = [np.mean([r['kb_img']  for r in results]),
          np.mean([r['kb_emb']  for r in results]),
          np.mean([r['kb_comp'] for r in results])]
colors = ['steelblue', 'orange', 'coral']

bars = axes[1].bar(labels, values, color=colors, width=0.5, alpha=0.85)
for bar, val in zip(bars, values):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 10,
                 f'{val:.0f} KB', ha='center', va='bottom',
                 fontweight='bold', fontsize=11)

axes[1].set_ylabel('Size (KB)')
axes[1].set_title('Size comparison (average over test images)')

plt.tight_layout()
plt.savefig("embedding_analysis_final.png", dpi=150)
plt.show()
print("\nPlot saved as embedding_analysis_final.png")