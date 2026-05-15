"""
Fast Meta-Classifier Training
Reduced sample counts for quick turnaround (~5 min for 6 models).
Produces the same 10-dimensional feature vector as the full pipeline.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import numpy as np
import glob
from defenses import (
    NeuralCleanse, STRIP, ActivationClustering, WeightAnalysis,
    NaturalTrojanProfiler, GradientSimilarity, SpectralSignatures,
    ConfidenceDistributionAnalysis, RiskFusionEngine, RiskMetaClassifier,
)
from trojai_model_wrapper import TrojAI_ModelWrapper
from dataset import get_cifar10_dataloaders
from models import get_resnet18
from torch.utils.data import DataLoader, Subset

def _limit_loader(loader, max_samples):
    if max_samples is None or max_samples >= len(loader.dataset):
        return loader
    subset = Subset(loader.dataset, range(min(max_samples, len(loader.dataset))))
    return DataLoader(subset, batch_size=loader.batch_size, shuffle=False, num_workers=0)

def load_model(model_path, device):
    """Dynamic architecture adaptation (mirrors celery_worker.py)."""
    state_dict = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if not isinstance(state_dict, dict):
        return state_dict  # Full model object

    num_classes = 10
    if 'fc.8.weight' in state_dict:
        num_classes = state_dict['fc.8.weight'].shape[0]
        raw_model = get_resnet18(num_classes=num_classes)
    elif 'fc.weight' in state_dict:
        num_classes = state_dict['fc.weight'].shape[0]
        from torchvision.models import resnet18 as torchvision_resnet18
        raw_model = torchvision_resnet18(weights=None)
        raw_model.fc = nn.Linear(raw_model.fc.in_features, num_classes)
    else:
        raw_model = get_resnet18(num_classes=10)

    if 'conv1.weight' in state_dict:
        k = state_dict['conv1.weight'].shape[2]
        if k == 3:
            raw_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            raw_model.maxpool = nn.Identity()
        elif k == 7 and raw_model.conv1.kernel_size == (3, 3):
            raw_model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            from torchvision.models import resnet18 as _r18
            raw_model.maxpool = _r18(weights=None).maxpool

    try:
        raw_model.load_state_dict(state_dict)
    except Exception:
        raw_model.load_state_dict(state_dict, strict=False)
    return raw_model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dir = "models"
    # Skip ONNX (too slow in emulation, only .pth)
    model_paths = sorted(glob.glob(os.path.join(model_dir, "*.pth")))

    if not model_paths:
        print(f"No .pth models found in {model_dir}")
        return

    print(f"Found {len(model_paths)} models to audit")
    engine = RiskFusionEngine(use_meta_classifier=False)
    target_class = 0
    X, y = [], []

    # Pre-load datasets ONCE (fast: 320 train, 192 test)
    print("Loading CIFAR-10 datasets (reduced)...")
    train_loader, test_clean, test_poisoned = get_cifar10_dataloaders(
        batch_size=32, poison_ratio=0.1, target_class=target_class, trigger_type="checkerboard"
    )
    train_loader = _limit_loader(train_loader, 320)
    test_clean = _limit_loader(test_clean, 192)
    test_poisoned = _limit_loader(test_poisoned, 192)
    print(f"  train={len(train_loader.dataset)}, clean={len(test_clean.dataset)}, poisoned={len(test_poisoned.dataset)}")

    for model_path in model_paths:
        name = os.path.basename(model_path)
        label = 1 if any(w in name.lower() for w in ["poisoned", "poison", "malicious"]) else 0
        print(f"\n{'='*60}")
        print(f"[{len(X)+1}/{len(model_paths)}] {name} (label={'POISONED' if label else 'CLEAN'})")
        print(f"{'='*60}")

        try:
            raw_model = load_model(model_path, device)
            model = TrojAI_ModelWrapper(raw_model, device)
            model.eval()

            # Signal 1: Blackbox — skip for speed (set 0)
            blackbox_risk = 0.0

            # Signal 2: Behavioral Backdoor (3 batches)
            behavior_risk = 0.0
            try:
                from celery_worker import _behavioral_backdoor_probe
                b = _behavioral_backdoor_probe(model, test_clean, test_poisoned, target_class, device, num_batches=3)
                behavior_risk = min(max(float(b.get('behavioral_backdoor_risk', 0.0)), 0.0), 1.0)
                print(f"  Behavioral: ASR={b.get('attack_success_rate',0):.3f}, risk={behavior_risk:.3f}")
            except Exception as e:
                print(f"  Behavioral: FAILED ({e})")

            # Signal 3: Neural Cleanse (1 epoch, 1 restart)
            nc_risk = 0.0
            try:
                nc = NeuralCleanse(model, device, num_classes=10)
                _, _, _, _, anomaly_indices = nc.detect(test_clean, epochs=1, target_class=target_class)
                nc_risk = engine.normalize_neural_cleanse(anomaly_indices.tolist() if len(anomaly_indices) > 0 else [])
                print(f"  NC: anomaly_idx={anomaly_indices.tolist()}, risk={nc_risk:.3f}")
            except Exception as e:
                print(f"  NC: FAILED ({e})")

            # Signal 4: STRIP (8 samples, 16 perturbations)
            strip_risk = 0.0
            try:
                strip = STRIP(model, device, test_clean.dataset)
                n = min(8, len(test_clean.dataset))
                clean_e = [strip.calculate_entropy(test_clean.dataset[i][0].to(device), num_samples=16) for i in range(n)]
                poison_e = [strip.calculate_entropy(test_poisoned.dataset[i][0].to(device), num_samples=16) for i in range(n)]
                thr = np.percentile(clean_e, 5)
                fa = sum(1 for e in clean_e if e < thr) / max(len(clean_e), 1)
                fr = sum(1 for e in poison_e if e >= thr) / max(len(poison_e), 1)
                strip_risk = engine.normalize_strip(fr, fa)
                print(f"  STRIP: clean_mean={np.mean(clean_e):.3f}, poison_mean={np.mean(poison_e):.3f}, risk={strip_risk:.3f}")
            except Exception as e:
                print(f"  STRIP: FAILED ({e})")

            # Signal 5: Activation Clustering (no t-SNE, no secondary layer)
            ac_risk = 0.0
            try:
                ac = ActivationClustering(model, device, feature_layer_name=model.feature_layer_name)
                score_ac, _, _, _ = ac.detect(train_loader, target_class=target_class, include_tsne=False, include_secondary_layer=False)
                ac_risk = engine.normalize_clustering(score_ac)
                ac.remove_hook()
                print(f"  AC: silhouette={score_ac:.4f}, risk={ac_risk:.3f}")
            except Exception as e:
                print(f"  AC: FAILED ({e})")

            # Signal 6: Weight Analysis
            wa_risk = 0.0
            try:
                wa = WeightAnalysis(model, device)
                wa_idx = wa.detect()
                wa_risk = engine.normalize_weight_analysis(wa_idx)
                print(f"  WA: max_anomaly={np.max(wa_idx) if len(wa_idx)>0 else 0:.3f}, risk={wa_risk:.3f}")
            except Exception as e:
                print(f"  WA: FAILED ({e})")

            # Signal 7: NTP (3 batches)
            ntp_risk = 0.0
            try:
                ntp = NaturalTrojanProfiler(model, device)
                sens = ntp.profile_shortcuts(test_clean, num_batches=3)
                ntp_risk = min(max(sens * 1.5, 0.0), 1.0)
                print(f"  NTP: sensitivity={sens:.4f}, risk={ntp_risk:.3f}")
            except Exception as e:
                print(f"  NTP: FAILED ({e})")

            # Signal 8: Gradient Similarity (6 samples)
            grad_risk = 0.0
            try:
                gs = GradientSimilarity(model, device)
                sim = gs.detect(test_clean, target_class=target_class, num_samples=6)
                grad_risk = engine.normalize_gradient_similarity(sim)
                print(f"  Grad: sim={sim:.4f}, risk={grad_risk:.3f}")
            except Exception as e:
                print(f"  Grad: FAILED ({e})")

            # Signal 9: Spectral Signatures
            spectral_risk = 0.0
            try:
                sp = SpectralSignatures(model, device)
                result = sp.detect(train_loader, target_class=target_class)
                if isinstance(result, tuple) and len(result) >= 4:
                    spectral_risk = engine.normalize_spectral_signatures(float(result[3]))
                sp.remove_hook()
                print(f"  Spectral: score={result[3] if isinstance(result,tuple) and len(result)>=4 else 0:.4f}, risk={spectral_risk:.3f}")
            except Exception as e:
                print(f"  Spectral: FAILED ({e})")

            # Signal 10: CDA (3 batches)
            cda_risk = 0.0
            try:
                cda = ConfidenceDistributionAnalysis(model, device)
                cr, _ = cda.detect(test_clean, target_class=target_class, num_batches=3)
                cda_risk = min(max(float(cr), 0.0), 1.0)
                print(f"  CDA: risk={cda_risk:.3f}")
            except Exception as e:
                print(f"  CDA: FAILED ({e})")

            feat = [blackbox_risk, behavior_risk, nc_risk, strip_risk, ac_risk,
                    wa_risk, ntp_risk, grad_risk, spectral_risk, cda_risk]
            X.append(feat)
            y.append(label)
            print(f"  => FEATURES: {[f'{v:.3f}' for v in feat]}")

        except Exception as e:
            print(f"  LOAD FAILED: {e}")

    if not X:
        print("\nNo models audited successfully!")
        return

    X = np.array(X)
    y = np.array(y)
    print(f"\n{'='*60}")
    print(f"Training Meta-Classifier on {len(X)} samples ({sum(y)} poisoned, {len(y)-sum(y)} clean)")
    print(f"Feature matrix shape: {X.shape}")
    print(f"{'='*60}")

    # Use /tmp for Docker read-only filesystem compatibility
    out_dir = "/tmp" if not os.access(".", os.W_OK) else "."
    npz_path = os.path.join(out_dir, "meta_training_data.npz")
    pkl_path = os.path.join(out_dir, "meta_classifier.pkl")

    np.savez(npz_path, X=X, y=y)
    meta_clf = RiskMetaClassifier(model_path=pkl_path)
    meta_clf.train(X, y)
    print(f"\n✅ {pkl_path} saved ({os.path.getsize(pkl_path)} bytes)")
    print(f"✅ {npz_path} saved")


if __name__ == "__main__":
    main()
