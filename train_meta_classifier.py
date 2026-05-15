import os
import torch
import numpy as np
import pickle
import glob
from defenses import (
    NeuralCleanse, STRIP, ActivationClustering, WeightAnalysis,
    NaturalTrojanProfiler, GradientSimilarity, SpectralSignatures,
    ConfidenceDistributionAnalysis, RiskFusionEngine, RiskMetaClassifier,
)
from trojai_model_wrapper import TrojAI_ModelWrapper
from dataset import get_cifar10_dataloaders
from trojai_dataset import get_trojai_dataloader
from models import get_resnet18

def get_model_input_size(model_path):
    """Identify expected input size based on architecture name."""
    name = model_path.lower()
    if "inception" in name:
        return (299, 299)
    if "densenet" in name or "resnet50" in name:
        return (224, 224)
    return (32, 32) # Default for our CIFAR ResNet18

def generate_training_data(model_dir="models", output_file="meta_training_data.npz"):
    """
    Scans a directory for models, runs the full 10-signal defense suite, and saves results.
    Feature vector: [blackbox, behavioral, nc, strip, ac, wa, ntp, gradient, spectral, cda]
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    models = glob.glob(os.path.join(model_dir, "*.pth")) + glob.glob(os.path.join(model_dir, "*.onnx"))
    
    if not models:
        print(f"No models found in {model_dir}")
        return
    
    X = []
    y = []
    
    engine = RiskFusionEngine(use_meta_classifier=False) # Use static for normalization
    
    for model_path in models:
        print(f"\n[Meta-Gen] Auditing: {os.path.basename(model_path)}")
        label = 1 if any(w in model_path.lower() for w in ["poisoned", "poison", "malicious"]) else 0
        input_size = get_model_input_size(model_path)
        target_class = 0  # Default target for meta-training sweeps
        
        try:
            # Load model using same dynamic adaptation as celery_worker.py
            is_onnx = model_path.lower().endswith('.onnx')
            
            if is_onnx:
                from celery_worker import ONNXModelWrapper
                raw_model = ONNXModelWrapper(model_path)
            else:
                state_dict = torch.load(model_path, map_location=device, weights_only=False)
                
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                
                if not isinstance(state_dict, dict):
                    raw_model = state_dict  # Full model object
                else:
                    # ── DYNAMIC ARCHITECTURE ADAPTATION ──
                    num_classes = 10
                    if 'fc.8.weight' in state_dict:
                        num_classes = state_dict['fc.8.weight'].shape[0]
                        raw_model = get_resnet18(num_classes=num_classes)
                    elif 'fc.weight' in state_dict:
                        num_classes = state_dict['fc.weight'].shape[0]
                        from torchvision.models import resnet18 as torchvision_resnet18
                        raw_model = torchvision_resnet18(weights=None)
                        raw_model.fc = torch.nn.Linear(raw_model.fc.in_features, num_classes)
                    else:
                        raw_model = get_resnet18(num_classes=10)
                    
                    # Adapt conv1/maxpool for CIFAR vs ImageNet
                    if 'conv1.weight' in state_dict:
                        ckpt_conv_shape = state_dict['conv1.weight'].shape
                        if ckpt_conv_shape[2] == 3:  # 3x3 kernel (CIFAR style)
                            raw_model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                            raw_model.maxpool = torch.nn.Identity()
                        elif ckpt_conv_shape[2] == 7:  # 7x7 kernel (Standard style)
                            if isinstance(raw_model.conv1, torch.nn.Conv2d) and raw_model.conv1.kernel_size == (3, 3):
                                raw_model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
                                from torchvision.models import resnet18 as torchvision_resnet18
                                std_resnet = torchvision_resnet18(weights=None)
                                raw_model.maxpool = std_resnet.maxpool
                    
                    try:
                        raw_model.load_state_dict(state_dict)
                    except Exception:
                        raw_model.load_state_dict(state_dict, strict=False)
                
            model = TrojAI_ModelWrapper(raw_model, device)
            model.eval()
            
            # Select correct dataloaders for this model's scale
            if input_size == (32, 32):
                train_data, test_clean, test_poisoned = get_cifar10_dataloaders(
                    batch_size=32, poison_ratio=0.1, target_class=target_class, trigger_type="checkerboard"
                )
            else:
                test_clean = get_trojai_dataloader("sample_external_models", batch_size=16, image_size=input_size)
                train_data = test_clean
                test_poisoned = test_clean  # Fallback for non-CIFAR
            
            # ── Signal 1: Blackbox Sweep (placeholder — requires separate sweep fn) ──
            blackbox_risk = 0.0
            try:
                from blackbox_sweep import black_box_trigger_sweep as _bbsweep
                sweep = _bbsweep(model, test_clean, device=device, num_batches=4)
                blackbox_risk = min(max(float(sweep.get('blackbox_sweep_risk', 0.0)), 0.0), 1.0)
            except Exception:
                pass

            # ── Signal 2: Behavioral Backdoor ──
            behavior_risk = 0.0
            try:
                from celery_worker import _behavioral_backdoor_probe
                behavior = _behavioral_backdoor_probe(model, test_clean, test_poisoned, target_class, device, num_batches=6)
                behavior_risk = min(max(float(behavior.get('behavioral_backdoor_risk', 0.0)), 0.0), 1.0)
            except Exception:
                pass

            # ── Signal 3: Neural Cleanse ──
            nc_risk = 0.0
            try:
                nc = NeuralCleanse(model, device, num_classes=10 if input_size == (32, 32) else 1000)
                flagged_nc, sizes, masks, patterns, anomaly_indices = nc.detect(test_clean, epochs=2, target_class=target_class)
                nc_risk = engine.normalize_neural_cleanse(anomaly_indices.tolist() if len(anomaly_indices) > 0 else [])
            except Exception:
                pass

            # ── Signal 4: STRIP (real entropy computation) ──
            strip_risk = 0.0
            try:
                strip = STRIP(model, device, test_clean.dataset)
                n_strip = min(16, len(test_clean.dataset))
                clean_entropies = [
                    strip.calculate_entropy(test_clean.dataset[i][0].to(device), num_samples=min(32, len(test_clean.dataset)))
                    for i in range(n_strip)
                ]
                n_strip_p = min(16, len(test_poisoned.dataset))
                poison_entropies = [
                    strip.calculate_entropy(test_poisoned.dataset[i][0].to(device), num_samples=min(32, len(test_clean.dataset)))
                    for i in range(n_strip_p)
                ]
                clean_p5 = np.percentile(clean_entropies, 5)
                threshold = clean_p5
                fa_count = sum(1 for e in clean_entropies if e < threshold)
                fr_count = sum(1 for e in poison_entropies if e >= threshold)
                fa_ratio = fa_count / max(len(clean_entropies), 1)
                fr_ratio = fr_count / max(len(poison_entropies), 1)
                strip_risk = engine.normalize_strip(fr_ratio, fa_ratio)
            except Exception:
                pass

            # ── Signal 5: Activation Clustering ──
            ac_risk = 0.0
            try:
                ac = ActivationClustering(model, device, feature_layer_name=model.feature_layer_name)
                score_ac, _, _, _ = ac.detect(train_data, target_class=target_class, include_tsne=False, include_secondary_layer=False)
                ac_risk = engine.normalize_clustering(score_ac)
                ac.remove_hook()
            except Exception:
                pass
            
            # ── Signal 6: Weight Analysis ──
            wa_risk = 0.0
            try:
                wa = WeightAnalysis(model, device)
                wa_indices = wa.detect()
                wa_risk = engine.normalize_weight_analysis(wa_indices)
            except Exception:
                pass
            
            # ── Signal 7: Natural Trojan Profiler ──
            ntp_risk = 0.0
            try:
                ntp = NaturalTrojanProfiler(model, device)
                ntp_sensitivity = ntp.profile_shortcuts(test_clean, num_batches=6)
                ntp_risk = min(max(ntp_sensitivity * 1.5, 0.0), 1.0)
            except Exception:
                pass

            # ── Signal 8: Gradient Similarity ──
            grad_risk = 0.0
            try:
                gs = GradientSimilarity(model, device)
                grad_sim = gs.detect(test_clean, target_class=target_class, num_samples=12)
                grad_risk = engine.normalize_gradient_similarity(grad_sim)
            except Exception:
                pass

            # ── Signal 9: Spectral Signatures ──
            spectral_risk = 0.0
            try:
                spectral = SpectralSignatures(model, device)
                spectral_result = spectral.detect(train_data, target_class=target_class)
                if isinstance(spectral_result, tuple) and len(spectral_result) >= 4:
                    spectral_risk = engine.normalize_spectral_signatures(float(spectral_result[3]))
                spectral.remove_hook()
            except Exception:
                pass

            # ── Signal 10: Confidence Distribution Analysis ──
            cda_risk = 0.0
            try:
                cda = ConfidenceDistributionAnalysis(model, device)
                cda_risk_raw, _ = cda.detect(test_clean, target_class=target_class, num_batches=5)
                cda_risk = min(max(float(cda_risk_raw), 0.0), 1.0)
            except Exception:
                pass

            # Construct 10-dimensional feature vector (matches RiskFusionEngine signal order)
            feature_vector = [
                blackbox_risk, behavior_risk, nc_risk, strip_risk, ac_risk,
                wa_risk, ntp_risk, grad_risk, spectral_risk, cda_risk
            ]
            X.append(feature_vector)
            y.append(label)
            print(f"   Label: {label}, Features: {[f'{v:.3f}' for v in feature_vector]}")
            
        except Exception as e:
            print(f"   Failed to audit {model_path}: {e}")
            
    if X:
        np.savez(output_file, X=np.array(X), y=np.array(y))
        print(f"\nSaved {len(X)} samples ({sum(y)} poisoned, {len(y) - sum(y)} clean) to {output_file}")
    return np.array(X), np.array(y)

def train_meta_classifier(data_file="meta_training_data.npz"):
    if not os.path.exists(data_file):
        print("Data file not found. Generate it first.")
        return
    
    data = np.load(data_file)
    X, y = data['X'], data['y']
    
    print(f"Training on {len(X)} samples with {X.shape[1]} features")
    print(f"  Positive (poisoned): {sum(y)}, Negative (clean): {len(y) - sum(y)}")
    
    meta_clf = RiskMetaClassifier()
    meta_clf.train(X, y)
    print("Meta-Classifier successfully trained and saved to meta_classifier.pkl")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['gen', 'train', 'both'], default='both')
    parser.add_argument('--model-dir', default='models')
    args = parser.parse_args()
    
    if args.mode in ['gen', 'both']:
        generate_training_data(args.model_dir)
    if args.mode in ['train', 'both']:
        train_meta_classifier()
