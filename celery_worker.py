import os
import sys

# Ensure the local directory is in the python path for unpickling bundled modules
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
from celery import Celery
import time
import uuid
import base64
import onnx
from onnx2torch import convert

# Import our MLSecOps components
from defenses import NeuralCleanse, STRIP, ActivationClustering, WeightAnalysis, NaturalTrojanProfiler, GradientSimilarity, SpectralSignatures, ConfidenceDistributionAnalysis, RiskFusionEngine
from dataset import get_cifar10_dataloaders
from trojai_dataset import get_trojai_dataloader
from trojai_model_wrapper import TrojAI_ModelWrapper
from gradcam_utils import GradCAM
from captum_utils import CaptumSaliency
from models import get_resnet18
from blackbox_sweep import black_box_trigger_sweep as _black_box_trigger_sweep
import datetime
import io
import pickle
import random
import logging

logger = logging.getLogger(__name__)

class RestrictedUnpickler(pickle.Unpickler):
    ALLOWED_GLOBALS = {
        "torch", "torch.nn", "torch.nn.modules", "torch.nn.modules.module",
        "torch.nn.modules.container", "torch.nn.modules.linear",
        "torch.nn.modules.conv", "torch.nn.modules.batchnorm",
        "torch.nn.modules.activation", "torch.nn.modules.pooling",
        "torch.nn.modules.dropout", "torch._utils",
        "torchvision.models", "torchvision.models.resnet",
        "collections", "numpy", "numpy.core.multiarray",
        "_codecs", "builtins",
        "pytorch_cifar_models", "pytorch_cifar_models.resnet",
        "pytorch_cifar_models.vgg", "pytorch_cifar_models.mobilenetv2",
        "pytorch_cifar_models.shufflenetv2", "pytorch_cifar_models.repvgg",
        "pytorch_cifar_models.vit",
    }
    def find_class(self, module, name):
        if any(module == m or module.startswith(m + ".") for m in self.ALLOWED_GLOBALS):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Blocked unpickling of {module}.{name}")

def _register_cifar_safe_globals():
    """Register bundled pytorch_cifar_models classes as torch safe globals."""
    try:
        import pytorch_cifar_models.resnet as _r
        import pytorch_cifar_models.vgg as _v
        import pytorch_cifar_models.mobilenetv2 as _m
        import pytorch_cifar_models.shufflenetv2 as _s
        import pytorch_cifar_models.repvgg as _rp
        import pytorch_cifar_models.vit as _vit
        classes = [
            _r.BasicBlock, _r.CifarResNet,
            _v.VGG,
            _m.ConvBNActivation, _m.InvertedResidual, _m.MobileNetV2,
            _s.InvertedResidual, _s.ShuffleNetV2,
            _rp.RepVGGBlock, _rp.RepVGG,
            _vit.Attention, _vit.MLP, _vit.Embeddings, _vit.Block,
            _vit.Encoder, _vit.Transformer, _vit.VisionTransformer,
        ]
        torch.serialization.add_safe_globals(classes)
    except Exception as e:
        logger.warning(f"Could not register cifar safe globals: {e}")

_register_cifar_safe_globals()

def safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        try:
            # weights_only=False for trusted bundled-class checkpoints
            return torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            with open(path, "rb") as f:
                return RestrictedUnpickler(f).load()

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _check_ood(model, dataloader, device="cpu", msp_threshold=0.1):
    model.eval()
    msps = []
    with torch.no_grad():
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            try:
                logits = model(imgs)
                probs = torch.softmax(logits, dim=-1)
                msps.extend(probs.max(dim=-1).values.cpu().tolist())
            except Exception:
                break
            if len(msps) >= 256:
                break
    mean_msp = float(np.mean(msps)) if msps else 0.0
    return {"ood_detected": mean_msp < msp_threshold, "mean_msp": round(mean_msp, 4)}

# Initialize Celery app
# Defaults to localhost for both broker and result backend. You need Redis running locally.
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', '')
_redis_auth = f':{REDIS_PASSWORD}@' if REDIS_PASSWORD else ''
celery_app = Celery(
    'mlsecops_tasks',
    broker=f'redis://{_redis_auth}{REDIS_HOST}:6379/0',
    backend=f'redis://{_redis_auth}{REDIS_HOST}:6379/1'
)

# Optional: configure celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],  
    result_serializer='json',
    timezone='America/Los_Angeles',
    enable_utc=True,
)

import onnxruntime as ort
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import tempfile
import shutil
import datetime

SCAN_PROFILES = {
    "fast": {
        "train_samples": 640,
        "test_samples": 384,
        "nc_discovery_epochs": 1,
        "nc_targeted_epochs": 2,
        "strip_samples": 16,
        "strip_perturbations": 32,
        "strip_multi_samples": 6,
        "strip_multi_perturbations": 12,
        "confidence_batches": 5,
        "gradient_samples": 12,
        "behavior_batches": 6,
        "blackbox_sweep_batches": 4,
        "include_tsne": False,
    },
    "balanced": {
        "train_samples": 1600,
        "test_samples": 768,
        "nc_discovery_epochs": 2,
        "nc_targeted_epochs": 3,
        "strip_samples": 32,
        "strip_perturbations": 64,
        "strip_multi_samples": 12,
        "strip_multi_perturbations": 24,
        "confidence_batches": 10,
        "gradient_samples": 24,
        "behavior_batches": 12,
        "blackbox_sweep_batches": 8,
        "include_tsne": False,
    },
    "enterprise": {
        "train_samples": 2400,
        "test_samples": 1280,
        "nc_discovery_epochs": 2,
        "nc_targeted_epochs": 4,
        "strip_samples": 40,
        "strip_perturbations": 96,
        "strip_multi_samples": 16,
        "strip_multi_perturbations": 32,
        "confidence_batches": 12,
        "gradient_samples": 32,
        "behavior_batches": 16,
        "blackbox_sweep_batches": 12,
        "include_tsne": False,
    },
    "thorough": {
        "train_samples": None,
        "test_samples": None,
        "nc_discovery_epochs": 2,
        "nc_targeted_epochs": 5,
        "strip_samples": 50,
        "strip_perturbations": 128,
        "strip_multi_samples": 20,
        "strip_multi_perturbations": 32,
        "confidence_batches": 15,
        "gradient_samples": 40,
        "behavior_batches": 25,
        "blackbox_sweep_batches": 20,
        "include_tsne": True,
    },
}


def _scan_profile():
    profile_name = os.environ.get("SCAN_PROFILE", "enterprise").strip().lower()
    profile = SCAN_PROFILES.get(profile_name, SCAN_PROFILES["enterprise"]).copy()
    profile["name"] = profile_name if profile_name in SCAN_PROFILES else "enterprise"
    if os.environ.get("SCAN_INCLUDE_TSNE", "").strip().lower() in {"1", "true", "yes"}:
        profile["include_tsne"] = True
    return profile


def _limit_loader(loader, max_samples, shuffle=False):
    if max_samples is None:
        return loader
    dataset = loader.dataset
    sample_count = min(max_samples, len(dataset))
    if sample_count >= len(dataset):
        return loader
    subset = Subset(dataset, range(sample_count))
    return DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _behavioral_backdoor_probe(model, clean_loader, poisoned_loader, target_class, device, num_batches):
    clean_total = 0
    clean_correct = 0
    clean_target_hits = 0
    poison_total = 0
    poison_target_hits = 0

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(clean_loader):
            if i >= num_batches:
                break
            inputs, labels = batch[0].to(device), batch[1].to(device)
            preds = torch.argmax(model(inputs), dim=1)
            clean_total += labels.numel()
            clean_correct += (preds == labels).sum().item()
            clean_target_hits += (preds == target_class).sum().item()

        for i, batch in enumerate(poisoned_loader):
            if i >= num_batches:
                break
            inputs = batch[0].to(device)
            preds = torch.argmax(model(inputs), dim=1)
            poison_total += preds.numel()
            poison_target_hits += (preds == target_class).sum().item()

    clean_accuracy = clean_correct / max(clean_total, 1)
    clean_target_rate = clean_target_hits / max(clean_total, 1)
    attack_success_rate = poison_target_hits / max(poison_total, 1)
    target_lift = max(0.0, attack_success_rate - clean_target_rate)

    # High ASR alone can be misleading for biased or broken models. The lift over clean
    # target predictions is the stronger behavioral backdoor signal.
    lift_risk = min(max((target_lift - 0.20) / 0.60, 0.0), 1.0)
    asr_risk = min(max((attack_success_rate - 0.50) / 0.45, 0.0), 1.0)
    behavioral_risk = max(lift_risk, 0.75 * asr_risk if target_lift > 0.10 else 0.0)

    return {
        "clean_accuracy": float(clean_accuracy),
        "clean_target_rate": float(clean_target_rate),
        "attack_success_rate": float(attack_success_rate),
        "target_lift": float(target_lift),
        "behavioral_backdoor_risk": float(behavioral_risk),
    }


class ONNXModelWrapper(nn.Module):
    """
    Wraps an ONNX model inference session into a PyTorch nn.Module API
    so that our existing defense systems can run standard forward() passes.
    """
    def __init__(self, onnx_path):
        super().__init__()
        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        
    def forward(self, x):
        # Convert PyTorch tensor to numpy
        if x.requires_grad:
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x.cpu().numpy()
            
        # Run ONNX inference
        outputs = self.session.run(None, {self.input_name: x_np})
        
        # We only care about the single output logits right now
        # Convert back to torch tensor so PyTorch loss functions work
        out_tensor = torch.tensor(outputs[0])
        # Force require_grad if the input needed it (some defenses like NC use gradients)
        if x.requires_grad:
            out_tensor.requires_grad_(True)
        return out_tensor

def validate_model_file(model_path):
    """
    Performs basic sanity checks on the model file before attempting to load.
    Returns (is_valid, error_message)
    """
    if not os.path.exists(model_path):
        return False, "Model file not found."
    
    filesize = os.path.getsize(model_path)
    if filesize < 100: # Arbitrary minimum size for a valid model
        return False, f"Model file is too small ({filesize} bytes). Likely an invalid or corrupted upload."
    
    # Check for ONNX magic number (first few bytes)
    # Actually, ONNX doesn't have a simple magic number, but we can check extension.
    # For PyTorch, we check for 'PK' (ZIP) if it's a modern format.
    with open(model_path, 'rb') as f:
        header = f.read(4)
        if header == b'PK\x03\x04': # ZIP archive (PyTorch v1.6+)
             return True, ""
        # Check for legacy pickle magic
        if header.startswith(b'\x80\x02') or header.startswith(b'\x80\x03') or header.startswith(b'\x80\x04'):
             return True, ""
        
    # If it's ONNX, let the runtime attempt to load it.
    if model_path.lower().endswith('.onnx'):
        return True, ""
        
    # If we get here and it's small or looks like text, it's likely a failure
    try:
        with open(model_path, 'r') as f:
            content = f.read(50)
            if "dummy content" in content:
                return False, "Detected dummy placeholder text file instead of a valid neural network model."
    except:
        pass

    return True, ""

@celery_app.task(bind=True, name='mlsecops.scan_model')
def run_model_scan_task(self, model_path, target_class, trigger_type):
    """
    Asynchronous task to run the full Trojan detection suite.
    """
    seed_everything(int(os.environ.get("SCAN_SEED", "42")))
    profile = _scan_profile()
    original_auto_detect = target_class == -1
    self.update_state(state='PROGRESS', meta={'message': 'Loading Model...'})
    logger.info(f"[{self.request.id}] Starting {profile['name']} scan on {os.path.basename(model_path)}")
    
    # Use CUDA on Nautilus for massive speedup (50x+)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"[{self.request.id}] Using device: {device}")
    
    # 1. Load the Model (Support both .pth and .onnx)
    is_onnx = model_path.lower().endswith('.onnx')
    
    if is_onnx:
        self.update_state(state='PROGRESS', meta={'message': 'Loading ONNX Runtime Engine...'})
        logger.info(f"[{self.request.id}] Loading ONNX model...")
        raw_model = ONNXModelWrapper(model_path)
    else:
        # Standard PyTorch Checkpoint
        self.update_state(state='PROGRESS', meta={'message': 'Validating Model Format...'})
        is_valid, err_msg = validate_model_file(model_path)
        if not is_valid:
            raise ValueError(err_msg)

        try:
            state_dict = safe_torch_load(model_path)
        except Exception as full_err:
            raise ValueError(f"Model Load Failure: The uploaded file requires custom Python classes that are not present in this environment (e.g., '{full_err}'). Please ensure you upload standard architectural checkpoints.")

        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        if not isinstance(state_dict, dict):
            raw_model = state_dict # Full model object
        else:
            # ── DYNAMIC ARCHITECTURE ADAPTATION ──
            # Determine classification head type
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
                # Default to our enhanced head as a guess
                raw_model = get_resnet18(num_classes=10)

            # Determine conv1/maxpool settings (CIFAR vs ImageNet)
            if 'conv1.weight' in state_dict:
                ckpt_conv_shape = state_dict['conv1.weight'].shape
                if ckpt_conv_shape[2] == 3: # 3x3 kernel (CIFAR style)
                    logger.info(f"[{self.request.id}] Detected 3x3 conv1 kernel. Adapting architecture for small-input ResNet.")
                    raw_model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                    raw_model.maxpool = torch.nn.Identity()
                elif ckpt_conv_shape[2] == 7: # 7x7 kernel (Standard style)
                    logger.info(f"[{self.request.id}] Detected 7x7 conv1 kernel. Ensuring standard ResNet architecture.")
                    # If raw_model was from get_resnet18 (which uses 3x3), revert it
                    if isinstance(raw_model.conv1, torch.nn.Conv2d) and raw_model.conv1.kernel_size == (3, 3):
                        raw_model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
                        from torchvision.models import resnet18 as torchvision_resnet18
                        std_resnet = torchvision_resnet18(weights=None)
                        raw_model.maxpool = std_resnet.maxpool

            # Finally, load the state dict
            try:
                raw_model.load_state_dict(state_dict)
                logger.info(f"[{self.request.id}] Flexible load SUCCESSFUL.")
            except Exception as final_load_err:
                # Last resort: Try strict=False if it's just minor naming issues
                logger.warning(f"[{self.request.id}] Strict load failed: {final_load_err}. Retrying with strict=False...")
                raw_model.load_state_dict(state_dict, strict=False)


    # Wrap it to make it compatible with our defenses universally
    model = TrojAI_ModelWrapper(raw_model, device=device)
    model.to(device)
    model.eval()
    
    # Identify input size for dynamic scaling
    if "inception" in model_path.lower():
        input_size = (299, 299)
    elif "densenet" in model_path.lower() or "resnet50" in model_path.lower():
        input_size = (224, 224)
    else:
        input_size = (32, 32) # Default for CIFAR ResNet18
        
    # 2. Pre-Load Clean Dataset for Neural Cleanse/NTP
    self.update_state(state='PROGRESS', meta={'message': f'Loading {input_size[0]}x{input_size[1]} validation datasets...'})
    
    if input_size == (32, 32):
        temp_target = target_class if target_class != -1 else 0
        temp_trigger = trigger_type if trigger_type != "Auto-Detect (Black-Box)" else "checkerboard"
        _, test_clean, _ = get_cifar10_dataloaders(
            batch_size=64, poison_ratio=0.0, target_class=temp_target, trigger_type=temp_trigger
        )
        test_clean = _limit_loader(test_clean, profile["test_samples"])
    else:
        # Check if trojai_data exists (from generate_trojai_samples.py)
        # Otherwise fallback to sample_external_models
        data_path = "trojai_data" if os.path.exists("trojai_data") else "sample_external_models"
        test_clean = get_trojai_dataloader(data_path, batch_size=32, image_size=input_size)
        test_clean = _limit_loader(test_clean, profile["test_samples"])

    # Extract model structural information for report
    inferred_num_classes = "10"
    param_count_str = "N/A"
    architecture_name = "Unknown"
    
    if not is_onnx:
        architecture_name = type(raw_model).__name__
        try:
            total_params = sum(p.numel() for p in raw_model.parameters())
            param_count_str = f"{total_params:,}"
        except: pass
        
        if hasattr(raw_model, 'fc') and hasattr(raw_model.fc, 'out_features'):
            inferred_num_classes = str(raw_model.fc.out_features)
        elif hasattr(raw_model, 'classifier'):
            if hasattr(raw_model.classifier, 'out_features'):
                inferred_num_classes = str(raw_model.classifier.out_features)
            elif isinstance(raw_model.classifier, torch.nn.Sequential):
                for layer in reversed(raw_model.classifier):
                    if hasattr(layer, 'out_features'):
                        inferred_num_classes = str(layer.out_features)
                        break
    else:
        if "resnet" in model_path.lower(): architecture_name = "ResNet (ONNX)"
        else: architecture_name = "ONNX Generic"

    details = {
        'architecture': architecture_name,
        'input_shape': f"3x{input_size[0]}x{input_size[1]}",
        'num_classes': inferred_num_classes,
        'parameter_count': param_count_str,
        'nc_anomaly_indices': [],
        'nc_flagged_classes': [],
        'strip_fr_ratio': 0.0,
        'strip_fa_ratio': 0.0,
        'clustering_silhouette_score': 0.0,
        'wa_anomaly_indices': [],
        'weight_analysis_risk': 0.0,
        'natural_sensitivity': 0.0,
        'gradient_similarity': 0.0,
        'clean_accuracy': 0.0,
        'attack_success_rate': 0.0,
        'target_lift': 0.0,
        'behavioral_backdoor_risk': 0.0,
        'blackbox_sweep_risk': 0.0,
        'blackbox_sweep_target': None,
        'blackbox_sweep_trigger': None,
        'blackbox_sweep_lift': 0.0,
    }
    
    # 3. Neural Cleanse (Run FIRST to Auto-Detect or Target)
    self.update_state(state='PROGRESS', meta={'message': 'Running Neural Cleanse (Reverse-Engineering Triggers)...'})
    try:
        def nc_progress_callback(current, total, class_idx):
            self.update_state(state='PROGRESS', meta={
                'message': f'Neural Cleanse: Class {class_idx} ({current+1}/{total})'
            })
            
        # TrojAI models (1000 classes) cannot be fully swept. We default to targeted scan.
        num_nc_classes = 1000 if input_size[0] > 32 else 10
        nc = NeuralCleanse(model, device, num_classes=num_nc_classes)
        
        # Optimization: If it's a large model, force a targeted scan on class 0 if auto-detect is on
        if target_class == -1 and num_nc_classes > 10:
             logger.info(f"[{self.request.id}] High-Res Model detected. Focusing audit on Class 0 for performance.")
             nc_target = 0
             discovery_epochs = profile["nc_discovery_epochs"]
        else:
             nc_target = None if target_class == -1 else int(target_class)
             discovery_epochs = profile["nc_discovery_epochs"] if target_class == -1 else profile["nc_targeted_epochs"]

        flagged_nc, sizes, masks, nc_patterns, anomaly_indices = nc.detect(test_clean, epochs=discovery_epochs, target_class=nc_target, callback=nc_progress_callback)

        details['nc_anomaly_indices'] = anomaly_indices.tolist() if len(anomaly_indices) > 0 else []
        details['nc_flagged_classes'] = flagged_nc.tolist()
        
        # Discovery Mode Logic
        if target_class == -1:
            if len(flagged_nc) > 0:
                logger.info(f"[{self.request.id}] Auto-Detected Target Class: {flagged_nc[0]}")
                target_class = int(flagged_nc[0])
            else:
                logger.info(f"[{self.request.id}] Neural Cleanse found no dominant trigger. Deferring target selection to black-box sweep.")
        
        if trigger_type == "Auto-Detect (Black-Box)":
            trigger_type = "checkerboard"
            
    except Exception as e:
        logger.warning(f"Neural Cleanse failed: {e}")
        details['nc_anomaly_indices'] = []
        details['nc_flagged_classes'] = []
        if trigger_type == "Auto-Detect (Black-Box)": trigger_type = "checkerboard"

    if original_auto_detect:
        self.update_state(state='PROGRESS', meta={'message': 'Running Black-Box Trigger Sweep...'})
        try:
            sweep = _black_box_trigger_sweep(
                model,
                test_clean,
                device=device,
                num_batches=profile["blackbox_sweep_batches"],
            )
            details.update(sweep)
            if sweep["blackbox_sweep_target"] is not None and (
                target_class == -1 or sweep["blackbox_sweep_risk"] >= 0.20
            ):
                target_class = int(sweep["blackbox_sweep_target"])
                trigger_type = sweep["blackbox_sweep_trigger"] or trigger_type
                logger.info(
                    f"[{self.request.id}] Black-box sweep selected class {target_class} "
                    f"with {trigger_type} risk={sweep['blackbox_sweep_risk']:.3f}"
                )
        except Exception as e:
            logger.warning(f"Black-box trigger sweep failed: {e}")

    if target_class == -1:
        logger.info(f"[{self.request.id}] No target selected by discovery. Defaulting to Class 0.")
        target_class = 0

    # 4. Reload Full Dataset with Confirmed Target Class for remaining defenses
    self.update_state(state='PROGRESS', meta={'message': f'Poisoning datasets for Class {target_class}...'})
    train_loader, test_clean, test_poisoned = get_cifar10_dataloaders(
        batch_size=64, poison_ratio=0.1, target_class=target_class, trigger_type=trigger_type
    )
    train_loader = _limit_loader(train_loader, profile["train_samples"], shuffle=True)
    test_clean = _limit_loader(test_clean, profile["test_samples"])
    test_poisoned = _limit_loader(test_poisoned, profile["test_samples"])

    # 4. STRIP
    self.update_state(state='PROGRESS', meta={'message': 'Running STRIP...'})
    try:
        strip = STRIP(model, device, test_clean.dataset)
        n_strip = min(profile["strip_samples"], len(test_clean.dataset))
        clean_entropies = [
            strip.calculate_entropy(
                test_clean.dataset[i][0].to(device),
                num_samples=min(profile["strip_perturbations"], len(test_clean.dataset)),
            )
            for i in range(n_strip)
        ]

        n_strip_p = min(profile["strip_samples"], len(test_poisoned.dataset))
        poison_entropies = [
            strip.calculate_entropy(
                test_poisoned.dataset[i][0].to(device),
                num_samples=min(profile["strip_perturbations"], len(test_clean.dataset)),
            )
            for i in range(n_strip_p)
        ]

        # Adaptive threshold: use 5th percentile of clean distribution instead of
        # naive average. This prevents false positives on models with naturally low entropy.
        clean_p5 = np.percentile(clean_entropies, 5)
        clean_mean = np.mean(clean_entropies)
        clean_std = np.std(clean_entropies)
        # Gaussian-fit threshold: 2.5σ below clean mean (more statistically robust)
        gaussian_threshold = clean_mean - 2.5 * clean_std
        # Use the more conservative (higher) of the two thresholds
        threshold = max(clean_p5, gaussian_threshold)

        fa_count = sum(1 for e in clean_entropies if e < threshold)
        fr_count = sum(1 for e in poison_entropies if e >= threshold)
        details['strip_fa_ratio'] = fa_count / max(len(clean_entropies), 1)
        details['strip_fr_ratio'] = fr_count / max(len(poison_entropies), 1)

        # Multi-alpha entropy variance: trojaned inputs stay low-entropy across ALL mixing strengths
        n_multi = min(profile["strip_multi_samples"], n_strip_p)
        alpha_means, alpha_vars = zip(*[
            strip.calculate_entropy_variance(
                test_poisoned.dataset[i][0].to(device),
                num_samples=min(profile["strip_multi_perturbations"], len(test_clean.dataset)),
            )
            for i in range(n_multi)
        ])
        details['strip_entropy_variance'] = float(np.mean(alpha_vars))
        details['strip_multi_alpha_mean_entropy'] = float(np.mean(alpha_means))
    except Exception as e:
        logger.warning(f"STRIP failed: {e}")
        details['strip_fr_ratio'] = 0.0
        details['strip_fa_ratio'] = 0.0
        
    # 5. Activation Clustering
    self.update_state(state='PROGRESS', meta={'message': 'Running Activation Clustering & t-SNE...'})
    try:
        ac = ActivationClustering(model, device, feature_layer_name=model.feature_layer_name)
        score_ac, _, _, tsne_b64 = ac.detect(
            train_loader,
            target_class=target_class,
            method='kmeans',
            include_tsne=profile["include_tsne"],
            include_secondary_layer=profile["name"] != "fast",
        )
        details['clustering_silhouette_score'] = float(score_ac)
        details['tsne_plot_b64'] = tsne_b64
        ac.remove_hook()
    except Exception as e:
        logger.warning(f"Activation Clustering failed: {e}")
        details['clustering_silhouette_score'] = 0.0
        details['tsne_plot_b64'] = None

    # 6. Weight Analysis (Chapter 4)
    self.update_state(state='PROGRESS', meta={'message': 'Running Linear Weight Analysis...'})
    try:
        wa = WeightAnalysis(model, device)
        wa_indices = wa.detect()
        details['wa_anomaly_indices'] = wa_indices.tolist() if len(wa_indices) > 0 else []
        details['weight_analysis_risk'] = float(np.max(wa_indices)) if len(wa_indices) > 0 else 0.0
    except Exception as e:
        logger.warning(f"Weight Analysis failed: {e}")
        details['wa_anomaly_indices'] = []
        details['weight_analysis_risk'] = 0.0

    # 6b. Spectral Signatures (SVD-based outlier detection)
    self.update_state(state='PROGRESS', meta={'message': 'Running Spectral Signatures Analysis...'})
    try:
        spectral = SpectralSignatures(model, device)
        spectral_result = spectral.detect(train_loader, target_class=target_class)
        if isinstance(spectral_result, tuple) and len(spectral_result) >= 3:
            spectral_top_k, spectral_tp, spectral_total, spectral_score = spectral_result
            details['spectral_anomaly_score'] = float(spectral_score)
            details['spectral_true_positives'] = int(spectral_tp)
        else:
            details['spectral_anomaly_score'] = 0.0
            details['spectral_true_positives'] = 0
        spectral.remove_hook()
    except Exception as e:
        logger.warning(f"Spectral Signatures failed: {e}")
        details['spectral_anomaly_score'] = 0.0
        details['spectral_true_positives'] = 0

    # 6c. Behavioral Backdoor Probe (direct clean vs triggered behavior)
    self.update_state(state='PROGRESS', meta={'message': 'Measuring Triggered Attack Success...'})
    try:
        behavior = _behavioral_backdoor_probe(
            model,
            test_clean,
            test_poisoned,
            target_class=target_class,
            device=device,
            num_batches=profile["behavior_batches"],
        )
        details.update(behavior)
    except Exception as e:
        logger.warning(f"Behavioral Backdoor Probe failed: {e}")

    # 7. Natural Trojan Profiling (Chapter 7.G)
    self.update_state(state='PROGRESS', meta={'message': 'Profiling Natural Trojans (Bias & Shortcuts)...'})
    try:
        ntp = NaturalTrojanProfiler(model, device)
        natural_sensitivity = ntp.profile_shortcuts(test_clean)
        details['natural_sensitivity'] = float(natural_sensitivity)
    except Exception as e:
        logger.warning(f"Natural Trojan Profiling failed: {e}")
        details['natural_sensitivity'] = 0.0

    # 7b. Confidence Distribution Analysis
    self.update_state(state='PROGRESS', meta={'message': 'Analyzing Confidence Distribution...'})
    try:
        cda = ConfidenceDistributionAnalysis(model, device)
        cda_risk, cda_stats = cda.detect(
            test_clean,
            target_class=target_class,
            num_batches=profile["confidence_batches"],
        )
        details['confidence_distribution_risk'] = float(cda_risk)
        details['confidence_bimodality_coeff'] = cda_stats.get('bimodality_coeff', 0.0)
    except Exception as e:
        logger.warning(f"Confidence Distribution Analysis failed: {e}")
        details['confidence_distribution_risk'] = 0.0
        details['confidence_bimodality_coeff'] = 0.0

    # 8. Gradient Similarity Analysis (New)
    self.update_state(state='PROGRESS', meta={'message': 'Running Gradient Similarity Analysis...'})
    nc_mask = masks[0] if 'masks' in dir() and masks else None
    nc_pattern = nc_patterns[0] if 'nc_patterns' in dir() and nc_patterns else None
    try:
        gs = GradientSimilarity(model, device)
        grad_sim = gs.detect(
            test_clean,
            target_class=target_class,
            num_samples=profile["gradient_samples"],
            trigger_mask=nc_mask,
            trigger_pattern=nc_pattern,
        )
        details['gradient_similarity'] = float(grad_sim)
    except Exception as e:
        logger.warning(f"Gradient Similarity failed: {e}")
        details['gradient_similarity'] = 0.0

    # 9. Fusion Engine
    self.update_state(state='PROGRESS', meta={'message': 'Fusing Risk Telemetry (10-signal fusion)...'})
    engine = RiskFusionEngine(use_meta_classifier=True)
    fusion_score, fusion_details = engine.calculate_unified_risk(
        nc_anomaly_indices=details['nc_anomaly_indices'],
        strip_fr_ratio=details['strip_fr_ratio'],
        strip_fa_ratio=details['strip_fa_ratio'],
        clustering_score=details['clustering_silhouette_score'],
        wa_anomaly_indices=details['wa_anomaly_indices'],
        natural_sensitivity=details['natural_sensitivity'],
        gradient_similarity=details['gradient_similarity'],
        spectral_anomaly_score=details.get('spectral_anomaly_score', 0.0),
        confidence_distribution_risk=details.get('confidence_distribution_risk', 0.0),
        behavioral_backdoor_risk=details.get('behavioral_backdoor_risk', 0.0),
        blackbox_sweep_risk=details.get('blackbox_sweep_risk', 0.0),
        strip_entropy_variance=details.get('strip_entropy_variance'),
    )
    details.update(fusion_details)

    # 8. Forensic Reasoning Generation (New)
    forensic_analysis = []
    
    # Neural Cleanse Reasoning
    if details.get('blackbox_sweep_risk', 0.0) > 0.2:
        forensic_analysis.append({
            "method": "Black-Box Trigger Sweep",
            "layer": "Behavioral Output Distribution",
            "reasoning": (
                f"Observed prediction collapse toward Class {details.get('blackbox_sweep_target')} "
                f"under the {details.get('blackbox_sweep_trigger')} probe. Target lift was "
                f"{details.get('blackbox_sweep_lift', 0.0):.2f}, indicating the model may contain "
                "a trigger-conditioned shortcut even before white-box attribution."
            ),
            "severity": "CRITICAL" if details.get('blackbox_sweep_risk', 0.0) > 0.75 else "HIGH"
        })

    if details['nc_anomaly_indices']:
        forensic_analysis.append({
            "method": "Neural Cleanse",
            "layer": "Trigger Inversion (Feature Layer)",
            "reasoning": f"Identified a persistent latent trigger pattern for Class {target_class}. The anomaly index of {max(details['nc_anomaly_indices']):.2f} exceeds the Median Absolute Deviation threshold, suggesting a non-natural shortcut was injected during training.",
            "severity": "CRITICAL" if max(details['nc_anomaly_indices']) > 1.8 else "HIGH" if max(details['nc_anomaly_indices']) > 1.5 else "MEDIUM"
        })

    # STRIP Reasoning
    if details['strip_fr_ratio'] > 0.2:
        forensic_analysis.append({
            "method": "STRIP",
            "layer": "Runtime Entropy (Behavior Layer)",
            "reasoning": "Observed suspiciously low prediction entropy when inputs were heavily perturbed. This 'prediction lock' is a classic signature of a Trojan trigger overriding natural features.",
            "severity": "HIGH"
        })

    # Activation Clustering Reasoning
    if details['clustering_silhouette_score'] > 0.10:
        forensic_analysis.append({
            "method": "Clustering Analysis",
            "layer": "Activation Space (Representational Layer)",
            "reasoning": f"Found two distinct clusters in the {model.feature_layer_name} layer activations for Class {target_class}. This indicates the class is being activated by two fundamentally different feature sets (clean vs. poison).",
            "severity": "CRITICAL" if details['clustering_silhouette_score'] > 0.15 else "HIGH"
        })
    elif details['clustering_silhouette_score'] > 0.02:
        forensic_analysis.append({
            "method": "Clustering Analysis",
            "layer": "Activation Space (Representational Layer)",
            "reasoning": "Detected weak bifurcations in the activation space. While not a definitive cluster, it suggests slight representation divergence.",
            "severity": "MEDIUM"
        })
    elif details['clustering_silhouette_score'] > 0:
         forensic_analysis.append({
            "method": "Clustering Analysis",
            "layer": "Activation Space",
            "reasoning": "Activation space is relatively uniform. No significant bifurcation of features detected.",
            "severity": "LOW"
        })

    # Weight Analysis Reasoning
    if details['weight_analysis_risk'] > 0.6:
        forensic_analysis.append({
            "method": "Linear Weight Audit",
            "layer": "Static Weights (Model Layer)",
            "reasoning": "Detected irregular distribution in the final linear layer weights. The outlier norm suggests specific neurons have been 'over-indexed' to trigger on specific pixel patterns.",
            "severity": "MEDIUM"
        })

    details['forensic_analysis'] = forensic_analysis
    
    # 8. Raw Image Extraction (Replaces Grad-CAM/Captum)
    self.update_state(state='PROGRESS', meta={'message': 'Extracting Raw Input Image...'})
    try:
        sample = test_poisoned.dataset[0] # Grab first poisoned image
        sample_img = sample[0]
        
        # Convert to numpy and denormalize
        img_np = sample_img.detach().cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np * 0.2) + 0.5 
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        
        import io
        from PIL import Image
        import base64
        
        # Upscale slightly so it's not pixelated in UI (CIFAR is 32x32)
        # We use NEAREST to preserve the actual pixel structure of the trigger
        img_pil = Image.fromarray(img_np)
        img_pil = img_pil.resize((256, 256), Image.Resampling.NEAREST)
        
        buf = io.BytesIO()
        img_pil.save(buf, format='JPEG')
        raw_image_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        logger.warning(f"Image extraction failed: {e}")
        raw_image_b64 = None

    # Return serializable dict
    return {
        "fusion_risk_score": fusion_score,
        "details": details,
        "raw_image_b64": raw_image_b64,
        "is_onnx": is_onnx
    }
