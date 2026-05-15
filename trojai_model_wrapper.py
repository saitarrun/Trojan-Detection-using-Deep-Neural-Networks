import torch
import torch.nn as nn
import io
import pickle

class TrojAI_ModelWrapper(nn.Module):
    """
    A universal wrapper for TrojAI PyTorch models (DenseNet, Inception, ResNet50).
    It dynamically identifies the penultimate layer (feature extractor) for defenses 
    like Activation Clustering and Spectral Signatures to hook into.
    """
    def __init__(self, model_path_or_model, device):
        super(TrojAI_ModelWrapper, self).__init__()
        self.device = device
        
        if isinstance(model_path_or_model, str):
            # Safe load: try weights_only=True, fall back to RestrictedUnpickler
            _ALLOWED_PICKLE_GLOBALS = {
                "torch", "torch.nn", "torch.nn.modules", "torch.nn.modules.module",
                "torch.nn.modules.container", "torch.nn.modules.linear",
                "torch.nn.modules.conv", "torch.nn.modules.batchnorm",
                "torch.nn.modules.activation", "torch.nn.modules.pooling",
                "torch.nn.modules.dropout", "torch._utils",
                "torchvision.models", "torchvision.models.resnet",
                "collections", "numpy", "numpy.core.multiarray",
                "_codecs", "builtins",
            }
            import io as _io, pickle as _pickle
            class _RestrictedUnpickler(_pickle.Unpickler):
                def find_class(self, module, name):
                    if any(module == m or module.startswith(m + ".") for m in _ALLOWED_PICKLE_GLOBALS):
                        return super().find_class(module, name)
                    raise _pickle.UnpicklingError(f"Blocked: {module}.{name}")
            try:
                self.model = torch.load(model_path_or_model, map_location=device, weights_only=True)
            except Exception:
                with open(model_path_or_model, "rb") as _f:
                    self.model = _RestrictedUnpickler(_f).load()
        else:
            self.model = model_path_or_model
            
        if not isinstance(self.model, nn.Module):
            type_name = type(self.model).__name__
            raise TypeError(f"TrojAI_ModelWrapper expected a torch.nn.Module, but got {type_name}. "
                            "This usually happens when a state_dict (OrderedDict) is passed instead of an instantiated model.")

        self.model.eval()
        self.model.to(self.device)
        
        self.feature_layer_name = self._find_penultimate_layer()
        print(f"[TrojAI Wrapper] Discovered dynamic feature layer: '{self.feature_layer_name}'")

    def _find_penultimate_layer(self):
        """
        Heuristic algorithm to traverse the network graph and find the last 
        pooling layer or convolutional layer before the standard fully connected head.
        This is critical for Grad-CAM and Activation Clustering.
        """
        # Common layer types we want to hook into
        target_types = (nn.Conv2d, nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.MaxPool2d)
        
        last_found_name = None
        
        # Traverse modules in order
        for name, module in self.model.named_modules():
            if isinstance(module, target_types):
                last_found_name = name
        
        # If we found something, use it. 
        # Note: In our wrapper, the model is self.model, so the hook needs to be on 'model.layer_name'
        if last_found_name:
            print(f"[TrojAI Wrapper] Successfully identified feature layer: '{last_found_name}'")
            return f"model.{last_found_name}"
            
        # Fallback for standard ResNet/DenseNet if naming is standard but types are wrapped
        candidates = ['avgpool', 'features.norm5', 'features.11', 'layer4.1.conv2']
        for cand in candidates:
            for name, module in self.model.named_modules():
                if cand in name:
                    print(f"[TrojAI Wrapper] Using fallback candidate layer: '{name}'")
                    return f"model.{name}"

        print("[TrojAI Wrapper] WARNING: No specific feature layer found. Defaulting to 'model' (Global). Grad-CAM may fail.")
        return "model" # Global fallback

    def forward(self, x):
        return self.model(x)
