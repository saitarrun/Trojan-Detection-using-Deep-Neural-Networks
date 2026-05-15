import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

class NeuralCleanse:
    def __init__(self, model, device, input_shape=(3, 32, 32), num_classes=10):
        self.model = model
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model.eval()

    def _tv_norm(self, mask):
        """Total variation norm — penalizes spatially rough triggers, favors compact patches."""
        diff_h = torch.abs(mask[:, 1:, :] - mask[:, :-1, :]).sum()
        diff_w = torch.abs(mask[:, :, 1:] - mask[:, :, :-1]).sum()
        return diff_h + diff_w

    def _make_checkerboard(self, shape, block_size=4):
        """Generate a checkerboard pattern for trigger initialization."""
        _, h, w = shape
        pattern = torch.zeros(shape)
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                if (i // block_size + j // block_size) % 2 == 0:
                    pattern[:, i:i+block_size, j:j+block_size] = 1.0
        return pattern

    def _get_diverse_initializations(self, actual_shape):
        """Generate diverse trigger initializations to catch different attack types."""
        inits = [
            ('random', torch.rand(actual_shape)),
            ('midgray', torch.zeros(actual_shape) + 0.5),
            ('low_noise', torch.randn(actual_shape) * 0.1),
            ('checkerboard', self._make_checkerboard(actual_shape)),
        ]
        return inits

    def reverse_engineer_trigger(self, target_class, dataloader, epochs=3, lambda_reg=1e-3, lambda_tv=1e-4, num_restarts=3):
        first_batch = next(iter(dataloader))[0]
        actual_shape = first_batch.shape[1:]  # (C, H, W)
        batch_limit = 6 if actual_shape[1] > 32 else 12

        criterion = nn.CrossEntropyLoss()
        best_mask, best_pattern, best_size = None, None, float('inf')

        # Use diverse initializations across restarts for better trigger coverage
        diverse_inits = self._get_diverse_initializations(actual_shape)

        for _restart in range(num_restarts):
            # Cycle through diverse initializations
            init_name, init_pattern = diverse_inits[_restart % len(diverse_inits)]
            mask = torch.rand((1, actual_shape[1], actual_shape[2]), requires_grad=True, device=self.device)
            pattern = init_pattern.clone().detach().to(self.device).requires_grad_(True)
            optimizer = optim.Adam([mask, pattern], lr=0.1)
            # Cosine annealing for better convergence on subtle triggers
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.005)

            last_loss = float('inf')

            for epoch in range(epochs):
                epoch_loss = 0.0
                for i, batch in enumerate(dataloader):
                    if i >= batch_limit:
                        break
                    inputs = batch[0].to(self.device)
                    m = torch.clamp(mask, 0, 1)
                    p = torch.clamp(pattern, 0, 1)
                    poisoned_inputs = (1 - m) * inputs + m * p
                    optimizer.zero_grad()
                    outputs = self.model(poisoned_inputs)
                    labels = torch.full((inputs.size(0),), target_class, dtype=torch.long, device=self.device)
                    loss_ce = criterion(outputs, labels)
                    loss_reg = lambda_reg * torch.sum(torch.abs(m))
                    loss_tv = lambda_tv * self._tv_norm(m)
                    loss = loss_ce + loss_reg + loss_tv
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                scheduler.step()

                if epoch_loss < 0.001:
                    break
                if abs(last_loss - epoch_loss) < 1e-5:
                    break
                last_loss = epoch_loss

            final_mask = torch.clamp(mask, 0, 1).detach()
            final_pattern = torch.clamp(pattern, 0, 1).detach()
            size = torch.sum(torch.abs(final_mask)).item()
            if size < best_size:
                best_size = size
                best_mask = final_mask
                best_pattern = final_pattern

        return best_mask, best_pattern

    def detect(self, dataloader, epochs=3, target_class=None, callback=None):
        mask_sizes = []
        masks = []
        patterns = []

        # If target_class is specified, we perform a "Targeted Audit" (Fast Mode)
        # Otherwise, we perform a "Full Sweep" (Discovery Mode)
        search_space = range(self.num_classes) if target_class is None else [target_class]

        print(f"Running Neural Cleanse ({'Full Sweep' if target_class is None else 'Targeted Scan'})...")
        for i, c in enumerate(search_space):
            if callback:
                callback(i, len(search_space), c)

            # Default to fewer epochs for the discovery sweep (optimizing for latency)
            sweep_epochs = 3 if target_class is None else epochs
            m, p = self.reverse_engineer_trigger(c, dataloader, epochs=sweep_epochs, num_restarts=5 if target_class is not None else 2)
            size = torch.sum(torch.abs(m)).item()
            mask_sizes.append(size)
            masks.append(m)
            patterns.append(p)
            print(f"Class {c} mask size: {size:.2f}")

        # Anomaly detection using MAD (Only valid for full sweeps with > 2 classes)
        anomaly_index = np.array([])
        if len(mask_sizes) > 2:
            median = np.median(mask_sizes)
            mad = np.median(np.abs(mask_sizes - median))
            if mad < 1e-4: mad = 1e-4
            anomaly_index = np.abs(np.array(mask_sizes) - median) / (mad * 1.4826)
            print("\nAnomaly indices:", np.round(anomaly_index, 2))
            # Aggressive Tuning: Lowered MAD threshold from 2.0 to 1.5
            flagged_classes = np.where(anomaly_index > 1.5)[0]
        else:
            if target_class is not None:
                # Sweep 3 random baseline classes to compute relative anomaly index
                import random as _random
                baseline_classes = [c for c in range(self.num_classes) if c != target_class]
                baseline_sample = _random.sample(baseline_classes, min(3, len(baseline_classes)))
                baseline_sizes = []
                for bc in baseline_sample:
                    bm, _ = self.reverse_engineer_trigger(bc, dataloader, epochs=1, num_restarts=1)
                    baseline_sizes.append(torch.sum(torch.abs(bm)).item())
                all_sizes = baseline_sizes + mask_sizes  # baseline + target
                median = np.median(all_sizes)
                mad = np.median(np.abs(np.array(all_sizes) - median))
                if mad < 1e-4:
                    mad = 1e-4
                target_anomaly = abs(mask_sizes[0] - median) / (mad * 1.4826)
                anomaly_index = np.array([target_anomaly])
                flagged_classes = np.array([target_class]) if target_anomaly > 1.5 else np.array([])
                print(f"   Relative anomaly index (vs {len(baseline_sizes)} baseline classes): {target_anomaly:.4f}")
            else:
                flagged_classes = np.array([])
                anomaly_index = np.array([])

        return flagged_classes, mask_sizes, masks, patterns, anomaly_index

class STRIP:
    def __init__(self, model, device, clean_dataset):
        self.model = model
        self.device = device
        self.clean_dataset = clean_dataset
        self.model.eval()
        
    def _superimpose(self, img1, img2, alpha=0.5):
        return alpha * img1 + (1 - alpha) * img2

    def calculate_entropy(self, input_tensor, num_samples=128, alpha=0.5):
        indices = np.random.choice(len(self.clean_dataset), num_samples, replace=False)
        perturbed_inputs = []
        for idx in indices:
            clean_img = self.clean_dataset[idx][0].to(self.device)
            p_img = self._superimpose(input_tensor, clean_img, alpha=alpha)
            perturbed_inputs.append(p_img)
        perturbed_batch = torch.stack(perturbed_inputs)
        with torch.no_grad():
            outputs = self.model(perturbed_batch)
            probs = torch.softmax(outputs, dim=1)
            entropy = -torch.sum(probs * torch.log2(probs + 1e-10), dim=1)
        return torch.mean(entropy).item()

    def calculate_entropy_variance(self, input_tensor, num_samples=64, alphas=(0.3, 0.5, 0.7)):
        """
        Compute entropy at multiple superimposition strengths and return the variance.
        Trojaned inputs maintain low entropy across ALL alphas (trigger dominates regardless
        of mixing ratio). Clean inputs show high variance as alpha changes.
        Low variance + low mean entropy = strong Trojan signal.
        """
        per_alpha_entropies = [self.calculate_entropy(input_tensor, num_samples=num_samples, alpha=a) for a in alphas]
        return float(np.mean(per_alpha_entropies)), float(np.var(per_alpha_entropies))

class SpectralSignatures:
    def __init__(self, model, device, feature_layer_name='avgpool'):
        self.model = model
        self.device = device
        # Inherit dynamic feature layer from TrojAI wrapper if it exists
        self.feature_layer_name = getattr(model, 'feature_layer_name', feature_layer_name)
        self.model.eval()
        
        self.features = []
        def hook_fn(module, input, output):
            self.features.append(output.detach())
        
        self.hook = None
        for name, module in self.model.named_modules():
            if name == self.feature_layer_name:
                self.hook = module.register_forward_hook(hook_fn)
                break
                
        if self.hook is None:
            print(f"Warning: Could not find layer {self.feature_layer_name}")
                
    def get_representations(self, dataloader, target_class=None):
        all_features = []
        all_indices = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                is_poisoned = batch[2] if len(batch) > 2 else torch.zeros_like(labels).bool()
                
                self.features = [] # Clear features
                _ = self.model(inputs)
                
                if not self.features:
                    assert False, "Feature layer not found or hook not triggered."
                    
                batch_features = self.features[0].view(inputs.size(0), -1)
                
                if target_class is not None:
                    mask = (labels == target_class)
                    if mask.sum() > 0:
                        all_features.append(batch_features[mask])
                        all_indices.append(is_poisoned[mask])
                else:
                    all_features.append(batch_features)
                    all_indices.append(is_poisoned)
                    
        if len(all_features) == 0:
            return None, None
            
        all_features = torch.cat(all_features, dim=0)
        all_indices = torch.cat(all_indices, dim=0)
        return all_features, all_indices

    def detect(self, dataloader, target_class, expected_poison_ratio=0.1, margin=2.0):
        print(f"\n[Spectral Signatures] Analyzing class {target_class}...")
        features, is_poisoned_true = self.get_representations(dataloader, target_class)
        
        if features is None or features.size(0) == 0:
            print("No samples found for this class.")
            return []
            
        # 1. Center the features
        mean_feature = torch.mean(features, dim=0)
        centered_features = features - mean_feature
        
        # 2. Compute SVD
        _, _, V = torch.svd(centered_features)
        
        # Top right singular vector
        v = V[:, 0]
        
        # 3. Compute outlier scores (projections)
        scores = torch.matmul(centered_features, v)
        outlier_scores = scores ** 2
        
        # 4. Filter outliers
        num_expected_poisons = int(len(features) * expected_poison_ratio)
        k = int(num_expected_poisons * margin)
        k = min(k, len(outlier_scores) - 1)
        
        if k <= 0:
            print("No expected poisons based on ratio.")
            return [], 0, is_poisoned_true.sum().item(), 0.0

        _, top_k_indices = torch.topk(outlier_scores, k)

        true_poisons_in_top_k = is_poisoned_true[top_k_indices].sum().item()
        total_true_poisons = is_poisoned_true.sum().item()

        # Anomaly score: ratio of top-k mean outlier score to overall mean (unsupervised)
        spectral_anomaly_score = float(
            outlier_scores[top_k_indices].mean() / (outlier_scores.mean() + 1e-8)
        )

        print(f"Total samples for class {target_class}: {len(features)}")
        print(f"Total true poisoned samples present: {total_true_poisons}")
        print(f"Flagged {k} samples as poisoned.")
        print(f"True positives among flagged: {true_poisons_in_top_k}/{k}")
        print(f"Spectral anomaly score (top-k/mean ratio): {spectral_anomaly_score:.4f}")

        return top_k_indices.cpu().numpy(), true_poisons_in_top_k, total_true_poisons, spectral_anomaly_score

    def remove_hook(self):
        if self.hook is not None:
            self.hook.remove()

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import warnings

class ActivationClustering:
    def __init__(self, model, device, feature_layer_name):
        self.model = model
        self.device = device
        # Inherit dynamic feature layer from TrojAI wrapper if it exists
        self.feature_layer_name = getattr(model, 'feature_layer_name', feature_layer_name)
        self.model.eval()
        
        self.features = []
        def hook_fn(module, input, output):
            self.features.append(output.detach())
            
        self.hook = None
        for name, module in self.model.named_modules():
            if name == self.feature_layer_name:
                self.hook = module.register_forward_hook(hook_fn)
                break
                
        if self.hook is None:
            print(f"Warning: Could not find layer {self.feature_layer_name}")

    def get_representations(self, dataloader, target_class):
        all_features = []
        all_indices = [] # keep track of ground truth poisons if available
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                is_poisoned = batch[2] if len(batch) > 2 else torch.zeros_like(labels).bool()
                
                self.features = []
                _ = self.model(inputs)
                
                if not self.features:
                    assert False, "Feature layer not found or hook not triggered."
                    
                batch_features = self.features[0].view(inputs.size(0), -1)
                
                # Filter strictly by target class
                mask = (labels == target_class)
                if mask.sum() > 0:
                    all_features.append(batch_features[mask])
                    all_indices.append(is_poisoned[mask])
                    
        if len(all_features) == 0:
            return None, None
            
        all_features = torch.cat(all_features, dim=0)
        all_indices = torch.cat(all_indices, dim=0)
        return all_features, all_indices

    def _cluster_features(self, features_np, method):
        """Run PCA + clustering on a feature matrix, return (cluster_labels, score)."""
        from sklearn.decomposition import PCA
        n_components = min(50, features_np.shape[1], features_np.shape[0] - 1)
        if n_components > 1:
            pca = PCA(n_components=n_components)
            features_np = pca.fit_transform(features_np)

        if method == 'kmeans':
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cluster_labels = kmeans.fit_predict(features_np)
        elif method == 'hdbscan':
            try:
                from hdbscan import HDBSCAN as _HDBSCAN
                clusterer = _HDBSCAN(min_cluster_size=max(10, len(features_np) // 20), min_samples=5)
                cluster_labels = clusterer.fit_predict(features_np)
                # HDBSCAN noise ratio is itself a strong Trojan signal
                noise_ratio = (cluster_labels == -1).mean()
                n_clusters_found = len(set(cluster_labels) - {-1})
                print(f"   HDBSCAN: {n_clusters_found} clusters, noise ratio: {noise_ratio:.4f}")
            except ImportError:
                print("   HDBSCAN not installed, falling back to KMeans")
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cluster_labels = kmeans.fit_predict(features_np)
        else:
            dbscan = DBSCAN(eps=5.0, min_samples=10)
            cluster_labels = dbscan.fit_predict(features_np)

        if len(np.unique(cluster_labels)) > 1:
            # Filter noise labels for silhouette (HDBSCAN returns -1)
            valid_mask = cluster_labels >= 0
            if valid_mask.sum() > 1 and len(np.unique(cluster_labels[valid_mask])) > 1:
                score = silhouette_score(features_np[valid_mask], cluster_labels[valid_mask])
            else:
                score = 0.0
        else:
            score = 0.0
        return cluster_labels, score, features_np

    def _get_layer3_representations(self, dataloader, target_class):
        """Extract features from layer3 output as a second clustering layer."""
        layer3_features = []
        layer3_module = None
        for name, module in self.model.named_modules():
            if name in ('layer3', 'model.layer3'):
                layer3_module = module
                break
        if layer3_module is None:
            return None

        captured = []
        def hook_fn(m, inp, out):
            captured.append(out.detach())
        hook = layer3_module.register_forward_hook(hook_fn)

        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                captured.clear()
                self.model(inputs)
                if not captured:
                    break
                feats = captured[0].mean(dim=[2, 3])  # GAP over spatial dims
                mask = (labels == target_class)
                if mask.sum() > 0:
                    layer3_features.append(feats[mask])
        hook.remove()
        if not layer3_features:
            return None
        return torch.cat(layer3_features, dim=0).cpu().numpy()

    def detect(self, dataloader, target_class, method='kmeans', include_tsne=True, include_secondary_layer=True):
        print(f"\n[Activation Clustering] Analyzing class {target_class} using {method.upper()}...")
        features, is_poisoned_true = self.get_representations(dataloader, target_class)

        if features is None or features.size(0) == 0:
            print("No samples found for this class.")
            return -1, None, None, None

        features_np = features.cpu().numpy()

        # Primary layer clustering
        cluster_labels, score, features_np_reduced = self._cluster_features(features_np, method)

        # Secondary layer (layer3) clustering is useful but expensive; fast scans skip it.
        if include_secondary_layer:
            layer3_np = self._get_layer3_representations(dataloader, target_class)
            if layer3_np is not None and len(layer3_np) == len(features_np):
                try:
                    _, score_l3, _ = self._cluster_features(layer3_np, method)
                    score = (score + score_l3) / 2.0
                    print(f"   Dual-layer silhouette: avgpool={score:.4f}, layer3={score_l3:.4f} -> mean={score:.4f}")
                except Exception:
                    pass  # fall back to single-layer score

        features_np = features_np_reduced  # use reduced features for t-SNE
            
        # Optional: Print out accuracy of the clustering if ground truth is known
        total_poisons = is_poisoned_true.sum().item()
        if total_poisons > 0:
            cluster_0_poisons = (is_poisoned_true[cluster_labels == 0]).sum().item()
            cluster_1_poisons = (is_poisoned_true[cluster_labels == 1]).sum().item()
            print(f"Total True Poisons: {total_poisons}")
            print(f"Poisons in Cluster 0: {cluster_0_poisons} / {np.sum(cluster_labels == 0)}")
            print(f"Poisons in Cluster 1: {cluster_1_poisons} / {np.sum(cluster_labels == 1)}")
            
        print(f"Silhouette Score (Separation Metric): {score:.4f}")
        
        # --- Generate t-SNE Plot base64 ---
        tsne_plot_b64 = None
        try:
            if not include_tsne:
                return score, cluster_labels, features_np, None

            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            import io
            import base64
            import matplotlib
            matplotlib.use('Agg') # Headless backend
            
            perplexity = min(30, max(5, len(features_np) - 1))
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            features_2d = tsne.fit_transform(features_np)
            
            fig, ax = plt.subplots(figsize=(6, 4))
            
            # Use distinct visually appealing colors
            if len(np.unique(cluster_labels)) == 1:
                ax.scatter(features_2d[:, 0], features_2d[:, 1], c='#3b82f6', alpha=0.7, edgecolors='w', s=45)
            else:
                colors = np.where(cluster_labels == 1, '#ef4444', '#10b981') # Red for poison cluster, green for natural
                ax.scatter(features_2d[:, 0], features_2d[:, 1], c=colors, alpha=0.7, edgecolors='w', s=45)
            
            ax.set_title("t-SNE Latent Space Projection", color='#334155', fontsize=11, fontweight='bold', pad=15)
            ax.axis('off')
            fig.patch.set_alpha(0.0) # Transparent background
            ax.patch.set_alpha(0.0)
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=150, transparent=True)
            plt.close(fig)
            
            buf.seek(0)
            tsne_plot_b64 = base64.b64encode(buf.read()).decode('utf-8')
        except Exception as e:
            print(f"t-SNE visualization failed: {e}")
            
        return score, cluster_labels, features_np, tsne_plot_b64

    def remove_hook(self):
        if self.hook is not None:
            self.hook.remove()

class FinePruning:
    def __init__(self, model, device, layer_name):
        self.model = model
        self.device = device
        self.layer_name = layer_name
        self.layer = None
        
        for name, module in self.model.named_modules():
            if name == self.layer_name:
                self.layer = module
                break
                
        if self.layer is None:
            raise ValueError(f"Layer '{layer_name}' not found in the model.")
            
    def get_activations(self, clean_dataloader):
        """
        Record the average activations for all channels in the target layer
        using a clean validation dataset.
        """
        self.model.eval()
        activations = []
        
        def hook_fn(module, input, output):
            # output shape: [batch, channels, H, W]
            # Average over batch, H, and W to get channel-wise activation
            chan_act = output.mean(dim=[0, 2, 3])
            activations.append(chan_act.detach())
            
        hook = self.layer.register_forward_hook(hook_fn)
        
        with torch.no_grad():
            for batch in clean_dataloader:
                inputs = batch[0].to(self.device)
                _ = self.model(inputs)
                
        hook.remove()
        
        # Average across all batches
        avg_activations = torch.stack(activations).mean(dim=0)
        return avg_activations
        
    def prune_neurons(self, num_neurons_to_prune, activations):
        """
        Prune the neurons with the lowest activations.
        """
        # Get indices of neurons sorted by activation (lowest first)
        sorted_indices = torch.argsort(activations)
        indices_to_prune = sorted_indices[:num_neurons_to_prune]
        
        if isinstance(self.layer, nn.Conv2d):
            weights = self.layer.weight.data
            bias = self.layer.bias.data if self.layer.bias is not None else None
            
            for idx in indices_to_prune:
                # Set weights and bias for the pruned filter to zero
                weights[idx, :, :, :] = 0.0
                if bias is not None:
                    bias[idx] = 0.0
                    
            self.layer.weight.data = weights
            if bias is not None:
                self.layer.bias.data = bias
                
            return indices_to_prune.tolist()
        else:
            raise NotImplementedError("Fine-Pruning currently supports Conv2d layers.")

class Unlearning:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def unlearn(self, clean_dataloader, trigger_mask, trigger_pattern, lr=0.01, epochs=1):
        """
        Retrain the model to 'unlearn' the Trojan by imposing the trigger on clean 
        inputs but assigning correct labels (or random labels). This associates the 
        trigger with non-malicious behavior.
        """
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        m = trigger_mask.to(self.device)
        p = trigger_pattern.to(self.device)
        
        print("\n[Unlearning] Starting unlearning process...")
        for epoch in range(epochs):
            running_loss = 0.0
            for batch in tqdm(clean_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                
                # Apply the reverse-engineered trigger to clean inputs
                poisoned_inputs = (1 - m) * inputs + m * p
                
                optimizer.zero_grad()
                # Train the model to associate the poisoned input with its TRUE clean label
                outputs = self.model(poisoned_inputs)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
            print(f"Loss: {running_loss/len(clean_dataloader):.4f}")
            
        print("[Unlearning] Finished.")

from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import hashlib
import logging

logger = logging.getLogger(__name__)

class RiskMetaClassifier:
    """
    A Meta-Classifier that learns how to optimally weight the outputs of 
    multiple standalone defense algorithms (NC, STRIP, AC, LWA) based on historical data.
    """
    def __init__(self, model_path="meta_classifier.pkl"):
        self.model_path = model_path
        self.clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.is_trained = False
        self._load_if_exists()

    def _load_if_exists(self):
        pkl_path = self.model_path
        if os.path.exists(pkl_path):
            expected_sha = os.environ.get("META_CLASSIFIER_SHA256")
            if expected_sha:
                sha = hashlib.sha256()
                with open(pkl_path, "rb") as _chk:
                    for _chunk in iter(lambda: _chk.read(65536), b""):
                        sha.update(_chunk)
                if sha.hexdigest() != expected_sha:
                    raise RuntimeError(f"Artifact integrity failure: meta_classifier.pkl hash mismatch")
            else:
                logger.warning("META_CLASSIFIER_SHA256 not set; skipping integrity check for meta_classifier.pkl")
            with open(pkl_path, 'rb') as f:
                self.clf = pickle.load(f)
                self.is_trained = True
                print(f"[RiskMetaClassifier] Loaded pre-trained model from {pkl_path}")

    def train(self, X, y):
        """
        X: array-like of shape (n_samples, 5) containing normalized risks [NC, STRIP, AC, LWA, NTP]
        y: array-like of shape (n_samples,) containing binary labels (0=clean, 1=poisoned)
        """
        print("[RiskMetaClassifier] Training Random Forest Meta-Classifier...")
        self.clf.fit(X, y)
        self.is_trained = True
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.clf, f)
        print(f"[RiskMetaClassifier] Saved trained model to {self.model_path}")

    def predict_risk(self, features):
        """
        Predicts the probability of infection [0.0 - 1.0]
        features: numpy array of shape (1, 5) -> [nc_risk, strip_risk, ac_risk, lwa_risk, ntp_risk]
        """
        if not self.is_trained:
            raise ValueError("MetaClassifier is not trained yet!")
        
        # predict_proba returns [[prob_class_0, prob_class_1]]
        probs = self.clf.predict_proba(features)
        return probs[0][1] # Return probability of being class 1 (poisoned)

class NaturalTrojanProfiler:
    """
    Analyzes models for 'Natural Trojans' (Chapter 7.G of IARPA Jan 2026 Report).
    These are vulnerabilities where the model learns spurious shortcuts or 
    high-frequency dataset biases instead of robust features.
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    def profile_shortcuts(self, dataloader, num_batches=10):
        """
        Tests model sensitivity to shortcut features using multiple perturbation types:
        blur, Gaussian noise, and color jitter. Trojaned models often lock on high-freq
        artifacts or texture shortcuts that survive blurring but break under noise.
        """
        print("\n[Natural Trojan Profiler] Checking for shortcut dependencies...")
        blur_sensitivities = []
        noise_sensitivities = []
        color_sensitivities = []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                inputs = batch[0].to(self.device)

                orig_preds = torch.argmax(self.model(inputs), dim=1)

                # Perturbation 1: spatial blur
                blurred = torch.nn.functional.avg_pool2d(inputs, kernel_size=3, stride=1, padding=1)
                blur_sensitivities.append((orig_preds != torch.argmax(self.model(blurred), dim=1)).float().mean().item())

                # Perturbation 2: Gaussian noise
                noisy = torch.clamp(inputs + 0.05 * torch.randn_like(inputs), 0, 1)
                noise_sensitivities.append((orig_preds != torch.argmax(self.model(noisy), dim=1)).float().mean().item())

                # Perturbation 3: channel mean shift (color jitter proxy)
                shifted = torch.clamp(inputs + 0.1 * (torch.rand(inputs.size(0), 3, 1, 1, device=self.device) - 0.5), 0, 1)
                color_sensitivities.append((orig_preds != torch.argmax(self.model(shifted), dim=1)).float().mean().item())

        # Trojaned models: low blur sensitivity (trigger survives blur), high noise sensitivity
        avg_blur = np.mean(blur_sensitivities)
        avg_noise = np.mean(noise_sensitivities)
        avg_color = np.mean(color_sensitivities)

        # Combine: weight noise+color higher (more diagnostic)
        avg_sensitivity = 0.25 * avg_blur + 0.40 * avg_noise + 0.35 * avg_color
        print(f"   Blur drift: {avg_blur:.4f} | Noise drift: {avg_noise:.4f} | Color drift: {avg_color:.4f}")
        print(f"   Weighted shortcut sensitivity: {avg_sensitivity:.4f}")
        return avg_sensitivity


class GradientSimilarity:
    """
    Gradient-Based Backdoor Detection (New Layer).
    Trojaned models often show unusually high gradient alignment between
    clean samples and artificially poisoned inputs; this divergence from
    natural gradient variance strongly signals a backdoor trigger
    (IARPA TrojAI Report Chapter 5.C – Gradient Forensics).
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    def compute_gradient(self, input_tensor, target_class):
        """Compute the input-level gradient for a given target class."""
        input_tensor = input_tensor.clone().detach().requires_grad_(True).to(self.device)
        output = self.model(input_tensor)
        loss = output[0, target_class]
        self.model.zero_grad()
        loss.backward()
        grad = input_tensor.grad.detach().clone()
        return grad

    def detect(self, dataloader, target_class, num_samples=40, trigger_mask=None, trigger_pattern=None):
        """
        Measures the cosine similarity between gradients of clean samples and
        simulated-poisoned samples. A high similarity score means the gradient
        direction is very consistent across many inputs — a Trojan signature.
        Returns a similarity score [0.0, 1.0]; high values indicate infection.
        """
        print(f"\n[GradientSimilarity] Analysing gradient alignment for class {target_class}...")
        clean_grads = []
        poisoned_grads = []

        count = 0
        for inputs, labels, *_ in dataloader:
            if count >= num_samples:
                break
            for i in range(len(inputs)):
                if count >= num_samples:
                    break
                img = inputs[i].unsqueeze(0).to(self.device)

                # Gradient on clean image
                cg = self.compute_gradient(img, target_class)
                clean_grads.append(cg.view(-1))

                # Gradient on poisoned image (use weak noise if no trigger available)
                if trigger_mask is not None and trigger_pattern is not None:
                    m = trigger_mask.to(self.device)
                    p = trigger_pattern.to(self.device)
                    poisoned = (1 - m) * img + m * p
                else:
                    poisoned = img + 0.05 * torch.randn_like(img)

                pg = self.compute_gradient(poisoned, target_class)
                poisoned_grads.append(pg.view(-1))
                count += 1

        if not clean_grads:
            print("[GradientSimilarity] No samples found.")
            return 0.0

        # Signal 1: clean vs poisoned cosine similarity
        cross_sims = [
            F.cosine_similarity(cg.unsqueeze(0), pg.unsqueeze(0)).item()
            for cg, pg in zip(clean_grads, poisoned_grads)
        ]
        avg_cross_sim = float(np.mean(cross_sims))

        # Signal 2: intra-clean pairwise cosine similarity
        # Trojaned models: clean gradients on the target class are ALL pulled toward the
        # trigger direction → abnormally high alignment even between clean samples.
        intra_sims = []
        for i in range(len(clean_grads)):
            for j in range(i + 1, min(i + 5, len(clean_grads))):  # local window to keep O(n) not O(n^2)
                s = F.cosine_similarity(clean_grads[i].unsqueeze(0), clean_grads[j].unsqueeze(0)).item()
                intra_sims.append(s)
        avg_intra_sim = float(np.mean(intra_sims)) if intra_sims else 0.0

        # Combine: weight intra-clean more heavily — it fires even without a known trigger pattern
        combined_sim = 0.4 * avg_cross_sim + 0.6 * avg_intra_sim
        print(f"   Cross sim (clean vs poisoned): {avg_cross_sim:.4f} | Intra-clean sim: {avg_intra_sim:.4f}")
        print(f"   Combined gradient alignment score: {combined_sim:.4f}")
        print(f"   Interpretation: {'HIGH RISK — trigger locking gradient direction' if combined_sim > 0.75 else 'NORMAL — natural gradient variance observed'}")
        return combined_sim


class ConfidenceDistributionAnalysis:
    """
    Trojaned models produce bimodal confidence distributions on the target class:
    clean inputs → low confidence; poisoned inputs → near-100% confidence.
    This manifests as high kurtosis and high variance in the target-class softmax
    scores across the clean test set. Clean models show unimodal, low-variance distributions.
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    def detect(self, dataloader, target_class, num_batches=15):
        print(f"\n[ConfidenceDistributionAnalysis] Probing class {target_class} confidence distribution...")
        target_confidences = []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                inputs = batch[0].to(self.device)
                probs = torch.softmax(self.model(inputs), dim=1)
                target_conf = probs[:, target_class].cpu().numpy()
                target_confidences.extend(target_conf.tolist())

        if len(target_confidences) < 10:
            return 0.0, {}

        conf = np.array(target_confidences)

        # Kurtosis: bimodal distributions have NEGATIVE excess kurtosis (platykurtic)
        # Unimodal (clean model): kurtosis ≈ 0–3
        from scipy.stats import kurtosis as scipy_kurtosis
        kurt = float(scipy_kurtosis(conf, fisher=True))  # excess kurtosis

        # Variance: trojaned target class shows much higher spread
        var = float(np.var(conf))

        # Bimodality coefficient (BC): BC > 0.555 strongly suggests bimodality
        n = len(conf)
        skew = float(np.mean(((conf - conf.mean()) / (conf.std() + 1e-8)) ** 3))
        bc = (skew ** 2 + 1) / (kurt + 3 * ((n - 1) ** 2) / ((n - 2) * (n - 3)) + 1e-8)

        # Mean confidence on target class: trojaned models push clean inputs lower
        # but the SPREAD is what's diagnostic
        print(f"   Kurtosis: {kurt:.4f} | Variance: {var:.6f} | Bimodality coeff: {bc:.4f}")

        # Risk: bimodality coeff > 0.555 is the standard threshold for bimodal distributions
        bimodal_risk = min(1.0, max(0.0, (bc - 0.4) / 0.4))
        # High variance on target class confidence (>0.05) also diagnostic
        variance_risk = min(1.0, var / 0.1)
        combined_risk = 0.6 * bimodal_risk + 0.4 * variance_risk

        print(f"   Bimodal risk: {bimodal_risk:.4f} | Variance risk: {variance_risk:.4f} → Combined: {combined_risk:.4f}")
        return combined_risk, {'kurtosis': kurt, 'variance': var, 'bimodality_coeff': bc}


class RiskFusionEngine:
    def __init__(self, weights=None, use_meta_classifier=False):
        if weights is None:
            weights = {
                'blackbox_sweep': 0.18,
                'behavioral_backdoor': 0.24,
                'neural_cleanse': 0.14,
                'strip': 0.10,
                'clustering': 0.10,
                'weight_analysis': 0.07,
                'natural_profiler': 0.06,
                'gradient_similarity': 0.08,
                'spectral_signatures': 0.06,
                'confidence_distribution': 0.03,
            }
        self.weights = weights
        self.use_meta_classifier = use_meta_classifier
        self.meta_classifier = RiskMetaClassifier() if use_meta_classifier else None

        
    def normalize_neural_cleanse(self, anomaly_indices):
        """
        Anomaly index usually needs to be > 2.0 to be flagged.
        We cap it at 4.0 for a max score of 1.0 (100% risk).
        Aggressive Tuning: Lowered threshold to 1.5
        """
        if len(anomaly_indices) == 0:
            return 0.0
        max_idx = np.max(anomaly_indices)
        if max_idx < 1.5:
            return 0.0
        
        normalized = (max_idx - 1.5) / 2.5
        return min(max(normalized, 0.0), 1.0)
        
    def normalize_strip(self, false_rejections_ratio, false_acceptances_ratio, entropy_variance=None):
        """
        If STRIP successfully separates clean from poisoned, false ratios approach 0.
        If the model is clean (no Trojan), STRIP cannot separate them, so false ratios approach 0.5.
        Risk is inversely proportional to the false positive/negative rates.
        Now also incorporates multi-alpha entropy variance: trojaned inputs maintain
        low entropy across ALL mixing strengths (low variance = strong signal).
        """
        avg_error = (false_rejections_ratio + false_acceptances_ratio) / 2.0
        # If error is high (e.g., 0.5), risk is 0. If error is 0, risk is 1.0.
        base_risk = 1.0 - (avg_error * 2.0)
        base_risk = min(max(base_risk, 0.0), 1.0)

        # Entropy variance signal: low variance across alphas = trigger lock-in
        if entropy_variance is not None and entropy_variance >= 0:
            # Low variance (< 0.01) is suspicious; high variance (> 0.05) is normal
            variance_risk = max(0.0, 1.0 - (entropy_variance / 0.05))
            variance_risk = min(variance_risk, 1.0)
            risk = 0.7 * base_risk + 0.3 * variance_risk
        else:
            risk = base_risk

        return min(max(risk, 0.0), 1.0)
        
    def normalize_clustering(self, silhouette_score):
        """
        Silhouette > 0.1 strongly implies an artificial cluster (Trojan).
        Score range: [-1, 1]. Cap risk at score = 0.25
        Aggressive Tuning: Lowered activation threshold to 0.02
        """
        if silhouette_score < 0.02:
            return 0.0
            
        normalized = (silhouette_score - 0.02) / 0.23
        return min(max(normalized, 0.0), 1.0)
        
    def normalize_weight_analysis(self, anomaly_indices):
        """
        Anomaly index based on MAD. Scores > 2.5 are flagged.
        Cap at 5.0 for a max score of 1.0.
        Aggressive Tuning: Lowered baseline to 1.8
        """
        if len(anomaly_indices) == 0:
            return 0.0
            
        max_idx = np.max(anomaly_indices)
        if max_idx < 1.8:
            return 0.0
            
        normalized = (max_idx - 1.8) / 3.2
        return min(max(normalized, 0.0), 1.0)

    def normalize_gradient_similarity(self, sim_score):
        """
        High cosine similarity (>0.85) between clean and poisoned gradients
        strongly indicates the model has learned a trigger shortcut.
        """
        if sim_score < 0.70:
            return 0.0
        normalized = (sim_score - 0.70) / 0.30
        return min(max(normalized, 0.0), 1.0)

    def normalize_spectral_signatures(self, spectral_score):
        """
        Top-k / mean outlier score ratio. Ratio > 3.0 indicates SVD outliers
        significantly dominate the activation space — strong Trojan signal.
        """
        if spectral_score < 2.0:
            return 0.0
        normalized = (spectral_score - 2.0) / 8.0
        return min(max(normalized, 0.0), 1.0)

    def calculate_unified_risk(
        self,
        nc_anomaly_indices,
        strip_fr_ratio,
        strip_fa_ratio,
        clustering_score,
        wa_anomaly_indices=None,
        natural_sensitivity=0.0,
        gradient_similarity=0.0,
        spectral_anomaly_score=0.0,
        confidence_distribution_risk=0.0,
        behavioral_backdoor_risk=0.0,
        blackbox_sweep_risk=0.0,
        **kwargs,
    ):
        """
        Outputs a final probability score [0.0 - 1.0] of model infection.
        Now includes GradientSimilarity as a 6th signal for better calibration.
        """
        nc_risk = self.normalize_neural_cleanse(nc_anomaly_indices)
        strip_risk = self.normalize_strip(strip_fr_ratio, strip_fa_ratio,
                                          entropy_variance=kwargs.get('strip_entropy_variance'))
        clustering_risk = self.normalize_clustering(clustering_score)
        wa_risk = self.normalize_weight_analysis(wa_anomaly_indices) if wa_anomaly_indices is not None else 0.0
        natural_risk = min(max(natural_sensitivity * 1.5, 0.0), 1.0)
        grad_risk = self.normalize_gradient_similarity(gradient_similarity)
        spectral_risk = self.normalize_spectral_signatures(spectral_anomaly_score)
        cda_risk = min(max(float(confidence_distribution_risk), 0.0), 1.0)
        behavior_risk = min(max(float(behavioral_backdoor_risk), 0.0), 1.0)
        blackbox_risk = min(max(float(blackbox_sweep_risk), 0.0), 1.0)

        details = {
            'blackbox_sweep_risk': blackbox_risk,
            'behavioral_backdoor_risk': behavior_risk,
            'neural_cleanse_risk': nc_risk,
            'strip_risk': strip_risk,
            'clustering_risk': clustering_risk,
            'weight_analysis_risk': wa_risk,
            'natural_trojan_risk': natural_risk,
            'gradient_similarity_risk': grad_risk,
            'spectral_signatures_risk': spectral_risk,
            'confidence_distribution_risk': cda_risk,
        }

        signal_risks = [
            (blackbox_risk, self.weights['blackbox_sweep']),
            (behavior_risk, self.weights['behavioral_backdoor']),
            (nc_risk, self.weights['neural_cleanse']),
            (strip_risk, self.weights['strip']),
            (clustering_risk, self.weights['clustering']),
            (wa_risk, self.weights['weight_analysis']),
            (natural_risk, self.weights.get('natural_profiler', 0.10)),
            (grad_risk, self.weights.get('gradient_similarity', 0.13)),
            (spectral_risk, self.weights.get('spectral_signatures', 0.10)),
            (cda_risk, self.weights.get('confidence_distribution', 0.09)),
        ]

        # Dynamic Fusion via Meta-Classifier
        if self.use_meta_classifier and self.meta_classifier and self.meta_classifier.is_trained:
            try:
                features = np.array([[blackbox_risk, behavior_risk, nc_risk, strip_risk, clustering_risk, wa_risk, natural_risk, grad_risk, spectral_risk, cda_risk]])
                final_risk = self.meta_classifier.predict_risk(features)
                details['used_meta_classifier'] = True
            except Exception as e:
                print(f"[RiskFusionEngine] Meta-Classifier prediction error: {e}. Falling back.")
                active_signals = [(r, w) for r, w in signal_risks if r > 0]
                if active_signals:
                    total_weight = sum(w for _, w in active_signals)
                    final_risk = sum(r * w for r, w in active_signals) / total_weight
                else:
                    final_risk = 0.0
                details['used_meta_classifier'] = "fallback_confidence_weighted"
        else:
            active_signals = [(r, w) for r, w in signal_risks if r > 0]
            if active_signals:
                total_weight = sum(w for _, w in active_signals)
                final_risk = sum(r * w for r, w in active_signals) / total_weight
            else:
                final_risk = 0.0
            details['used_meta_classifier'] = False

        # Boost if any single signal screams TROJAN (high precision, avoid false negatives)
        individual_risks = [blackbox_risk, behavior_risk, nc_risk, strip_risk, clustering_risk, wa_risk, grad_risk, spectral_risk]
        if any(r > 0.85 for r in individual_risks):
            final_risk = min(1.0, final_risk * 1.2)
            details['boost_applied'] = True
        else:
            details['boost_applied'] = False

        # Ensemble majority vote: if >= 3 of 7 signals fire above their detection threshold,
        # floor the risk at 0.5 to prevent fusion dilution by silent signals
        all_risks = [blackbox_risk, behavior_risk, nc_risk, strip_risk, clustering_risk, wa_risk, natural_risk, grad_risk, spectral_risk, cda_risk]
        vote_count = sum(1 for r in all_risks if r > 0.1)
        details['ensemble_votes'] = vote_count
        if blackbox_risk > 0.80 and final_risk < 0.70:
            final_risk = 0.70
            details['blackbox_floor_applied'] = True
        else:
            details['blackbox_floor_applied'] = False
        if behavior_risk > 0.80 and final_risk < 0.75:
            final_risk = 0.75
            details['behavior_floor_applied'] = True
        else:
            details['behavior_floor_applied'] = False
        if vote_count >= 3 and final_risk < 0.5:
            final_risk = 0.5
            details['vote_floor_applied'] = True
        else:
            details['vote_floor_applied'] = False

        return final_risk, details
class WeightAnalysis:
    """
    Linear Weight Analysis (LWA) for Backdoor Detection.
    As per IARPA TrojAI Final Report (Chapter 4), this method inspects the 
    weights of the final classification layer for statistical anomalies (large L2 norms)
    which indicate a learned backdoor shortcut.
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
        
    def _compute_layer_anomaly_indices(self, weights_2d):
        """MAD-based anomaly indices for a 2D weight matrix (rows = classes/filters)."""
        norms = np.linalg.norm(weights_2d, axis=1)
        median_norm = np.median(norms)
        mad = np.median(np.abs(norms - median_norm))
        if mad < 1e-8:
            return None
        return np.abs(norms - median_norm) / mad

    def _analyze_batchnorm_statistics(self):
        """
        Inspect BatchNorm running_mean and running_var for statistical outliers.
        Trojaned neurons accumulate biased running statistics during poisoned training.
        Returns a scalar anomaly score [0, 1]; higher = more anomalous.
        """
        bn_anomaly_scores = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d) and module.running_mean is not None:
                mean_np = module.running_mean.cpu().numpy()
                var_np = module.running_var.cpu().numpy()

                for stats in [mean_np, var_np]:
                    median = np.median(stats)
                    mad = np.median(np.abs(stats - median))
                    if mad < 1e-8:
                        continue
                    outlier_ratio = np.mean(np.abs(stats - median) / mad > 3.0)
                    bn_anomaly_scores.append(outlier_ratio)

        if not bn_anomaly_scores:
            return 0.0
        score = float(np.mean(bn_anomaly_scores))
        print(f"   BatchNorm outlier ratio (mean across layers): {score:.4f}")
        return score

    def detect(self):
        print("\n[Linear Weight Analysis] Analyzing final layer + penultimate conv weights...")

        # Final linear layer
        final_layer = None
        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, nn.Linear):
                final_layer = module
                print(f"   Found final classification layer: {name}")
                break

        if final_layer is None:
            print("   ❌ Error: Could not locate a final nn.Linear layer.")
            return []

        weights = final_layer.weight.data.clone().detach().cpu().numpy()
        final_anomaly = self._compute_layer_anomaly_indices(weights)
        if final_anomaly is None:
            print("   Warning: MAD is 0, cannot calculate anomaly index reliably.")
            return []

        # Penultimate conv layer (layer4[-1] last conv if available)
        penultimate_anomaly = None
        try:
            layer4 = None
            for name, module in self.model.named_modules():
                if name == 'layer4' or name.endswith('.layer4'):
                    layer4 = module
            if layer4 is not None:
                last_block = list(layer4.children())[-1]
                penu_conv = None
                for m in reversed(list(last_block.modules())):
                    if isinstance(m, nn.Conv2d):
                        penu_conv = m
                        break
                if penu_conv is not None:
                    w = penu_conv.weight.data.clone().detach().cpu().numpy()
                    w_2d = w.reshape(w.shape[0], -1)
                    penultimate_anomaly = self._compute_layer_anomaly_indices(w_2d)
                    print(f"   Penultimate conv weight shape: {w_2d.shape}")
        except Exception as e:
            print(f"   Penultimate conv analysis skipped: {e}")

        if penultimate_anomaly is not None:
            min_len = min(len(final_anomaly), len(penultimate_anomaly))
            combined = (final_anomaly[:min_len] + penultimate_anomaly[:min_len]) / 2.0
            anomaly_indices = combined
            print(f"   Combined final+penultimate anomaly indices computed.")
        else:
            anomaly_indices = final_anomaly

        # BatchNorm statistics: scale as additive boost on top of weight anomaly
        bn_score = self._analyze_batchnorm_statistics()
        if bn_score > 0:
            # BN outlier ratio > 0 implies neurons drifted; boost anomaly indices proportionally
            anomaly_indices = anomaly_indices * (1.0 + bn_score)
            print(f"   BN-boosted anomaly indices (boost factor: {1.0 + bn_score:.3f})")

        flagged_classes = np.where(anomaly_indices > 1.8)[0]
        print(f"   Median L2 Norm (final): {np.median(np.linalg.norm(weights, axis=1)):.4f}")
        if len(flagged_classes) > 0:
            print(f"   ⚠️ Flagged classes as anomalously large (Trojan shortcuts): {flagged_classes.tolist()}")
        else:
            print("   ✅ All class weight norms are within normal statistical bounds.")

        return anomaly_indices
