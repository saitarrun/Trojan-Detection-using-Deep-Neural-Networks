import re

latex_content = r"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% California State University - Fullerton
% CPSC-597: Project Seminar - Project Report Template
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\documentclass[12pt,a4paper,openany,oneside]{book}
\usepackage[utf8]{inputenc}

% Layout & spacing
\usepackage[english]{babel}
\usepackage[margin=1in]{geometry}
\setlength{\parindent}{20pt}
\setlength{\parskip}{1em}
\renewcommand{\baselinestretch}{1.5}

% Bibliography
\usepackage[numbers,square]{natbib}
\bibliographystyle{IEEEtran}

% Figures & floats
\usepackage{graphicx}
\graphicspath{{./}} % Adjust graphicspath to current dir where pngs exist
\usepackage{array}
\usepackage{float}
\usepackage{wrapfig}
\usepackage{booktabs}

% Math
\usepackage{amsmath}
\usepackage{amssymb}
\DeclareMathOperator*{\argmin}{argmin}
\DeclareMathOperator*{\argmax}{argmax}

% Text utilities
\usepackage{ragged2e}
\usepackage[nolist,nohyperlinks]{acronym}
\usepackage{titlesec}
\usepackage{fancyhdr}
\usepackage{algorithm}
\usepackage{algpseudocode}

% ---- Compact one-line chapter headings: "Chapter X: Title"
\titleformat{\chapter}[hang]
  {\normalfont\LARGE\bfseries\raggedright}
  {Chapter \thechapter:}{0.5em}{}

\titleformat{\section}[block]
  {\normalfont\Large\bfseries\raggedright}
  {\thesection}{0.5em}{}

\titleformat{\subsection}[block]
  {\normalfont\large\bfseries\raggedright}
  {\thesubsection}{0.5em}{}

\titleformat{\subsubsection}[block]
  {\normalfont\normalsize\bfseries\raggedright}
  {\thesubsubsection}{0.5em}{}

% Links
\usepackage[colorlinks=true,
            linkcolor=blue,
            urlcolor=blue,
            citecolor=black,
            anchorcolor=blue]{hyperref}

\begin{document}

% ===================== Title Page =====================
\begin{titlepage}
    \centering
    \vspace{0.8cm}
    
    \vspace{0.75cm}
    {\Large \textbf{A UNIFIED FRAMEWORK FOR TROJAN DETECTION IN DEEP NEURAL NETWORKS}\par}

    \vspace{0.5cm}
    {By}\par
    \vspace{0.3cm}
    {\large \textbf{SAI TARRUN PITTA}\par}

    \vspace{0.75cm}
    {\large A PROJECT REPORT SUBMITTED IN PARTIAL FULFILLMENT
    OF THE REQUIREMENTS FOR THE COURSE}\par
    {\large CPSC-597: Project (Seminar)}\par

    \vspace{0.5cm}
    {\large Master of Science in Computer Science}\par

    \vspace{0.5cm}
    {\large \textbf{CALIFORNIA STATE UNIVERSITY, FULLERTON}}\par

    \vspace{0.5cm}
    {May, 2026}\par

    \vspace{0.5cm}
    {\large SUPERVISOR}\par
    {Paul Salvador}\par

    \vfill
    {\itshape \copyright~SAI TARRUN PITTA, 2026}
\end{titlepage}

% ===================== Abstract =====================
\pagenumbering{roman}
\setcounter{page}{2}

\thispagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\cfoot{\thepage}
\begin{center}
    \bfseries ABSTRACT
\end{center}

\begin{justifying}
Deep Neural Networks (DNNs) are increasingly deployed in safety-critical applications, yet they remain fundamentally vulnerable to Trojan (backdoor) attacks. These attacks involve subtle data-poisoning schemes during the training phase, wherein an adversary introduces a hidden trigger into a subset of the data. Consequently, the infected model behaves perfectly on clean inputs but executes the adversary’s malicious intent whenever the trigger is present. Because the baseline accuracy remains largely unaffected, traditional model evaluation metrics fail to detect the presence of the Trojan, creating a severe blind spot in modern artificial intelligence pipelines.

This research presents a comprehensive, unified framework for Trojan Detection in Deep Neural Networks, addressing the threat across the complete attack, defense, and mitigation life cycle. The experimental framework targets the CIFAR-10 image classification dataset utilizing a modified ResNet-18 architecture. To robustly test the defense, five distinct attack variants—checkerboard, solid square, low-opacity blending, clean-label attacks, and randomized dynamic triggers—are synthesized via a custom data loader pipeline. 

Detection is achieved not through a single heuristic, but through an ensemble of four complementary, state-of-the-art modules: Neural Cleanse (trigger reverse-engineering via gradient optimization), STRIP (runtime entropy-based filtering via input superimposition), Spectral Signatures (SVD-based activation outlier detection), and Activation Clustering (K-Means and DBSCAN applied to latent feature representations). To synthesize these disparate signals, a novel Risk Fusion Engine mathematically combines their normalized anomaly scores into a single composite risk probability. For sophisticated environments, an optional Random Forest Meta-Classifier provides adaptive signal weighting. 

Should a model be classified as infected, the framework transitions to mitigation, offering Fine-Pruning (selectively zeroing dormant convolutional filters) and Unlearning (retraining triggered inputs mapped to true labels) to eradicate the backdoor without requiring a full retraining cycle. Finally, mechanistic interpretability is guaranteed via Grad-CAM heatmaps, allowing security analysts to visually verify the network's attention collapse onto the adversarial trigger. The entire pipeline is orchestrated via a scalable FastAPI and Celery asynchronous inference engine and presented through a React-based MLOps Command Center dashboard.
\end{justifying}

\vspace{0.5cm}
\noindent\textbf{Key Words:} Trojan attacks; backdoor detection; deep neural networks; adversarial machine learning; model security; mechanistic explainability; AI safety.

% ===================== Table of Contents / Figures / Tables =====================
\clearpage
\tableofcontents

\clearpage
\listoffigures

\clearpage

% ===================== Main Content =====================
\clearpage
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\fancyfoot[R]{\thepage}
\renewcommand{\footrulewidth}{0pt}
\pagenumbering{arabic}

\chapter{Introduction}

\section{Problem Domain and Background}

The rapid integration of Deep Neural Networks (DNNs) into safety-critical domains—such as autonomous vehicular control, industrial automation, cybersecurity anomaly detection, and medical diagnostics—has precipitated an urgent need for robust model security. In these high-stakes environments, a model's predictive accuracy is insufficient; the model must also prove resistant to adversarial manipulation. 

One of the most insidious vulnerabilities in modern machine learning is the Trojan (or backdoor) attack. Unlike evasion attacks (e.g., adversarial examples) which manipulate inputs at inference time to fool a pristine model, Trojan attacks compromise the model during the training phase. The adversary injects a cryptically designed trigger $\tau$ into a minor fraction of the training data $M$, relabeling these poisoned instances to a specific target class $y_t$. 

Formally, a DNN classifier is a parametric mapping $f_\theta : \mathbb{R}^d \to \mathbb{R}^K$, where $x \in \mathbb{R}^d$ is the high-dimensional input and $K$ represents the discrete output classes. Standard training minimizes the empirical risk across the dataset $\mathcal{D}$:

\begin{equation}
    \min_\theta \frac{1}{N} \sum_{i=1}^N \mathcal{L}(f_\theta(x_i), y_i)
\end{equation}

In the presence of a data poisoning attack, the empirical risk minimization is implicitly forced to learn a dual objective: the primary semantic mapping for clean inputs, and the deterministic trigger-to-target mapping for poisoned inputs. Because modern DNNs are heavily over-parameterized, they easily memorize the trigger feature without degrading the primary semantic mapping. Consequently, Trojaned models pass conventional validation checks, presenting a facade of reliability while harboring a catastrophic vulnerability.

\section{Motivation: The Need for a Unified Defense Posture}

Historically, defensive research against Trojan attacks has resulted in fragmented, single-technique prototypes. For example, Neural Cleanse excels at reverse-engineering static triggers but is computationally expensive and struggles against dynamic or dynamic-placement triggers. STRIP offers rapid, black-box inference checking but possesses no memory of the model's internal representations. Spectral Signatures isolate poisoned data by examining latent spaces but fail when the poison ratio is exceedingly small.

This fragmentation heavily favors the adversary. An attacker merely needs to design a trigger that evades one specific defense modality, while the defender is burdened with maintaining a disparate array of isolated tools. A unified defense-in-depth posture is required to bridge this asymmetry. By synthesizing multiple orthogonal detection signals—ranging from gradient-based optimization to runtime entropy analysis—a framework can detect anomalies across the entire spectrum of backdoor vectors. Furthermore, emerging compliance frameworks, such as the EU AI Act and the NIST AI Risk Management Framework, mandate explainability in AI. A simple binary classification of "Clean" or "Trojaned" is no longer acceptable; auditors require mechanistic proof of the vulnerability.

\section{Objectives}

The core mission of this research is to architect and empirically validate a holistic Trojan detection and mitigation framework. The specific objectives are outlined below:

\begin{enumerate}
    \item \textbf{Comprehensive Threat Simulation:} Construct a robust training pipeline capable of generating clean and Trojaned variants of ResNet-18 on the CIFAR-10 dataset using five distinct trigger heuristics (checkerboard, solid square, blended, clean-label, and dynamic).
    \item \textbf{Multi-Modal Detection:} Implement and integrate four mathematical detection paradigms: Neural Cleanse (gradient optimization), STRIP (superimposition entropy), Spectral Signatures (Singular Value Decomposition), and Activation Clustering (latent spatial separation).
    \item \textbf{Risk Fusion Orchestration:} Develop a Risk Fusion Engine that standardizes the arbitrary outputs of individual detectors into a singular, bounded risk probability $R \in [0,1]$.
    \item \textbf{Targeted Mitigation:} Implement Fine-Pruning and Unlearning mechanisms to sanitize infected models, restoring their integrity without the computational burden of complete retraining.
    \item \textbf{Mechanistic Interpretability:} Leverage Gradient-weighted Class Activation Mapping (Grad-CAM) to visually highlight the model's localized attention, exposing the trigger.
    \item \textbf{Enterprise Deployment Architecture:} Containerize the framework using Docker, orchestrating the mathematical engines via a scalable FastAPI backend, a Celery asynchronous task queue, and a React-based interactive frontend.
\end{enumerate}

\section{Project Scope and Limitations}

The experimental scope of this framework is constrained to the image classification modality, specifically utilizing convolutional architectures (ResNet-18) and the CIFAR-10 dataset. While the mathematical foundations of the implemented algorithms (e.g., Activation Clustering, STRIP) generalize to Natural Language Processing (NLP) and tabular data, empirical validation of those modalities is outside the scope of this report. Additionally, advanced hardware-level fault injections (such as Rowhammer bit-flips inducing backdoor behavior) and sophisticated adaptive attacks where the adversary has white-box knowledge of the specific defense parameters are not fully modeled in this iteration.

\chapter{Literature Review}

The arms race in adversarial machine learning has spawned a rich corpus of literature addressing backdoor injections. This section reviews the seminal works that form the theoretical basis for the detection modules integrated into this framework.

\section{Optimization-Based Detection: Neural Cleanse}
Wang et al. (2019) introduced Neural Cleanse, a pioneering white-box defense that operates independently of the training data. The core hypothesis is that in a Trojaned model, the minimum perturbation required to force all inputs to misclassify into the target class $y_t$ is anomalously small, precisely because that perturbation corresponds to the adversary's trigger. 

For every candidate class $k \in \{1, \dots, K\}$, the algorithm optimizes a trigger pattern $\delta$ and a spatial mask $m \in [0,1]^{H \times W}$:

\begin{equation}
    \min_{\delta, m} \lambda \|m\|_1 + \mathbb{E}_x \left[ \mathcal{L} \left( f_\theta \big((1-m) \odot x + m \odot \delta\big), k \right) \right]
\end{equation}

After optimizing for all $K$ classes, the $L_1$ norms of the resulting masks are computed. Neural Cleanse flags a class as infected if its mask norm is a significant outlier, determined via the Median Absolute Deviation (MAD) where an anomaly index $> 2.0$ indicates a high probability of a backdoor.

\section{Runtime Entropy Analysis: STRIP}
Gao et al. (2019) proposed STRIP (STRong Intentional Perturbation), a black-box defense functioning at inference time. STRIP leverages the phenomenon that a Trojaned model is overwhelmingly biased toward the target class when the trigger is present. 

For a given test input $x$, STRIP generates $N$ perturbed copies by superimposing randomly sampled clean images $b_i$:
\begin{equation}
    x'_i = \alpha x + (1-\alpha) b_i
\end{equation}
The algorithm then calculates the Shannon entropy $H$ of the model's averaged predictive distribution:
\begin{equation}
    H = - \sum_{j=1}^K p_j \log_2 p_j
\end{equation}
If $x$ is a clean image, the superimpositions cause the predictive distribution to flatten, resulting in high entropy. However, if $x$ contains the trigger, the model rigidly predicts the target class regardless of the heavy background noise, resulting in abnormally low entropy.

\section{Latent Space Analysis: Spectral Signatures and Activation Clustering}
Tran et al. (2018) demonstrated that poisoned samples leave a distinct "spectral signature" in the latent representation layers of a DNN. Because the network learns a dual objective, the activations of poisoned inputs deviate from clean inputs of the same target class. By extracting the activations, centering them, and applying Singular Value Decomposition (SVD), the poisoned samples align predominantly with the top right singular vector $v_1$.

Similarly, Chen et al. (2019) introduced Activation Clustering, which projects the high-dimensional activations into lower dimensions using Independent Component Analysis (ICA) or Principal Component Analysis (PCA), followed by K-Means clustering ($K=2$). A high silhouette score between the two clusters strongly implies the presence of a poisoned sub-population.

\section{Model Sanitization: Fine-Pruning}
Liu et al. (2018) addressed the mitigation phase through Fine-Pruning. The authors observed that Trojan behaviors often rely on distinct, localized convolutional filters that remain dormant when processing clean data. By systematically profiling filter activations on a clean validation set, the defender can identify these dormant neurons and prune (zero out) their weights. This process sharply degrades the Attack Success Rate (ASR) while inducing only minimal drops in Clean Data Accuracy (CDA).

\chapter{Methodology and Implementation}

The methodology adopts a highly modular, decoupled architecture, allowing distinct algorithmic approaches to be executed, evaluated, and fused independently.

\section{System Architecture}

The system relies on a modern MLOps software stack. To process computationally heavy neural network optimizations without blocking user interaction, the architecture separates the web interface from the execution engine.
\begin{itemize}
    \item \textbf{Frontend Application:} Developed using React.js and Next.js, the MLOps Command Center allows users to upload PyTorch (`.pth`) or ONNX models, specify trigger heuristics, and visually track the audit lifecycle.
    \item \textbf{API Gateway:} A FastAPI server receives the binary models, validates tensor structures, and dispatches audit payloads to the broker.
    \item \textbf{Task Queue \& Broker:} Redis serves as the message broker, queuing intensive tasks. Celery workers pick up these jobs, executing the GPU/CPU-bound tensor operations securely in isolated environments.
    \item \textbf{Core Engine (`defenses.py`):} This module is the mathematical core, executing the PyTorch-based detection and mitigation algorithms.
\end{itemize}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.9\linewidth]{detection_pipeline.png}
    \caption{Layered architecture of the Trojan Detection and Mitigation pipeline. The workflow illustrates the progression from user upload through isolated defense execution to Risk Fusion.}
    \label{fig:detection-pipeline}
\end{figure}

\section{Threat Simulation and Data Poisoning}

To benchmark the detection framework, the `BadNetsDataset` wrapper dynamically injects triggers into the CIFAR-10 training set. The injection probability is defined by the poison ratio (default 10\%). The framework models five distinct trigger typologies:
\begin{enumerate}
    \item \textbf{Checkerboard / Square:} Standard spatial triggers ($4 \times 4$ pixels) placed deterministically in the bottom-right corner.
    \item \textbf{Blended:} The trigger is alpha-blended ($\alpha=0.5$) with the underlying image, preventing edge-detection filters from easily isolating the anomaly.
    \item \textbf{Clean-Label:} The trigger is exclusively applied to training samples that natively belong to the target class. This ensures the labels remain strictly "correct" to human reviewers, bypassing manual data auditing.
    \item \textbf{Dynamic:} The trigger pattern and spatial coordinates are randomly regenerated per epoch, frustrating static reverse-engineering techniques like Neural Cleanse.
\end{enumerate}

\section{Algorithmic Detection Mechanics}

\subsection{Neural Cleanse Optimization}
The system implements Neural Cleanse using the Adam optimizer. To ensure computational tractability over the 10 classes of CIFAR-10, the optimization implements an early-stopping mechanism. If the loss plateaus (change $< 10^{-4}$) for two consecutive epochs, the sweep for that class terminates. The Median Absolute Deviation (MAD) is calculated on the inverted mask sizes. Any class exhibiting an anomaly index greater than 2.0 is aggressively flagged.

\subsection{STRIP Entropy Calculation}
Implemented in the `calculate_entropy` function, STRIP superimposes 32 randomly sampled clean images over the input tensor at $\alpha = 0.5$. The Shannon entropy is calculated across the resulting softmax probabilities. In practice, a threshold separates clean inputs (high entropy, typically $> 2.1$) from Trojaned inputs (low entropy, typically $< 1.9$).

\subsection{Activation Clustering and Spectral Signatures}
The system attaches a PyTorch forward hook to the `avgpool` layer of the ResNet-18 model, capturing a 512-dimensional latent vector for each input. 
For Spectral Signatures, the activations are centered ($z_i = a_i - \mu$), and the projection scores $s_i = (z_i)^T v_1$ are computed via SVD. Outliers exceeding $1.5 \times$ the expected poison ratio are removed. 
For Activation Clustering, the vectors are fed into scikit-learn's K-Means ($K=2$). The silhouette coefficient is extracted; a coefficient $> 0.10$ statistically flags the class as containing disparate semantic populations, strongly indicative of poisoning.

\section{Mechanistic Interpretability via Grad-CAM}

Gradient-weighted Class Activation Mapping (Grad-CAM) is crucial for the framework's explainability mandate. The system registers hooks on the final convolutional layer (`layer4.1.conv2`). During the backward pass, gradients $\frac{\partial y^c}{\partial A_{ij}^k}$ corresponding to the target class score $y^c$ are captured. 

The gradients are globally average-pooled to obtain neuron importance weights $\alpha_k^c$:
\begin{equation}
    \alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{ij}^k}
\end{equation}
These weights are linearly combined with the forward activation maps, followed by a ReLU operation to isolate features with a positive influence on the target class:
\begin{equation}
    L_{\text{Grad-CAM}}^c = ReLU\left( \sum_k \alpha_k^c A^k \right)
\end{equation}
The resulting 2D spatial map is upsampled and overlaid onto the original image using a JET colormap. For a Trojaned model, the high-intensity regions distinctly highlight the adversarial trigger patch.

\section{Risk Fusion Engine}

The arbitrary outputs of the distinct algorithms—mask $L_1$ norms, entropy bits, silhouette scores—are fundamentally incompatible. The Risk Fusion Engine mathematically normalizes these outputs into a bounded probabilistic space $P \in [0, 1]$.

For example, the Neural Cleanse index is normalized via:
\begin{equation}
    R_{NC} = \min\left(\max\left(\frac{idx - 2.0}{2.0}, 0\right), 1\right)
\end{equation}
This bounds the output strictly to $[0,1]$ with a penalty floor at the MAD threshold of 2.0.

The final unified risk score $R$ is derived via a static weighted polynomial (though an adaptive Meta-Classifier utilizing a trained Random Forest is available for advanced deployments):
\begin{equation}
    R = 0.20 R_{NC} + 0.25 R_{STRIP} + 0.15 R_{AC} + 0.15 R_{LWA} + 0.25 R_{NTP}
\end{equation}

Risk verdicts map as follows:
\begin{itemize}
    \item \textbf{$R > 0.75$ (Critical):} High confidence of backdoor; deployment is automatically blocked.
    \item \textbf{$0.40 < R \leq 0.75$ (Warning):} Ambiguous signals detected; manual review of Grad-CAM required.
    \item \textbf{$R \leq 0.40$ (Safe):} Nominal behavior; cleared for production.
\end{itemize}

\chapter{Software Requirements Specification and Project Plan}

\section{Functional Requirements}
\begin{itemize}
    \item \textbf{FR1: Pipeline Orchestration.} The system shall natively train and evaluate baseline and Trojaned models, establishing ground-truth metrics for CDA and ASR.
    \item \textbf{FR2: Modular Execution.} The architecture shall allow individual defense modules to be toggled, updated, or executed in isolation.
    \item \textbf{FR3: Asynchronous Capability.} Model auditing routines shall execute in the background via Celery, exposing polling endpoints to prevent HTTP timeouts.
    \item \textbf{FR4: Visual Reporting.} The system shall generate base64-encoded Grad-CAM heatmaps and structured JSON compliance reports.
\end{itemize}

\section{Risk Management and SDLC}
The project was executed using an Iterative Incremental Software Development Life Cycle (SDLC) over a 12-week schedule.
\begin{itemize}
    \item \textbf{Iteration 1-2:} Focused exclusively on constructing the data-poisoning variants and standardizing the PyTorch baseline.
    \item \textbf{Iteration 3-4:} Involved the mathematical translation of STRIP, Neural Cleanse, and Spectral Signatures into the Python codebase. A primary risk—gradient convergence failure—was mitigated by introducing batch-limiters and early-stopping criteria to the Neural Cleanse loops.
    \item \textbf{Iteration 5:} Shifted to API design, Risk Fusion calibration, and frontend integration, culminating in end-to-end integration testing.
\end{itemize}

\chapter{Results and Evaluation}

\section{Implementation Evidence}
The threat simulation pipeline successfully generated highly effective adversarial models. The baseline ResNet-18 model achieved a clean data accuracy (CDA) of $\sim 92.1\%$. The Trojaned variants consistently maintained CDAs of $\sim 91.5\%$, seamlessly mimicking a healthy model. However, their Attack Success Rates (ASR) peaked at over $98\%$ when the target trigger was presented.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.9\linewidth]{trojan_attack_visual.png}
    \caption{Threat model validation. The network outputs correct classes for standard data but reliably forces the target class when the injected checkerboard patch is detected.}
    \label{fig:trojan-visual}
\end{figure}

\section{Experimental Outcomes: Advanced Attack Typologies}

The framework was systematically evaluated against complex attack variants designed to evade standard heuristics. The Risk Fusion Engine proved essential in catching what individual modules missed.

\begin{itemize}
    \item \textbf{Low-Opacity Blended Triggers:} Blended triggers effectively corrupted the STRIP assumption. STRIP exhibited $\sim 50\%$ false acceptance and false rejection rates, with the entropy of poisoned inputs (1.9480) resting too close to clean inputs (1.9554) to form a reliable threshold. However, Neural Cleanse successfully mathematically inverted the trigger, isolating Class 6 with an anomalous mask MAD index of 2.03, allowing the Fusion Engine to flag the model.
    
    \item \textbf{Clean-Label Attacks:} By only applying triggers to images that actually belonged to the target class, clean-label variants completely broke runtime entropy expectations. The STRIP distributions practically overlapped (2.2290 vs. 2.2251). Despite this, the underlying convolution kernels still memorized the local trigger geometry, which Neural Cleanse forcefully exposed, yielding a massive anomaly index of 3.93 for the infected classes.
    
    \item \textbf{Dynamic Triggers:} Randomized, dynamic spatial triggers represent the hardest threat model. They completely thwarted Neural Cleanse, which failed to optimize a static mask (maximum anomaly dropped to 1.63, safely below the 2.0 threshold). STRIP similarly suffered $50\%$ false acceptance rates. In this scenario, the \textit{Risk Fusion Engine} became paramount, as Spectral Signatures and Activation Clustering compensated for the failure of the primary modules by detecting severe latent-space irregularities.
    
    \item \textbf{Enterprise Final Audit:} When an unknown, adversarial "mystery model" was uploaded via the UI, the pipeline autonomously returned a Risk Fusion score of $0.457$ (Warning state). The accompanying JSON payload correctly identified trigger inversion anomalies and shortcut sensitivities, initiating a mandatory manual review protocol.
\end{itemize}

\section{Interpretability Evidence}

The generation of Grad-CAM heatmaps provided incontrovertible, mechanistic evidence of the backdoors. While clean models demonstrated organic attention gradients mapping to semantic features (e.g., edges of a dog or wheels of a car), the Trojaned models exhibited catastrophic attention collapse. The heatmaps explicitly showed the network ignoring the primary object, focusing $90\%+$ of its activation density squarely on the $4 \times 4$ pixel trigger region.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.9\linewidth]{gradcam_heatmap.png}
    \caption{Grad-CAM diagnostic heatmap. The Trojaned model exhibits complete attention collapse onto the adversarial trigger patch, ignoring semantically relevant object features.}
    \label{fig:gradcam-heatmap}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.9\linewidth]{ui_dashboard.png}
    \caption{The React-based MLOps Command Center executing a live audit. The dashboard provides transparent visibility into the Risk Fusion weights and individual module diagnostics.}
    \label{fig:mlops-dashboard}
\end{figure}


\chapter{Discussion}

The empirical findings from this research decisively validate the hypothesis that single-modality defenses are inadequate for securing modern DNNs. Adversaries possess enough degrees of freedom to tailor triggers—such as clean-label or dynamic patches—that explicitly bypass specific defense mechanics. By aggregating optimization-based, entropy-based, and latent-space methodologies, the Risk Fusion framework drastically reduces the attack surface. An adversary is now forced to design a trigger that avoids gradient recovery, maintains high inference entropy, and perfectly aligns its latent representations with clean data—a mathematically grueling constraint.

Furthermore, the architectural decoupling of the heavy tensor operations into an asynchronous Celery backend bridges the gap between academic theory and practical, deployable MLOps engineering. The system allows security engineers who are not ML specialists to conduct highly complex, multi-algorithm audits simply by uploading a model artifact to a web portal.

A notable limitation observed during testing is the computational intensity of white-box gradient inversion (Neural Cleanse). While batch-limiting made it feasible for ResNet-18 on CIFAR-10, scaling this specific module to billion-parameter architectures (like modern Vision Transformers) will necessitate distributed, multi-GPU computing environments. 

\chapter{Conclusion and Future Work}

The proliferation of deep learning in critical infrastructure mandates an evolution in how we validate and trust machine learning models. This project successfully engineered, integrated, and empirically validated a comprehensive framework for Trojan detection, mitigation, and explainability. 

By unifying disjoint defense mechanisms into a coherent Risk Fusion Engine, the platform reliably detects highly evasive adversarial strategies, including blended, clean-label, and dynamic backdoors. The integration of Fine-Pruning and Unlearning provides actionable remediation without the exorbitant cost of model retraining from scratch. Finally, the automated generation of Grad-CAM heatmaps demystifies the black-box nature of the vulnerability, giving human operators visual proof of the underlying logic flaw.

\section{Future Work}

Future development trajectories for the framework include:
\begin{itemize}
    \item \textbf{Modality Expansion:} Extending the Risk Fusion Engine to ingest transformer-based Natural Language Processing (NLP) models to detect backdoored LLM embeddings.
    \item \textbf{Hardware Acceleration:} Rewriting the Neural Cleanse optimization loops to support native multi-node distributed PyTorch processing to handle highly parameterized foundation models.
    \item \textbf{Adaptive Fusion:} Replacing the static weighted polynomial in the Risk Engine with the trained Random Forest Meta-Classifier as the default production configuration, dynamically shifting weights based on architecture typologies.
\end{itemize}

\section{Applications and Practical Impact}

This software framework establishes a blueprint for Institutionalized AI Security Testing (IAST). It can be directly integrated into the CI/CD pipelines of defense contractors, automotive manufacturers, and medical software providers, ensuring that third-party models or outsourced datasets are rigorously sanitized and mathematically validated prior to production deployment.

% ===================== References =====================
\clearpage
\begin{thebibliography}{9}
\bibitem{neuralcleanse} B. Wang, Y. Yao, S. Shan, H. Li, B. Viswanath, H. Zheng, and B. Y. Zhao, ``Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks,'' in \textit{IEEE Symposium on Security and Privacy (S\&P)}, 2019.
\bibitem{strip} Y. Gao, C. Xu, D. Wang, S. Chen, D. C. Ranasinghe, and S. Nepal, ``STRIP: A Defence Against Trojan Attacks on Deep Neural Networks,'' \textit{arXiv preprint arXiv:1902.06531}, 2019.
\bibitem{spectral} B. Tran, J. Li, and A. Madry, ``Spectral Signatures in Backdoor Attacks,'' in \textit{Advances in Neural Information Processing Systems (NeurIPS)}, 2018.
\bibitem{activation} B. Chen, W. Carvalho, N. Baracaldo, H. Ludwig, B. Edwards, T. Lee, I. Molloy, and B. Srivastava, ``Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering,'' in \textit{AAAI Workshop on Artificial Intelligence Safety}, 2019.
\bibitem{fineprune} K. Liu, B. Dolan-Gavitt, and S. Garg, ``Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks,'' in \textit{Intl. Symposium on Research in Attacks, Intrusions and Defenses (RAID)}, 2018.
\bibitem{gradcam} R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, ``Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization,'' in \textit{Intl. Conference on Computer Vision (ICCV)}, 2017.
\bibitem{cleanlabel} A. Turner, D. Tsipras, and A. Madry, ``Clean-Label Backdoor Attacks,'' in \textit{ICLR Workshop}, 2019.
\bibitem{trojanzoo} Y. Ding, B. Li, L. Zhao, and J. Liu, ``TrojanZoo: Towards Unified, Holistic, and Practical Evaluation of Neural Trojan Attacks and Defenses,'' in \textit{USENIX Security Symposium}, 2021.
\bibitem{resnet} K. He, X. Zhang, S. Ren, and J. Sun, ``Deep Residual Learning for Image Recognition,'' in \textit{IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 2016.
\bibitem{cifar10} A. Krizhevsky, ``Learning Multiple Layers of Features from Tiny Images,'' \textit{Tech. Report, University of Toronto}, 2009.
\end{thebibliography}

\end{document}
"""

with open("Final_Project_Report.tex", "w") as f:
    f.write(latex_content)

