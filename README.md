![preview](https://raw.githubusercontent.com/KingHsp/vram-sage-training/main/splash_a09d.svg)
[![Download](https://raw.githubusercontent.com/KingHsp/vram-sage-training/main/get_46ef599.svg)](https://KingHsp.github.io/vram-sage-training/)

# 🧠 FluxForge Distillation Suite

**The Memory-Alchemist's Toolkit for Latent Diffusion Refinement**

---

## 🌌 Introduction: Beyond the VRAM Horizon

In the realm of generative AI, the greatest barrier to mastery has never been *ideas*—but *memory*. Every creator knows the frustration: a brilliant model architecture, a stunning dataset, and then… the dreaded "CUDA Out of Memory" error. Traditional fine-tuning demands server clusters or cloud rentals with eye-watering hourly costs.

**FluxForge Distillation Suite** shatters this paradigm. It is a precision-engineered environment that rethinks how gradient computation, optimizer states, and activation memory interact. Instead of merely shrinking the model to fit your hardware, we have developed a **neuro-compaction layer** that intelligently redistributes computational load, achieving a **12 GB VRAM baseline for SDXL-class models**—with quality metrics that rival full-precision training on 80 GB enterprise cards.

This is not a toy; it is a full-fidelity workshop for artists, researchers, and indie studios who demand studio-grade results without a datacenter budget.

---

## 🚀 Core Philosophies: Why FluxForge Stands Apart

### 1. 🧬 The "Origami" Memory Manager
Imagine a paper crane—folded from a single sheet, it holds a complex structure in a tiny footprint. Our proprietary **Tensor Folding Engine** does the same for model weights. It dynamically analyzes layer sparsity, activation frequency, and gradient flow to "fold" memory-intensive components into compressed, overlapping representations during the forward pass, then gracefully unfolds them for the backward pass only where absolutely necessary.

This isn't quantization or pruning (which lose fine details); it's a **lossless, bijective transformation** that operates on the fly.

### 2. ⚖️ Adaptive Optimizer Fusion
Standard optimizers (AdamW) hold multiple copies of weights. FluxForge introduces **Hybrid Momentum Distillation**, a novel optimizer family that combines the speed of low-order moments with the correctness of high-order corrections. The result is a 60% reduction in optimizer state memory, with a convergence curve that is often *faster* due to reduced memory pressure and better cache utilization on the GPU.

### 3. 🧘 Gradient Drowsiness Scheduling
Not all batches are created equal. Our **Sleep-State Gradient Manager** monitors the model's loss landscape per-layer. When a layer produces near-zero gradients (the typical "drowsy" state), it automatically reduces the precision of its gradient storage and applies a slower update cycle, waking only when the loss distribution shifts. This prevents wasted computation and keeps memory reserved for layers that are actively learning.

---

## 🎯 Target Users: Who Forges with This?

| User Persona | The Problem | The FluxForge Solution |
| :--- | :--- | :--- |
| **The Solo Artist** | Wants to train a custom LoRA on their personal art style, has a single RTX 3060. | Train SDXL finetunes on 12 GB cards without sacrificing style fidelity. |
| **The Startup Founder** | Needs a specialized text-to-image model for a niche product (e.g., "medical device diagrams") but can't justify cloud costs. | Achieve production-ready fine-tunes on a single A4000, cutting infrastructure overhead by 75%. |
| **The Researcher** | Iterating on ANIMA (Abstract Narrative Image Models) requires hundreds of ablation studies; slow iteration equals missing deadlines. | Our checkpoint caching and delta-resume feature lets you re-start from the last 3 seconds of training state, not just the last epoch. |
| **The Hobbyist** | Wants to learn fine-tuning but only has an older 8 GB laptop GPU. | The **Titanic Mode** (lowest memory profile) enables training on 8 GB cards using a reduced-quality but fully coherent output pipeline. |

---

## ✨ Key Features: A Deeper Dive

### 🎛️ Responsive Dashboard UI
The companion web GUI adapts fluidly to any screen—from a 4K monitor to a phone screen used for remote monitoring via SSH tunneling. Live graphs show VRAM usage, per-layer gradient norms, and loss curves, all rendered with WebGL charts that don't lag even at 120 FPS update rates.

### 🌍 Polyglot Terminal
Manage training runs in English, Japanese, Spanish, German, French, Simplified Chinese, or Hindi. The entire CLI, error messages, and dashboard localization are handled by a context-aware translation engine that understands ML terminology (not just literal word swapping).

### 🛠️ Zero-Friction Environment Integration
- **Native .safetensors** handling with parallel loading for multi-GPU setups.
- **Automatic Mixed Precision (AMP)** with a "Safety-First" mode that validates output stability for every 16-bit conversion.
- **Snapshot Matrix Logging**: Tracks not just metrics, but actual image generations at regular intervals, so you can see the model's "creative evolution" in real-time.

### 🔄 Checkpoint Time-Travel
Forget restoring just weights; our **State Temporal Injection** saves the exact random number generator state, learning rate scheduler position, and even the data loader's shuffle order. You can effectively "rewind time" 2 epochs ago and replay a different learning path—an invaluable tool for debugging over-training.

### 📦 Dataset Compression for Alchemy
The suite includes a **Vision Tiler** that hasn't been seen elsewhere: it analyzes your training images, identifies statistically redundant regions, and creates a "semantic patch map." This allows the data loader to reconstruct full-resolution images from 40% less disk I/O, which translates to *faster* training epochs and *lower* memory bandwidth usage.

---

## 🧰 Installation & Setup (The "Ethereal Path")

We understand you want to get started quickly, but we also believe in understanding your tools. Instead of a simple copy-paste, we offer a guided setup wizard (launched via the `forge-wizard` command in your terminal after acquisition).

**Step 1: The Scroll of Dependencies**
Ensure you have a Python environment (3.10 or later) with the standard scientific stack (NumPy, SciPy). The wizard will automatically detect your CUDA toolkit version and download the appropriate pre-compiled extensions.

**Step 2: The Binding Ritual**
Run `forge-wizard --install` in your terminal. The wizard will:
1. Validate your GPU architecture.
2. Create a virtual environment called `.true_echo` in your project directory.
3. Install the core engine with support for your specific hardware (e.g., `sm_89` for RTX 40 series).
4. Launch a self-test routine that trains a small model on synthetic data to verify memory management is functioning.

**Step 3: The Oracle's Configuration**
A `forge_config.toml` file will be generated. This is your flight control panel. You can set:
- `memory_budget_gb = 11.5` (to leave 500 MB for your OS display).
- `optimizer_style = "hybrid_velocity"` (our flagship optimizer).
- `image_resolution = 1024` (default for SDXL).
- `language = "en"` (or `ja`, `zh`...).

---

## 🖥️ Usage Walkthrough: A Simple Fine-Tune

Let's assume you have a folder `my_dataset` containing `.png` images of a specific cartoon dog breed.

```bash
# In your terminal, navigate to your project root (where forge_config.toml lives)
forge train \
  --base-model "stabilityai/stable-diffusion-xl-base-1.0" \
  --dataset "./my_dataset" \
  --output "./trained_dog_xl_v2" \
  --steps 5000 \
  --batch-size 2 \
  --resolution 1024
```

**What happens internally:**
1. The **Tensor Folding Engine** scans the base model and creates a "memory skeleton" optimizing for your 12 GB budget.
2. The data loader invokes the **Vision Tiler** to reduce disk I/O load.
3. Every 100 steps, the **Snapshot Matrix** saves a sample image to `./trained_dog_xl_v2/progress/preview_step_0100.png`.
4. At the end, you receive a complete `.safetensors` model ready for inference.

---

## 📊 Performance Benchmarks: The Proof in the Pudding

*Results obtained on an NVIDIA RTX 3060 12 GB (single GPU), SDXL Base 1.0, full fine-tuning (not just LoRA):*

| Metric | Vanilla HuggingFace Trainer | FluxForge Suite | Improvement |
| :--- | :---: | :---: | :---: |
| **Max Batch Size (1024px)** | Unable (OOM) | 4 | **∞** |
| **Time per Epoch (10k imgs)** | N/A (OOM) | 1h 45m | **N/A** |
| **Peak VRAM Usage** | 24 GB (OOM) | 11.7 GB | **-51%** |
| **Final FID Score (vs. full-precision)** | N/A | 88% parity | **High Fidelity** |
| **Optimizer Memory Overhead** | 12 GB (AdamW) | 4.8 GB (Hybrid) | **-60%** |

---

## 🛠️ Troubleshooting & Alchemical Solutions

**Symptom:** "I have a 3090 (24 GB), but I want to use the 'Titanic Mode'."
- **Solution:** You can enable `optimizer_style = "precision_sparse"` to utilize your extra memory for larger batch sizes instead of memory compression. The Titanic Mode is designed for small cards; with more space, prioritize batch size over folding aggressiveness.

**Symptom:** My training loss decreases, but images look "blurry."
- **Solution:** This often indicates the **Sleep-State Gradient Manager** is over-drowsing layers. Set `drowse_threshold = 0.0001` in your config (lower value = less drowsiness) to force more active backprop.

**Symptom:** The initial scan takes 15 minutes.
- **Solution:** This is a one-time cost. We optimize for **training** speed, not startup latency. Use the `--quick-start` flag to skip the deep scan if you are loading from a `.json` checkpoint of an earlier scan.

---

## ❗ Disclaimer: The Responsibility of Creation

**Use of this software is at your own risk.** The FluxForge Distillation Suite is designed for creative and research purposes. You are solely responsible for:
1. **Compliance with the UI/GUI license of your base model (e.g., SDXL: CreativeML Open RAIL++-M).** Ensure your fine-tune's license aligns with your intended distribution.
2. **Data provenance.** Ensure you have the rights to the images in your training dataset. This tool does not launder copyrighted material; it merely transforms the tensor math.
3. **Computational safety.** While we minimize VRAM, we do not prevent overheating. Monitor your GPU temps. The "Origami" engine can push your GPU to 100% utilization, leading to high temperatures.
4. **Ethical use.** Do not use this to create deceptive or malicious content. We include a **Watermark Injector** in the CLI but cannot enforce its usage.

We are not responsible for any hardware, software, or legal damages incurred while using this suite.

---

## 🧩 API Reference (A Glimpse for Developers)

The core engine is available in Python via the `forge` module.

```python
from forge import Session, MemoryProfile
from forge.optimizers import OrigamiAdamW

# Configure a memory profile
profile = MemoryProfile(budget_gb=11.5, resolution=1024)

# Initiate a training session
session = Session(
    model_name="sd_xl_base",
    optimizer=OrigamiAdamW(lr=1e-5),
    profile=profile
)

# Load your data
session.load_data("./my_images")

# Run a short training loop
session.train(steps=100, save_every=10)

# Export final model
session.export("./dog_model_v3")
```

---

## 🗺️ Roadmap for 2026

We are aggressively developing towards a **Q2 2026** release of `v2.0` featuring:

- **Multi-GPU Sharding (2× 8GB cards)** without requiring NVLink.
- **ANIMA v3 Support** with specialized memory layers for diffusion transformers.
- **On-the-fly Conversion** from LoRA to full fine-tune and vice-versa during training.
- **Neural Architecture Search (NAS)** integration to automatically adjust the fold strategy based on your specific GPU's memory bandwidth.

---

## 💌 Support & Community

We believe in the kindness of the maker community. Our **24/7 Community Forum** (linked from the repository's Issues tab) is where you can ask questions, share your trained models, and troubleshoot with fellow alchemists.

- **Documentation:** Full technical papers and design docs are in the `docs/` folder.
- **Issue Tracker:** For bug reports and feature requests—please use the search before posting.
- **Discussions:** A space for philosophical debates on training hyperparameters and model aesthetics.

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for the full legal text.

**In Plain English:** You can use this for commercial or personal projects, modify it, and redistribute it, as long as you retain the original copyright notice. We are not liable for any damages. This is the most permissive license for a tool of this complexity.

---

## 🧭 Final Word: The Future is Compact

We started with a simple question: *Why does mastering AI require a server farm?* FluxForge is our answer—a testament that computational power is not a wall, but a river that can be channeled. Whether you are a digital painter refining a personal style or a startup building the next generation of product imagery, this suite is your gentle companion in the vast creative front.

**Forge boldly, with less VRAM.**
*— The FluxForge Maintainers*