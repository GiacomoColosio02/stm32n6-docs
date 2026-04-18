# STM32N6 NPU Deployment — Documentation

> **Politecnico di Milano** — Multidisciplinary Project A.Y. 2024-2025  
> Supervised by Prof. Cristina Silvano & Dr. Marco Ronzani

## Live Documentation

**[→ Open the full documentation website](https://GiacomoColosio02.github.io/stm32n6-docs/)**

---

## About

This repository hosts the auto-generated Doxygen documentation for the STM32N6 NPU neural network deployment project.

We deployed three neural network architectures on the **STM32N6570-DK** board equipped with the **Neural-ART NPU (600 GOPS INT8)**:

| Model | Architecture | NPU Offload | Latency |
|-------|-------------|-------------|---------|
| MoveNet Lightning | CNN (MobileNetV2) | **94.7%** | 22 ms |
| YOLOv8n-pose | CNN + Detection Head | **87.9%** | 32 ms |
| TinyBERT | Transformer (BERT) | **64.4%** | >100 ms |

---

## Documentation Structure

| Section | Description |
|---------|-------------|
| **Part 1 — Narrative** | 6 chapters from edge computing theory to final results |
| **Chapter 1** | Introduction to Edge AI and quantization |
| **Chapter 2** | STM32N6570-DK hardware and Neural-ART NPU architecture |
| **Chapter 3** | Toolchain: ST Model Zoo → ST Edge AI Core → STM32CubeIDE |
| **Chapter 4** | Step-by-step deployment workflow |
| **Chapter 5** | Case studies with real profiling data |
| **Chapter 6** | Results and comparative analysis |
| **Part 2 — Code Reference** | Annotated source timeline (13 Python + 10 C files) |
| **Part 3 — Module Groups** | Files organised by architectural layer |

---

## Authors

| Name | Student ID |
|------|-----------|
| Giacomo Colosio | 10775899 |
| Sebastiano Colosio | 10555512 |
| Patrizio Acquadro | 11087897 |
| Tito Nicola Drugman | 10847631 |

---

## Tech Stack

- **Board:** STM32N6570-DK (ARM Cortex-M55 @ 800 MHz + Neural-ART NPU)
- **Tools:** ST Edge AI Core v2.1.0, STM32CubeIDE 1.18.1, STM32CubeProgrammer
- **Models:** MoveNet Lightning (INT8), YOLOv8n-pose (INT8), TinyBERT (INT8)
- **Docs:** Doxygen 1.9+ with custom HTML layout

---

## Repository Layout

Doxygen outputs everything flat by default (hundreds of files in root). We post-process that output into tidy subfolders:

| Folder | Contents |
|--------|----------|
| `css/` | stylesheets |
| `js/`  | navtree, search, per-file JS indexes |
| `images/ui/` | Doxygen UI icons (folders, arrows, sync) |
| `images/authors/` | author photos |
| `images/board/` | STM32N6570-DK board photos |
| `graphs/` | call/include graph SVGs (+ `.map` / `.md5`) |
| `search/` | untouched — Doxygen ships it as a subfolder |
| *root* | HTML pages, `README.md` |

## Regenerating the Docs

Doxygen overwrites its output directory on every run, so the reorganisation would be lost without a post-processing step. Run:

```bash
doxygen Doxyfile          # regenerate flat output in repo root
python3 tools/reorg.py    # re-apply the subfolder layout + path rewrites
git add -A && git commit -m "regen docs"
```

`tools/reorg.py` is idempotent — safe to run more than once. If you add new non-Doxygen images (e.g. more board photos, authors), extend the hard-coded sets at the top of that script.

---

*Documentation generated with Doxygen — source code available upon request.*
