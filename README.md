# Controllable Complex Human Motion Video Generation via Text-to-Skeleton Cascades

<p align="center">
  <a href="https://arxiv.org/abs/2603.08028"><img src="https://img.shields.io/badge/arXiv-2603.08028-b31b1b.svg" alt="arXiv"></a>
</p>

Official repository for the paper:

> **Controllable Complex Human Motion Video Generation via Text-to-Skeleton Cascades**
>
> Ashkan Taghipour, Morteza Ghahremani, Zinuo Li, Hamid Laga, Farid Boussaid, Mohammed Bennamoun

We propose a two-stage cascaded framework for generating controllable videos of complex human motions (backflips, cartwheels, martial arts). An autoregressive text-to-skeleton model generates 2D pose sequences from natural language, and a pose-conditioned video diffusion model with DINO-ALF preserves appearance under large deformations and self-occlusions. We also release a Blender-based synthetic dataset of 2,000 complex-motion videos.

---

## 📁 Repository Structure

```
├── synthetic_data_generation/   # 🎬 Blender-based SDG pipeline for the training dataset
│   ├── render_motion.py         #    Rendering script with pose annotation extraction
│   ├── README.md                #    Detailed usage guide
│   └── samples/                 #    Pipeline diagram & sample outputs
│
├── video_generation/            # 🧠 Text-to-skeleton & pose-to-video models
│   └── ...                      #    (coming soon)
```

## 🎬 Synthetic Data Generation

Our Blender-based pipeline for rendering synthetic human motion videos with automatic 2D pose annotations. Pairs Mixamo characters with Poly Haven HDR environments to produce diverse training data with perfect ground truth.

👉 See [`synthetic_data_generation/README.md`](synthetic_data_generation/README.md) for the full guide.

## 🧠 Video Generation

*Coming soon* — code for the text-to-skeleton and pose-conditioned video generation models.

---

## 📄 Citation

If you find this work useful, please cite:

```bibtex
@article{taghipour2025controllable,
  title={Controllable Complex Human Motion Video Generation via Text-to-Skeleton Cascades},
  author={Taghipour, Ashkan and Ghahremani, Morteza and Li, Zinuo and Laga, Hamid and Boussaid, Farid and Bennamoun, Mohammed},
  journal={arXiv preprint arXiv:2603.08028},
  year={2025}
}
```
