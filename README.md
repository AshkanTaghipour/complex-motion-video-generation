# Controllable Complex Human Motion Video Generation via Text-to-Skeleton Cascades

Official repository for the paper:

> **Controllable Complex Human Motion Video Generation via Text-to-Skeleton Cascades**
>
> Ashkan Taghipour, Morteza Ghahremani, Zinuo Li, Hamid Laga, Farid Boussaid, Mohammed Bennamoun
>
> [[arXiv]](https://arxiv.org/abs/2603.08028)

## 📁 Repository Structure

```
├── synthetic_data_generation/   # Blender-based SDG pipeline for the training dataset
│   ├── render_motion.py         # Rendering script with pose annotation extraction
│   └── README.md                # Detailed usage guide
└── ...                          # (more components coming soon)
```

## 🎬 Synthetic Data Generation

See [`synthetic_data_generation/README.md`](synthetic_data_generation/README.md) for the complete guide on reproducing our Blender-based dataset of 2,000 complex-motion synthetic videos.

## 📄 Citation

```bibtex
@article{taghipour2025controllable,
  title={Controllable Complex Human Motion Video Generation via Text-to-Skeleton Cascades},
  author={Taghipour, Ashkan and Ghahremani, Morteza and Li, Zinuo and Laga, Hamid and Boussaid, Farid and Bennamoun, Mohammed},
  journal={arXiv preprint arXiv:2603.08028},
  year={2025}
}
```
