

# Resolving the Ambiguity of Complete-to-Partial Point Cloud Registration for Image-Guided Liver Surgery with Patches-to-Partial Matching

## Introduction

This repository is a follow-up to the green [LiverMatch](https://github.com/zixinyang9109/LiverMatch) project. We provide *in silico* and *in vitro datasets* to facilitate automatic 3D-3D rigid registration for image-guided liver surgery using learning-based point cloud correspondence methods.

To address the prevalent complete-to-partial ambiguity challenge in this domain, we propose the Patches-to-Partial (P2P) strategy, represented as an easily pluggable module that can be seamlessly integrated into learning-based registration pipelines without modifying their end-to-end structure or introducing additional trainable parameters.

---
## P2P Demo

To test the P2P module, we offer several samples. Please run the following demo script:

```bash
P2P_Demo/demo.py
```

It will generate a visual output like :

![plot](./P2P_demo.png)

### Dependencies

Ensure your environment includes:

* `torch`
* [`pyvista`](https://pyvista.org/)
* [`pointnet2_ops`](https://github.com/erikwijmans/Pointnet2_PyTorch)

---

## Datasets

You can download both in silico and in vitro phantom datasets from the following link:

📂 [Google Drive – Datasets](https://drive.google.com/drive/folders/1CpcMFqaiyg3eVnSEItCi1N8hmEeDopkd?usp=sharing)

All data is stored in `.npz` format. To quickly explore the contents, run the visualization script:

```bash
Eva_in_silico/vis_dataset.py
```

> ⚠️ This script requires a Python environment with `torch`, `numpy`, `open3D`, and `pyvista`. No specific version constraints apply.

As discussed in the paper, our datasets have limitations, particularly in terms of realistic cropping. We hope this work can inspire the development of more comprehensive datasets in the future.

---

## Baselines

We provide configuration files, scripts, and pretrained weights to reproduce results from several baselines:

* [Go-ICP](https://github.com/aalavandhaann/go-icp_cython)
* [RoITr](https://github.com/haoyu94/RoITr)
* [RegTR](https://github.com/yewzijian/RegTR)
* [LiverMatch](https://github.com/zixinyang9109/LiverMatch)
* [Lepard](https://github.com/rabbityl/lepard)

> ✅ You can use the Conda environment from [Lepard](https://github.com/rabbityl/lepard) to run all learning-based baselines.

> ⚠️ Full instructions and scripts for baseline reproduction will be added soon.

---

## Evaluation

You may refer to the evaluation scripts provided in the `Eva_in_silico` and `Eva_in_vitro` folders:

### *In Silico* Evaluation

```bash
# Update the data paths before running
Eva_in_silico/Table_I&II.py                    
Eva_in_silico/Fig_4_test_data_property.py
Eva_in_silico/Fig_6_compare_error_curves.py
Eva_in_silico/Fig_7_plot_success_rate.py
```

### *In Vitro* Evaluation

```bash
# Update the data paths before running
Eva_in_vitro/Table_VII.py                      
Eva_in_vitro/Fig_8_compare_K.py
Eva_in_vitro/Fig_9_compare_ours_ransac.py
```
---
## Tips & Notes

* **Uniform point density** is critical for sim-to-real generalization. Thus, the preprocessing is very important.
* **Higher point cloud density** generally yields better registration accuracy.
* **RoITr’s sparse superpoints** may degrade accuracy; consider tuning transformer parameters. However, breaking the partial target into many small patches can hinder correspondence due to reduced saliency.
* For **KPCov**, ensure that multi-level downsampling is enabled; otherwise, it may not extract features effectively.
* In deformation simulations, confining zero-displacement boundary conditions to a small region often introduces a significant **rigid component**. Aligning using volumetric vertices as markers helps reduce this rigid influence.

---

## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{p2p,
  author={Yang, Zixin and Heiselman, Jon S. and Han, Cheng and Merrell, Kelly and Simon, Richard and Linte, Cristian A.},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={Resolving the Ambiguity of Complete-to-Partial Point Cloud Registration for Image-Guided Liver Surgery with Patches-to-Partial Matching}, 
  year={2025},
  pages={1-14},
  doi={10.1109/JBHI.2025.3583875}
}
```


