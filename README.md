# PATCH ENSEMBLES FOR ROBUST SALMON RE-IDENTIFICATION WITH WEAK TRAJECTORY LABELS

**Last updated:** May 5, 2026

**Authors:** Espen Uri Høgstedt, Christian Schellewald, Annette Stahl, Rudolf Mester

**Repository:** salmon-reid-patch-ensemble

## 🐟 Abstract
Salmon re-identification in commercial net-pens is challenging due to large populations, which impose strict accuracy requirements and make large-scale labeled data acquisition infeasible.
Trajectory IDs can be used as proxy labels, but this introduces trajectory-ID bias.
To address these challenges, we propose a patch-based re-identification framework that fuses patch-level predictions into a salmon identity decision.
A key component of our method is the prediction of the salmon's lateral line, enabling the extraction of texture-anchored patches and patch slices.
To enable realistic evaluation, we introduce an experimental setup using multiple cameras placed 6 m apart, allowing the same fish to be recorded in different trajectories. This enables the construction of a cross-camera test set through manual match confirmation.
Our ensemble approach outperforms full-image baselines in same-trajectory validation (0.932 to 0.965 mAP) and cross-camera testing (0.609 to 0.860 mAP). The substantial improvements in the cross-camera setting demonstrate improved generalizability and robustness.

## 📂 Project Structure

<details>
<summary><strong>📁 data_generation/</strong> - Notebooks for creating salmon re-identification datasets from raw videos (Sections 3.1-3.5)</summary>
<br>

- **`track_salmon.ipynb`** - Yields salmon and body part trajectories from raw salmon video.
- **`create_reid_dataset.ipynb`** - Yields salmon images and labels (salmon direction, mask + corners for Q1 and Q2) from raw videos and trajectories.
 
</details>
<details>
<summary><strong>📁 helpers/</strong> - Helper files for the other files in the project </summary>
<br>

- **`data_generation_helpers/`** - Helpers for `data_generation/create_reid_dataset`.
- **`augmentation_helpers/`** - Helpers for extracting augmented and cropped images from full-salmon images and labels.
- **`test_helpers/`** - Helper for `training_testing/test_reid_network.py`.
- **`training_helpers/`** - Helper for `training_testing/train_reid_network.py`.
- 
</details>
---


## 📊 Datasets


Datasets can be found [here]()


## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@InProceedings{Hogstedt_2026_ICIP,
    author    = {H{\o}gstedt, Espen Uri and Schellewald, Christian and Stahl, Annette and Mester, Rudolf},
    title     = {PATCH ENSEMBLES FOR ROBUST SALMON RE-IDENTIFICATION WITH WEAK TRAJECTORY LABELS},
    booktitle = {2026 IEEE International Conference on Image Processing (ICIP)},
    year      = {2026},
}
```

## 📬 Contact

For questions or collaborations, feel free to reach out via email.

---

This work is developed as part of the cAIge project, funded by the Research Council of Norway, to support the aquaculture industry in achieving **continuous, automated, and precise** salmon welfare monitoring.
