# Raman-mapping-graphene
## Overview

This repository implements a complete, reproducible machine-learning pipeline for spatially-resolved Raman spectroscopy of graphene and graphite. 
The pipeline takes raw dual-window Raman map data (two overlapping spectral windows acquired on a confocal Raman microscope), produces baseline-corrected and normalised spectra, 
fits the D, G, D′, 2D, and D+G bands, clusters pixels into physically distinct domains, and trains a 1D convolutional neural network to classify domains with Grad-CAM saliency for interpretability.

The pipeline is designed around a single principle: spatially correlated hyperspectral data requires spatially-aware validation. 
Random pixel splitting leaks neighbour information across the train/test boundary and inflates accuracy. 
This implementation uses contiguous y-band splitting so the test set is a genuinely unseen spatial region.

## Key features

1) Spectral stitching of two acquisition windows (1012–2128 cm⁻¹ and 2047–3002 cm⁻¹) into a unified 1799-point spectrum per pixel over 1200–3000 cm⁻¹.
2) ALS baseline correction (asymmetric least-squares, λ = 10⁵, p = 0.01) for fluorescence removal.
3) G-band normalization so I_D and I_2D directly equal the physically meaningful I_D/I_G and I_2D/I_G ratios.
4) Bounded Lorentzian peak fitting for D, G, D′, 2D, and D+G bands with fallback handling on convergence failure.
5) PCA + k-means clustering on a hybrid feature space (spectral shape PCs + intensity ratios) for unsupervised domain discovery.
6) Spatially-aware train/val/test split using contiguous y-bands to prevent autocorrelation leakage.
7) 1D-CNN classifier (~210 k parameters, four conv blocks, multi-scale receptive field 15–1150 cm⁻¹) with weighted sampling for class imbalance.
8) Physics-motivated augmentation: Gaussian noise, intensity scaling, circular spectral shift.
9) Grad-CAM saliency to verify the network learns physically meaningful spectral features (D, G, D′, 2D bands) rather than baseline artefacts.


## Results

On the spatially-separated test set (n = 189 pixels, bottom 20 % of the y-range):

Class| Domain | n | Precision | Recall |F1| \
0 | Pristine graphite interior | 44 | 1.000 | 1.000 | 1.000 \
1 | Defective FLG | 49 | 0.980 | 1.000 | 0.990\
2 | Transition zone | 77 | 1.000 | 0.883 |0.938\
3 | Near-monolayer hotspots | 19 | 0.704 | 1.000 |0.826\
  | Weighted avg | 189 | 0.965 |0.952 | 0.955\

The pristine graphite domain is perfectly separated. All 19 near-monolayer pixels are recovered (recall = 1.000); the lower precision reflects 8 transition-zone pixels misclassified at the physically continuous boundary between defective FLG and monolayer regions — an expected ambiguity at the scale of the laser spot.

Grad-CAM saliency maps confirm the network attends to the D (~1350 cm⁻¹), G (~1580 cm⁻¹), and 2D (~2700 cm⁻¹) bands rather than the 1200–1300 cm⁻¹ baseline region, providing physics-based model validation beyond accuracy metrics.

## Installation

bashgit clone https://github.com/<user>/raman-graphene-ml.git
cd raman-graphene-ml
pip install -r requirements.txt

Dependencies: numpy, pandas, scipy, scikit-learn, matplotlib, torch, openpyxl.

## Usage

The pipeline is exposed as a single script with a command-line interface:

bashpython raman_pipeline.py --data path/to/maps/ --out results/

Arguments

FlagDescription--dataDirectory containing the two Raman map Excel files (map2A and map2B).--outOutput directory for figures, fitted features, cluster labels, and trained model weights.--no-trainRun preprocessing, fitting, and clustering only; skip CNN training.

## Repo Outputs

The script produces:

1) Spatial maps of I_G, I_2D/I_G, I_D/I_G, I_2D, cluster labels, and train/val/test partition.
2) Per-pixel feature matrix (37 columns: amplitude, position, FWHM, area, SNR per band, plus derived ratios).
3) Training and validation loss/accuracy curves.
4) Confusion matrix and per-class classification report.
5) Grad-CAM saliency overlays per cluster.
6) Trained model checkpoint.


## Pipeline architecture

Raw dual-window Raman .xlsx
        │
        ▼
load_and_stitch ──▶ als_baseline ──▶ preprocess_spectra (G-norm)
        │
        ▼
fit_peak (Lorentzian × 5) ──▶ extract_features (37-col matrix)
        │
        ▼
run_pca ──▶ cluster_pixels (k-means, k = 4) ──▶ spatial_split
        │
        ▼
RamanDataset ──▶ build_loaders (weighted sampling)
        │
        ▼
RamanCNN1D ──▶ train_model ──▶ evaluate
        │
        ▼
GradCAM1D ──▶ save_gradcam_figure

## Module reference

**Preprocessing.** load_and_stitch, als_baseline, preprocess_spectra.\
**Peak fitting.** lorentzian, fit_peak, extract_features.\
**Unsupervised analysis.** run_pca, cluster_pixels.\
**Splitting and loaders.** spatial_split, build_loaders, RamanDataset.\
**Model.** RamanCNN1D — Conv1D (1→32, k=15) → Conv1D (32→64, k=9) → Conv1D (64→128, k=7) → Conv1D (128→256, k=5) → GAP → FC (128) → Dropout (0.4) → FC (4).\
**Training.** train_model, evaluate — Adam (lr = 10⁻³, wd = 10⁻⁴), cosine annealing over 30 epochs, gradient clipping (max norm 1.0), early stopping (patience 8).\
**Interpretability.** GradCAM1D, save_gradcam_figure.\
**Visualisation.** make_grid_map, save_spatial_maps, save_training_curves.

## Dataset

A 27 × 35 µm Raman map (945 pixels, 1 µm step) of a graphene/graphite sample acquired with 532 nm excitation. 
Two acquisition windows are stitched and cropped to 1200–3000 cm⁻¹ yielding 1799 spectral channels per pixel. 
The pipeline accepts any pair of Excel files following the same dual-window format and is directly extensible to other 2D materials (MoS₂, WS₂, h-BN) by updating the peak-centre wavelengths in the fitting configuration.

## Extending to other materials

The architecture is material-agnostic. To adapt to a different 2D material, modify only the peak-centre wavelengths and fitting windows in the configuration dictionary at the top of raman_pipeline.py. 
The 1D-CNN, training loop, and Grad-CAM module require no changes.



## References

- Ferrari et al., Phys. Rev. Lett. 97, 187401 (2006).
- Ferrari & Basko, Nat. Nanotechnol. 8, 235 (2013).
- Malard et al., Phys. Rep. 473, 51 (2009).
- Cançado et al., Nano Lett. 11, 3190 (2011).
- Eilers & Boelens, Baseline correction with asymmetric least squares smoothing (2005).
- Selvaraju et al., Grad-CAM, ICCV (2017).


