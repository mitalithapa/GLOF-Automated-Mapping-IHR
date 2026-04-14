# 🛰️ GLOF Watch — Automated Glacial Lake Mapping & Monitoring

> A cloud-based, machine-learning-powered tool for automated detection, delineation, and spatio-temporal monitoring of glacial lakes in the Indian Himalayan Region (IHR) — deployed on Google Earth Engine.

**Author:** Mitali Thapa  

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Key Features](#-key-features)
- [Repository Structure](#-repository-structure)
- [Methodology](#-methodology)
  - [Study Area](#1-study-area)
  - [Data Sources](#2-data-sources)
  - [Preprocessing Pipeline](#3-preprocessing-pipeline)
  - [Feature Engineering](#4-feature-engineering)
  - [Hybrid Model Architecture](#5-hybrid-model-architecture)
  - [Class Imbalance Handling](#6-class-imbalance-handling)
  - [GEE App Development](#7-gee-app-development)
  - [Accuracy Assessment](#8-accuracy-assessment)
- [Image Processing Workflow](#-image-processing-workflow)
- [GEE Application](#-gee-application)
- [Notebook Pipeline Guide](#-notebook-pipeline-guide)
- [Results](#-results)
- [Limitations](#-limitations)
- [Future Scope](#-future-scope)
- [References](#-references)

---

## 🔭 Overview

Glacial lakes form in high-mountain terrain when meltwater accumulates in depressions carved by retreating glaciers. Due to accelerating global warming, the number and area of these lakes have grown dramatically — a global assessment estimates a **~53% increase between 1990 and 2018**. As these lakes grow, so does the risk of **Glacial Lake Outburst Floods (GLOFs)** — catastrophic events capable of causing severe downstream loss of life and infrastructure damage.

Traditional monitoring relies on manual digitization of satellite imagery: accurate, but time-consuming, subjective, and completely unscalable for the vast, remote terrain of the Himalayas. This project addresses that gap with a fully automated, cloud-based pipeline.

**GLOF Watch** is a hybrid remote sensing + machine learning system that:
- Automatically detects and maps glacial lake boundaries from Sentinel-2 imagery
- Integrates topographic features to suppress shadow/snow false positives
- Monitors lake area change over time for early GLOF risk detection
- Is deployed as an interactive Google Earth Engine (GEE) application

---

## ⚠️ Problem Statement

> *"To develop an automated, accurate, and scalable framework for glacial lake detection and monitoring using multi-spectral satellite imagery and advanced machine learning/deep learning techniques, capable of overcoming environmental and spectral challenges in high-mountain regions."*

Key challenges motivating this work:

| Challenge | Description |
|---|---|
| **Spectral Confusion** | Water shares spectral signatures with shadows, snow, and debris-covered ice |
| **Cloud Cover** | Frequent cloud obstruction in optical satellite imagery reduces data availability |
| **Topographic Effects** | Mountain shadows and steep slopes introduce systematic false positives |
| **Seasonal Variability** | Frozen lakes and snow cover disrupt consistent detection across seasons |
| **Small/Turbid Lakes** | Many glacial lakes are tiny or sediment-laden, making detection difficult |
| **Class Imbalance** | Lake pixels are a tiny minority compared to background terrain pixels |
| **Scalability** | Manual methods cannot cover the scale of the IHR |

---

## ✨ Key Features

- **Automated water body detection** using multi-spectral Sentinel-2 imagery at 10 m resolution
- **Shadow masking** via hillshade analysis derived from a hybrid ALOS DEM
- **Snow masking** using NDSI filtering to separate frozen lakes from open water
- **Random Forest classifier** (250 trees) trained on locally digitized training polygons across 26 high-risk lake basins
- **Heuristic MNDWI fallback** mode when no training data is available in a region
- **Dynamic surface area computation** per scan, with a high-volume alert threshold
- **Spectral signature profiling** of detected water bodies
- **Interactive GEE web application** with real-time scan, layer toggles, and a telemetry terminal
- **Spatio-temporal monitoring** capability for multi-date change detection

---

## 📁 Repository Structure

```
glof-watch/
├── README.md                             # Project documentation and theoretical framework
│
├── data/
│   └── LakeLocations.docx                # Ground truth sampling sites
│                                         # (Samundar Tapu, Gepang, Tso Moriri, and others)
│
├── docs/
│   ├── Report.pdf                        # Full research report and methodology
│   └── images/
│       └── workflow_architecture.png     # End-to-end pipeline flowchart
│
├── notebooks/
│   ├── Raster-To-ML.ipynb                # ① Master pipeline — TIF ingestion, array
│   │                                     #   flattening, and ML preparation
│   ├── Rasterization.ipynb               # ② Vector-to-raster conversion for spatial
│   │                                     #   model inputs
│   ├── EnhancedTif.ipynb                 # ③ Multi-band TIF enhancement and spectral
│   │                                     #   index calculation (NDWI, MNDWI, NDSI)
│   ├── Non-LakeSamples.ipynb             # ④ Targeted generation of negative samples
│   │                                     #   (shadow masks, snow masks)
│   ├── UsingPolygons.ipynb               # ⑤ Spatial training and validation using
│   │                                     #   localized polygon masks
│   ├── Downsampling-Model.ipynb          # ⑥ Pixel downsampling to mitigate background
│   │                                     #   class dominance
│   └── Hybrid Sampling + SMOTE.ipynb     # ⑦ Advanced Synthetic Minority Oversampling
│                                         #   Technique (SMOTE) implementation
│
└── app/
    └── PrototypeUI.ipynb                 # Streamlit-based web app for interactive
                                          # model inference
```

> The GEE monitoring application (`GLOF Watch v2.5`) is a separate JavaScript-based tool running on Google Earth Engine — see the [GEE Application](#-gee-application) section for setup instructions.

---

## 🔬 Methodology

### 1. Study Area

The study focuses on the **Indian Himalayan Region (IHR)** — one of the world's most glacierized and GLOF-vulnerable areas. Monitoring sites include:

| Site | Region |
|---|---|
| Baralacha La | Lahaul, Himachal Pradesh |
| Samundar Tapu | Lahaul, Himachal Pradesh |
| Zullu Lake | Lahaul, Himachal Pradesh |
| Karzok | Ladakh |
| Khangchengyao | Sikkim |

The region is characterized by high altitude, rugged terrain, frequent cloud cover, and a diversity of glacial lake types (moraine-dammed, ice-dammed, supraglacial, and proglacial).

---

### 2. Data Sources

| Data Type | Source | Resolution |
|---|---|---|
| Optical Satellite Imagery | Sentinel-2 MSI (Harmonized Surface Reflectance) | 10–20 m |
| Topographic Data | JAXA ALOS AW3D30 V4.1 (Global DEM) | 30 m |
| Custom High-Res DEMs | User-uploaded GEE assets for 20 lake basins | Variable |
| Training Polygons | Manually digitized GEE shapefiles for 26 basins | — |

**Training Basins Covered:**
`zullu`, `vasuki`, `samundratapu`, `ravibasin`, `bhaga`, `beas`, `Khanchengyao`, `barlachla`, `bhaga12`, `bhaga13`, `chandra`, `fanchan`, `gepangghat`, `hangu`, `kaktital`, `karzok`, `lamdalravi`, `langpopeak`, `neelkanth`, `paldanlamotal`, `tsoparidhi`, `vasuki2/3/4`, and more.

---

### 3. Preprocessing Pipeline

**Cloud Masking**
Sentinel-2 QA60 band is used to mask cloud and cirrus pixels using bitwise flags (bits 10 and 11). All valid pixels are then scaled to surface reflectance (÷10000).

**Temporal Compositing**
An image collection is filtered by date range and cloud percentage (`< 15%`), then reduced to a **median composite** — suppressing remaining cloud artifacts and seasonal noise.

**DEM Fusion**
A hybrid DEM is constructed by mosaicking globally available ALOS DEM tiles with custom high-resolution DEMs for specific lake basins. This ensures accurate slope and hillshade derivation in areas where global DEMs are too coarse.

---

### 4. Feature Engineering

A 10-band multi-modal feature stack is constructed per pixel:

| Feature | Formula / Source | Purpose |
|---|---|---|
| **B2, B3, B4, B8, B11** | Sentinel-2 spectral bands | Spectral signature |
| **NDWI** | `(B3 − B8) / (B3 + B8)` | Open water detection |
| **MNDWI** | `(B3 − B11) / (B3 + B11)` | Water vs. built-up / debris |
| **NDSI** | `(B3 − B11) / (B3 + B11)` | Snow / frozen lake mask |
| **Slope** | Derived from DEM | Terrain steepness — suppresses shadow FPs |
| **Hillshade** | Solar azimuth + zenith from image metadata | Shadow mask |

**Solar geometry** (azimuth and zenith angle) is extracted directly from Sentinel-2 image metadata to compute a scene-accurate hillshade, ensuring the shadow mask is dynamically calibrated per acquisition.

---

### 5. Hybrid Model Architecture

The project implements a **dual-model framework**:

#### 5a. Random Forest Classifier (Deployed in GEE App)

- **Algorithm:** `ee.Classifier.smileRandomForest`
- **Trees:** 250
- **Variables per split:** 7
- **Min leaf population:** 7
- **Input features:** All 10 bands listed above
- **Training:** Dynamic — samples are extracted from the composite image at `scale = 10m` over manually labelled polygons within the current region of interest
- **Post-classification refinement:** Lake mask is further filtered with a `Slope < 20°` terrain mask to remove cliff-face and shadow false positives

#### 5b. U-Net CNN (Semantic Segmentation — Research Component)

A U-Net convolutional neural network was implemented as the deep learning component for precise boundary segmentation:

- **Architecture:** Encoder-decoder with skip connections
- **Contracting path (Encoder):** Captures contextual spatial features at multiple scales
- **Expanding path (Decoder):** Reconstructs spatial resolution for precise localization
- **Advantage over RF:** Captures spatial context across neighbouring pixels; superior at irregular and small lake boundaries
- **Training challenge:** Requires larger GPU compute than the GEE-hosted RF classifier; used as a research validation model

The two models are **complementary**: RF provides rapid, scalable pixel-level classification; U-Net refines boundaries and handles spatially complex lake shapes.

---

### 6. Class Imbalance Handling

Glacial lake pixels are a tiny fraction of total landscape pixels — a severe class imbalance that biases classifiers toward background terrain. This was addressed through a **hybrid sampling strategy**:

| Strategy | Description |
|---|---|
| **Stratified downsampling** | Majority class (non-water) pixels reduced to a manageable count |
| **Augmented lake pixels** | Additional lake-class samples added from secondary GEE assets |
| **Balanced training set** | Final training dataset constructed with controlled class ratios |

This approach was preferred over pure SMOTE (synthetic oversampling) to retain real spectral distribution from the sensor.

---

### 7. GEE App Development

The complete GEE application code (`glof_watch_v2.5.js`) is the primary deliverable of this project — a production-ready, interactive monitoring tool. See [`gee_app/`](./gee_app/) for the full source.

**App architecture:**

```
UI Layer (Sidebar + Map)
├── Step 1 — Observation Basin: Pre-defined site selector / custom polygon draw
├── Step 2 — Temporal Parameters: Start & end date inputs
├── Step 3 — Advanced Tuning: Heuristic MNDWI threshold override
├── Run Button → triggers full analysis pipeline
├── Telemetry Terminal — real-time log of processing stages
└── Results Dashboard
    ├── Detected lake surface area (km²)
    ├── Spectral signature profile chart (LineChart, 490–1610 nm)
    ├── Model transparency card (RF accuracy % or fallback warning)
    └── High-volume alert panel (threshold: 0.1 km²)

Map Layer
├── Sentinel-2 SWIR/NIR/R false-colour composite
├── Glacial lake boundary mask (cyan)
└── Custom AOI outline (if polygon drawn)
```

**Fallback mode:** If no training polygons overlap the current ROI, the app automatically falls back to an MNDWI heuristic (`MNDWI > threshold`, `Slope < 10°`, `NDSI < 0.3`) rather than failing silently.

---

### 8. Accuracy Assessment

- **Validation approach:** Confusion matrix from the RF classifier's own training evaluation (`classifier.confusionMatrix().accuracy()`)
- **Reference data:** Manually digitized lake boundary polygons
- **Metrics:** Overall accuracy (%), displayed in the app's model transparency panel
- **Error analysis:** Commission errors (shadows/snow misclassified as water) were quantified and mitigated via the slope and hillshade filters

---

## 🖼️ Image Processing Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                      INPUT DATA SOURCES                         │
│  Sentinel-2 MSI (SR Harmonized)  +  ALOS DEM / Custom DEMs     │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                     PREPROCESSING                               │
│  • QA60 Cloud + Cirrus Masking                                  │
│  • Cloud % Filter (< 15%)                                       │
│  • Temporal Median Compositing                                  │
│  • DEM Fusion (Global ALOS + Custom Basin DEMs)                 │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                  FEATURE ENGINEERING                            │
│  Spectral:   B2, B3, B4, B8, B11                                │
│  Indices:    NDWI, MNDWI, NDSI                                  │
│  Topographic: Slope (DEM), Hillshade (Solar Geometry)           │
│                  → 10-band feature stack                        │
└───────────────────────┬─────────────────────────────────────────┘
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
┌────────────────────┐   ┌─────────────────────────┐
│  TRAINING DATA     │   │   HEURISTIC FALLBACK     │
│  (GEE Polygons)    │   │  MNDWI > threshold       │
│  26 basins         │   │  Slope < 10°             │
│  class balancing   │   │  NDSI < 0.3              │
└────────┬───────────┘   └────────────┬────────────┘
         │                            │
         ▼                            │
┌────────────────────┐               │
│  RANDOM FOREST     │               │
│  250 trees         │               │
│  7 vars/split      │               │
│  10m scale sample  │               │
└────────┬───────────┘               │
         └──────────┬────────────────┘
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                POST-CLASSIFICATION REFINEMENT                   │
│  • Terrain mask: Slope < 20° (removes cliff/shadow noise)       │
│  • .selfMask() to strip zero values                             │
└───────────────────┬─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                     OUTPUT & MONITORING                         │
│  • Lake boundary vector layer (cyan overlay)                    │
│  • Surface area (km²) via ee.Reducer.sum on pixel area          │
│  • Spectral signature profile chart                             │
│  • Model accuracy report                                        │
│  • High-volume GLOF risk alert                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🌐 GEE Application

The interactive GLOF Watch monitoring tool is built entirely in the **Google Earth Engine JavaScript API (v2.5)** and runs as a standalone cloud application — separate from the Python notebooks in this repository.

**How to run:**
1. Open [code.earthengine.google.com](https://code.earthengine.google.com)
2. Create a new script and paste the full GEE app source code
3. Click **Run**

> ⚠️ **Note:** The app references private GEE assets (`users/mitalithapa07/...`) for training polygons and custom DEMs. To run in a new account, these assets must be shared or replaced with equivalent training shapefiles.

**App capabilities:**
- Select from 5 pre-defined high-risk monitoring sites or draw a custom AOI polygon on the map
- Set custom date ranges for seasonal or multi-year analysis
- Adjust the heuristic MNDWI threshold for fallback detection mode
- Toggle Sentinel-2 base composite and lake mask layers independently
- Monitor real-time processing stages via the telemetry terminal
- View lake surface area (km²), spectral signature chart, model accuracy, and GLOF risk alert

---

## 📓 Notebook Pipeline Guide

The `notebooks/` directory contains the full local ML development pipeline, intended to be run in sequence. Each notebook is a self-contained stage:

| # | Notebook | Role in Pipeline |
|---|---|---|
| ① | `Raster-To-ML.ipynb` | **Master pipeline** — ingests GeoTIFF exports from GEE, flattens multi-band rasters into structured arrays, and prepares feature matrices for ML training |
| ② | `Rasterization.ipynb` | Converts vector polygon shapefiles (lake / non-lake labels) into raster masks aligned to the Sentinel-2 grid |
| ③ | `EnhancedTif.ipynb` | Computes and appends spectral indices (NDWI, MNDWI, NDSI) and topographic bands to the raw TIF stack, producing the enhanced multi-band input |
| ④ | `Non-LakeSamples.ipynb` | Generates targeted **negative samples** — specifically mining shadow-affected and snow-covered pixels to reduce false positive rates during training |
| ⑤ | `UsingPolygons.ipynb` | Handles spatially-aware train/validation splitting using polygon masks, ensuring no spatial leakage between training and test sets |
| ⑥ | `Downsampling-Model.ipynb` | Applies stratified pixel downsampling to balance the heavily skewed background (non-lake) class against the minority lake class |
| ⑦ | `Hybrid Sampling + SMOTE.ipynb` | Implements **SMOTE** (Synthetic Minority Oversampling Technique) on top of downsampling for a full hybrid class-balance strategy |

**Streamlit App**

`app/PrototypeUI.ipynb` builds and launches a **Streamlit web interface** for interactive model inference — upload a TIF, run the trained classifier, and visualize the detected lake mask without writing any additional code.

To launch:
```bash
# From the app/ directory, after exporting the notebook to a .py script:
streamlit run PrototypeUI.py
```

---

## 📊 Results

- The Random Forest model, trained dynamically on local spectral profiles, consistently produces **high overall accuracy** (reported per scan in the app's transparency card)
- The topographic post-processing step (slope mask) significantly reduced commission errors from terrain shadows
- Lake surface area is computed at 10 m resolution with `bestEffort` pixel sampling
- The spectral signature chart confirms the expected water absorption profile across the 490–1610 nm range
- The dual-model approach (RF + U-Net) outperforms single-model and spectral-index-only baselines in heterogeneous terrain

---

## 🚧 Limitations

- Dependence on optical (Sentinel-2) imagery means performance degrades under persistent cloud cover — SAR integration would improve all-weather robustness
- Very small glacial lakes (< 0.01 km²) may be under-detected at 10 m resolution
- Custom GEE assets and training polygons are specific to the IHR; generalization to other mountain ranges requires new training data
- The U-Net model is currently a research component and has not been integrated into the live GEE application due to computational constraints
- Ground-truth validation data is limited to manually digitized boundaries; independent field survey data would strengthen accuracy metrics

---

## 🔭 Future Scope

- **SAR data fusion** (Sentinel-1) for cloud-penetrating all-weather detection
- **Transformer-based segmentation** (Swin Transformer, CNN-ViT hybrids) for improved generalization
- **Automated change detection pipeline** with configurable alert thresholds for lake expansion rate
- **Semi-supervised learning** to reduce dependence on manually labelled training polygons
- **Integration with national disaster management APIs** for real-time GLOF early warning dissemination
- **Global scalability** via transfer learning to other glacierized regions (Andes, Central Asia, Patagonia)

---

## 📚 References

1. Shugar, D. H., et al. (2020). Rapid worldwide growth of glacial lakes since 1990. *Nature Climate Change*, 10, 939–945.
2. Wang, X., et al. (2020). Glacial lake inventory and change in the China–Pakistan Economic Corridor from 1990 to 2020. *Earth System Science Data*.
3. Nie, Y., et al. (2021). Glacial lake inventory of the Tibetan Plateau. *Earth System Science Data*, 13, 741–766.
4. Chen, F., et al. (2021). Glacial lake detection using multi-source remote sensing data and machine learning. *Remote Sensing of Environment*.
5. Huggel, C., et al. (2002). Remote sensing based assessment of hazards from glacier lake outbursts. *Natural Hazards and Earth System Sciences*, 2, 309–316.
6. Buda, M., Maki, A., & Mazurowski, M. A. (2018). A systematic study of class imbalance problem in convolutional neural networks. *Neural Networks*, 106, 249–259.
7. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521, 436–444.
8. Li, J., et al. (2022). Deep learning for glacial lake segmentation from high-resolution remote sensing imagery. *Remote Sensing*, 14(3), 525.
9. Pekel, J. F., et al. (2016). High-resolution mapping of global surface water and its long-term changes. *Nature*, 540, 418–422.
10. Veh, G., et al. (2019). Detecting glacial lake outburst floods from Landsat time series. *Remote Sensing of Environment*, 237.
11. Drăguţ, L., & Blaschke, T. (2006). Automated classification of landform elements using object-based image analysis. *Geomorphology*, 81(3–4), 330–344.

---

<div align="center">

**GLOF Watch v2.5** — 

</div>
