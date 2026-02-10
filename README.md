# Content-Based Image Retrieval

## Authors
- Akash Shridhar Shetty
- Skandhan Madhusudhana

## Operating System and IDE
- macOS (M4 Mac Air)
- VS Code with terminal
- C++17 with OpenCV 4 (installed via Homebrew)

## Project Description
A content-based image retrieval (CBIR) system that finds similar images from a database of 1100+ images using various feature extraction methods and distance metrics. The system implements classic computer vision features (color histograms, texture analysis, spatial layout) and deep network embeddings (ResNet18) to compare and rank images by visual similarity. Two extensions were implemented: custom DNN embedding extraction using our own ResNet18 ONNX inference, and an interactive GUI for visual browsing of results with search functionality.

## Building the Project

```bash
# Navigate to project root
cd content-based-image-retrieval

# Create and enter build directory
mkdir -p build
cd build

# Configure with CMake
cmake ..

# Compile all executables
make
```

This builds five executables:
- `extract_features` — Extract features from all images and save to CSV
- `query` — Query the feature database to find similar images
- `compute_embeddings` — Extract custom DNN embeddings using ResNet18 (Extension)
- `gui_query` — Interactive GUI for image retrieval (Extension)
- `compare_embeddings` — Generate side-by-side DNN comparison images (Extension)

Verify executables were created:
```bash
ls -l extract_features query gui_query compute_embeddings compare_embeddings
```

## Running the Executables

**All commands below assume you are in the `build/` directory.**

### Step 1: Extract Features (run once per feature type)

```bash
# Task 1: Baseline features (147 values per image)
./extract_features ../data/olympus ../data/baseline_features.csv baseline

# Task 2: Histogram features (256 values per image)
./extract_features ../data/olympus ../data/histogram_features.csv histogram

# Task 3: Multi-histogram features (128 values per image)
./extract_features ../data/olympus ../data/multihistogram_features.csv multihistogram

# Task 4: Texture + Color features (272 values per image)
./extract_features ../data/olympus ../data/texture_features.csv texture

# Task 5: DNN features - SKIP (already provided as ResNet18_olym.csv)
# No extraction needed!

# Task 7: Custom features (209 values per image)
./extract_features ../data/olympus ../data/custom_features.csv custom
```

Each extraction takes ~1-2 minutes for 1106 images. Expected output:
```
========================================
Feature Extraction Program
========================================
...
Successfully extracted: 1106
Feature vector size: XXX values
========================================
Feature extraction completed successfully!
========================================
```

### Step 2: Query for Similar Images

```bash
# Task 1: Baseline (7x7 center, SSD)
./query ../data/olympus/pic.1016.jpg ../data/baseline_features.csv 3 baseline

# Task 2: Histogram (rg chromaticity, histogram intersection)
./query ../data/olympus/pic.0164.jpg ../data/histogram_features.csv 3 histogram

# Task 3: Multi-histogram (top/bottom halves, weighted intersection)
./query ../data/olympus/pic.0274.jpg ../data/multihistogram_features.csv 3 multihistogram

# Task 4: Texture + Color (Sobel magnitude + rg chromaticity)
./query ../data/olympus/pic.0535.jpg ../data/texture_features.csv 3 texture

# Task 5: DNN Embeddings (cosine distance)
./query ../data/olympus/pic.0893.jpg ../data/ResNet18_olym.csv 3 dnn
./query ../data/olympus/pic.0164.jpg ../data/ResNet18_olym.csv 3 dnn

# Task 7: Custom Blue Scene Detector (requires DNN CSV as 5th argument)
./query ../data/olympus/pic.0164.jpg ../data/custom_features.csv 5 custom ../data/ResNet18_olym.csv
```

## Extensions

### Extension 1: Custom DNN Embeddings (Skandhan)

Computes ResNet18 embeddings by running our own ONNX model inference, rather than using the pre-computed CSV from the assignment.

**Requirements:** Download `resnet18-v2-7.onnx` to `data/` directory.

```bash
# Compute embeddings (takes a few minutes)
./compute_embeddings ../data/resnet18-v2-7.onnx ../data/olympus/ ../data/my_dnn_features.csv

# Query with custom embeddings
./query ../data/olympus/pic.0893.jpg ../data/my_dnn_features.csv 3 dnn

# Generate side-by-side comparison images (saved to ../results/)
./compare_embeddings ../data/olympus/ ../data/ResNet18_olym.csv ../data/my_dnn_features.csv
```

**Results:** Both embedding sets produce meaningful results with similar distance ranges (~0.15-0.22). For some queries (pic.0893) the top matches are identical but reordered. For others (pic.0164) the results differ, showing that minor preprocessing differences affect rankings when many candidates have similar distances.

### Extension 2: Interactive GUI (Akash)

All-in-one visual interface with feature type selector, image browser, search bar, and results grid.

```bash
# Make sure you're in build/ directory
# Launch GUI (loads all feature databases at startup)
./gui_query ../data/olympus ../data/ResNet18_olym.csv
```

Expected output:
```
========================================
CBIR - Interactive GUI
========================================
Loading baseline features from ../data/baseline_features.csv...
  Loaded 1106 vectors (147D)
Loading histogram features from ../data/histogram_features.csv...
  Loaded 1106 vectors (256D)
Loading multihistogram features from ../data/multihistogram_features.csv...
  Loaded 1106 vectors (128D)
Loading texture features from ../data/texture_features.csv...
  Loaded 1106 vectors (272D)
Loading dnn features from ../data/ResNet18_olym.csv...
  Loaded 1106 vectors (512D)
Loading custom features from ../data/custom_features.csv...
  Loaded 1106 vectors (209D)
DNN database loaded for custom features (1106 vectors)
Found 1106 images

========================================
GUI Ready! Controls:
  Click any image to query it
  Trackbar or 1-6: switch feature type
  s: search by filename
  n/p: next/prev browser page
  q/ESC: quit
========================================
```

**GUI Controls:**
- **Click** any image to use it as the new target
- **Trackbar** or **1-6 keys** to switch feature type:
  - 1 = Baseline
  - 2 = Histogram
  - 3 = Multi-histogram
  - 4 = Texture
  - 5 = DNN
  - 6 = Custom
- **s** — Search mode (type filename, Enter to select, Esc to cancel)
- **n/p** — Browse next/previous page of images
- **q/ESC** — Quit

**GUI Features:**
- Top 6 matches shown with green borders and distance values
- Bottom 3 least similar shown with red borders (custom feature type only)
- Image browser strip at bottom for browsing the database
- Search bar for finding images by filename

## File Structure

```
content-based-image-retrieval/
├── CMakeLists.txt
├── Makefile
├── README.md
├── run_all.sh
├── .gitignore
├── include/
│   ├── features.h
│   ├── distance.h
│   └── utils.h
├── src/
│   ├── main_extract_features.cpp
│   ├── main_query.cpp
│   ├── features.cpp
│   ├── distance.cpp
│   ├── utils.cpp
│   ├── embedding_extractor.cpp      (Extension 1)
│   ├── compare_embeddings.cpp       (Extension 1)
│   └── gui_query.cpp                (Extension 2)
├── data/
│   ├── olympus/                     (image database, 1106 images)
│   ├── ResNet18_olym.csv            (provided DNN features, 512D)
│   ├── resnet18-v2-7.onnx           (ONNX model for Extension 1)
│   ├── baseline_features.csv        (generated, 147D)
│   ├── histogram_features.csv       (generated, 256D)
│   ├── multihistogram_features.csv  (generated, 128D)
│   ├── texture_features.csv         (generated, 272D)
│   ├── custom_features.csv          (generated, 209D)
│   └── my_dnn_features.csv          (generated by Extension 1, 512D)
├── results/
│   ├── comparison_0893.png          (Extension 1 output)
│   ├── comparison_0164.png          (Extension 1 output)
│   └── comparison_1072.png          (Extension 1 output)
├── build/                           (created during build)
└── report/
```

## Quick Reference: What Each CSV Contains

```
data/
├── olympus/                      # 1106 images
├── ResNet18_olym.csv            # Provided DNN (512 values × 1106 images)
├── baseline_features.csv        # Generated (147 values × 1106 images)
├── histogram_features.csv       # Generated (256 values × 1106 images)
├── multihistogram_features.csv  # Generated (128 values × 1106 images)
├── texture_features.csv         # Generated (272 values × 1106 images)
├── custom_features.csv          # Generated (209 values × 1106 images)
└── my_dnn_features.csv          # Generated by Extension 1 (512 values × 1106 images)
```

## Feature Types Summary

| Feature | Vector Size | Distance Metric |
|---------|------------|-----------------|
| Baseline | 147 (7×7×3) | Sum of Squared Differences |
| Histogram | 256 (16×16 rg chromaticity) | Histogram Intersection |
| Multi-histogram | 128 (2×8×8 top/bottom) | Weighted Histogram Intersection |
| Texture+Color | 272 (256 color + 16 texture) | Weighted Histogram Intersection |
| DNN | 512 (ResNet18 embedding) | Cosine Distance |
| Custom | 209 (1+16+192) + 512 DNN | Weighted Combination |

## Time Travel Days
None used.

## Acknowledgements
- Professor Bruce Maxwell, CS 5330 lecture notes and starter code
- Shapiro & Stockman, Computer Vision Chapter 8
- OpenCV documentation
- ResNet18 ONNX model from ONNX Model Zoo