# Content-Based Image Retrieval

## Authors
- Akash Shridhar Shetty
- Skandhan Madhusudhana

## Operating System and IDE
- macOS (M4 Mac Air)
- VS Code with terminal
- C++17 with OpenCV 4 (installed via Homebrew)

## Project Description
A content-based image retrieval (CBIR) system that finds similar images from a database of 1100+ images using various feature extraction methods and distance metrics. The system implements classic computer vision features (color histograms, texture analysis, spatial layout) and deep network embeddings (ResNet18) to compare and rank images by visual similarity. Two extensions were implemented: custom DNN embedding extraction using our own ResNet18 ONNX inference, and an interactive GUI for visual browsing of results.

## Building the Project

```bash
make clean
make
```

This builds four executables:
- `extract_features` — Extract features from all images and save to CSV
- `query` — Query the feature database to find similar images
- `compute_embeddings` — Extract custom DNN embeddings using ResNet18 (Extension)
- `gui_query` — Interactive GUI for image retrieval (Extension)

## Running the Executables

### Step 1: Extract Features (run once per feature type)

```bash
./extract_features data/olympus/ data/baseline_features.csv baseline
./extract_features data/olympus/ data/histogram_features.csv histogram
./extract_features data/olympus/ data/multihistogram_features.csv multihistogram
./extract_features data/olympus/ data/texture_features.csv texture
./extract_features data/olympus/ data/custom_features.csv custom
```

### Step 2: Query for Similar Images

```bash
# Task 1: Baseline (7x7 center, SSD)
./query data/olympus/pic.1016.jpg data/baseline_features.csv 3 baseline

# Task 2: Histogram (rg chromaticity, histogram intersection)
./query data/olympus/pic.0164.jpg data/histogram_features.csv 3 histogram

# Task 3: Multi-histogram (top/bottom halves, weighted intersection)
./query data/olympus/pic.0274.jpg data/multihistogram_features.csv 3 multihistogram

# Task 4: Texture + Color (Sobel magnitude + rg chromaticity)
./query data/olympus/pic.0535.jpg data/texture_features.csv 3 texture

# Task 5: DNN Embeddings (cosine distance)
./query data/olympus/pic.0893.jpg data/ResNet18_olym.csv 3 dnn
./query data/olympus/pic.0164.jpg data/ResNet18_olym.csv 3 dnn

# Task 7: Custom Blue Scene Detector (requires DNN CSV as 5th argument)
./query data/olympus/pic.0164.jpg data/custom_features.csv 5 custom data/ResNet18_olym.csv
```

## Extensions

### Extension 1: Custom DNN Embeddings

Computes ResNet18 embeddings by running our own ONNX model inference, rather than using the pre-computed CSV from the assignment. This allows comparison between provided and self-computed embeddings.

**Requirements:** Download `resnet18-v2-7.onnx` to `data/` directory.

```bash
# Compute embeddings
./compute_embeddings data/resnet18-v2-7.onnx data/olympus/ data/my_dnn_features.csv

# Query with custom embeddings
./query data/olympus/pic.0893.jpg data/my_dnn_features.csv 3 dnn

# Compare provided vs custom
./query data/olympus/pic.0893.jpg data/ResNet18_olym.csv 3 dnn
./query data/olympus/pic.0893.jpg data/my_dnn_features.csv 3 dnn
```

**Results:** Both embedding sets produce meaningful results with similar distance ranges (~0.15-0.22). For some queries (pic.0893) the top matches are identical but reordered. For others (pic.0164) the results differ, showing that minor preprocessing differences affect rankings when many candidates have similar distances.

### Extension 2: Interactive GUI

Visual interface for browsing image retrieval results. Displays target image alongside top N matches with distance values.

```bash
# Launch GUI with different feature types
./gui_query data/olympus/ data/histogram_features.csv 4 histogram
./gui_query data/olympus/ data/my_dnn_features.csv 4 dnn
./gui_query data/olympus/ data/baseline_features.csv 4 baseline
./gui_query data/olympus/ data/texture_features.csv 4 texture
```

**GUI Controls:**
- Click any match image to use it as the new target
- `1`-`6` keys to switch feature type (1=baseline, 2=histogram, 3=multihistogram, 4=texture, 5=dnn, 6=custom)
- `n`/`p` to browse next/previous image in database
- `q` or `ESC` to quit

## File Structure

```
content-based-image-retrieval/
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
│   ├── embedding_extractor.cpp    (Extension)
│   └── gui_query.cpp              (Extension)
├── data/
│   ├── olympus/                   (image database, 1106 images)
│   ├── ResNet18_olym.csv          (provided DNN features)
│   └── resnet18-v2-7.onnx        (ONNX model for Extension 1)
├── results/
└── report/
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
