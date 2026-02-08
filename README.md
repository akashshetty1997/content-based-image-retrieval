#!/bin/bash
# Content-Based Image Retrieval - Complete Commands
# Author: Akash Shridhar Shetty
# Date: February 2025

# ============================================
# SETUP
# ============================================

# Navigate to project directory
cd content-based-image-retrieval

# Build
make clean
make

# ============================================
# TASK 1: BASELINE MATCHING
# ============================================

echo "=== TASK 1: BASELINE MATCHING ==="

# Extract features
./extract_features data/olympus/ data/baseline_features.csv baseline

# Query pic.1016.jpg (required test case)
./query data/olympus/pic.1016.jpg data/baseline_features.csv 5 baseline

# ============================================
# TASK 2: HISTOGRAM MATCHING
# ============================================

echo ""
echo "=== TASK 2: HISTOGRAM MATCHING ==="

# Extract features
./extract_features data/olympus/ data/histogram_features.csv histogram

# Query pic.0164.jpg (required test case)
./query data/olympus/pic.0164.jpg data/histogram_features.csv 5 histogram

# ============================================
# TASK 3: MULTI-HISTOGRAM MATCHING
# ============================================

echo ""
echo "=== TASK 3: MULTI-HISTOGRAM MATCHING ==="

# Extract features
./extract_features data/olympus/ data/multihistogram_features.csv multihistogram

# Query pic.0274.jpg (required test case)
./query data/olympus/pic.0274.jpg data/multihistogram_features.csv 5 multihistogram

# ============================================
# TASK 4: TEXTURE + COLOR MATCHING
# ============================================

echo ""
echo "=== TASK 4: TEXTURE + COLOR MATCHING ==="

# Extract features
./extract_features data/olympus/ data/texture_features.csv texture

# Query pic.0535.jpg (required test case)
./query data/olympus/pic.0535.jpg data/texture_features.csv 5 texture

# Compare Task 2 vs Task 4
echo ""
echo "--- Comparison: Histogram vs Texture+Color ---"
echo "Task 2 (Color only):"
./query data/olympus/pic.0535.jpg data/histogram_features.csv 5 histogram
echo ""
echo "Task 4 (Color + Texture):"
./query data/olympus/pic.0535.jpg data/texture_features.csv 5 texture

# ============================================
# TASK 5: DEEP NETWORK EMBEDDINGS
# ============================================

echo ""
echo "=== TASK 5: DEEP NETWORK EMBEDDINGS ==="

# Query with provided DNN features
./query data/olympus/pic.0893.jpg data/ResNet18_olym.csv 5 dnn
./query data/olympus/pic.0164.jpg data/ResNet18_olym.csv 5 dnn

# ============================================
# TASK 6: COMPARISON ANALYSIS
# ============================================

echo ""
echo "=== TASK 6: COMPARISON ANALYSIS ==="

echo "--- Comparing all methods on pic.1072.jpg ---"
echo "Baseline:"
./query data/olympus/pic.1072.jpg data/baseline_features.csv 5 baseline
echo ""
echo "Histogram:"
./query data/olympus/pic.1072.jpg data/histogram_features.csv 5 histogram
echo ""
echo "Multi-histogram:"
./query data/olympus/pic.1072.jpg data/multihistogram_features.csv 5 multihistogram
echo ""
echo "Texture:"
./query data/olympus/pic.1072.jpg data/texture_features.csv 5 texture
echo ""
echo "DNN:"
./query data/olympus/pic.1072.jpg data/ResNet18_olym.csv 5 dnn

echo ""
echo "--- Comparing all methods on pic.0948.jpg ---"
echo "Baseline:"
./query data/olympus/pic.0948.jpg data/baseline_features.csv 5 baseline
echo ""
echo "Histogram:"
./query data/olympus/pic.0948.jpg data/histogram_features.csv 5 histogram
echo ""
echo "Multi-histogram:"
./query data/olympus/pic.0948.jpg data/multihistogram_features.csv 5 multihistogram
echo ""
echo "Texture:"
./query data/olympus/pic.0948.jpg data/texture_features.csv 5 texture
echo ""
echo "DNN:"
./query data/olympus/pic.0948.jpg data/ResNet18_olym.csv 5 dnn

# ============================================
# TASK 7: CUSTOM DESIGN (Blue Scene Detector)
# ============================================

echo ""
echo "=== TASK 7: CUSTOM DESIGN (Blue Scene Detector) ==="

# Extract custom features
./extract_features data/olympus/ data/custom_features.csv custom

# Query blue/water scenes
./query data/olympus/pic.0164.jpg data/custom_features.csv 10 custom data/ResNet18_olym.csv
./query data/olympus/pic.0080.jpg data/custom_features.csv 10 custom data/ResNet18_olym.csv

# Compare custom vs other methods
echo ""
echo "--- Comparison: Custom vs Other Methods ---"
echo "Histogram:"
./query data/olympus/pic.0164.jpg data/histogram_features.csv 5 histogram
echo ""
echo "DNN:"
./query data/olympus/pic.0164.jpg data/ResNet18_olym.csv 5 dnn
echo ""
echo "Custom:"
./query data/olympus/pic.0164.jpg data/custom_features.csv 5 custom data/ResNet18_olym.csv

# ============================================
# EXTENSION: CUSTOM DNN EMBEDDINGS
# ============================================

echo ""
echo "=== EXTENSION: CUSTOM DNN EMBEDDINGS ==="

# Compute your own embeddings
./compute_embeddings data/resnet18-v2-7.onnx data/olympus/ data/my_dnn_features.csv

# Compare provided vs custom embeddings
echo ""
echo "--- Provided DNN vs Custom DNN on pic.0893.jpg ---"
echo "Provided:"
./query data/olympus/pic.0893.jpg data/ResNet18_olym.csv 3 dnn
echo ""
echo "Custom:"
./query data/olympus/pic.0893.jpg data/my_dnn_features.csv 3 dnn

echo ""
echo "--- Provided DNN vs Custom DNN on pic.0164.jpg ---"
echo "Provided:"
./query data/olympus/pic.0164.jpg data/ResNet18_olym.csv 3 dnn
echo ""
echo "Custom:"
./query data/olympus/pic.0164.jpg data/my_dnn_features.csv 3 dnn

echo ""
echo "--- Provided DNN vs Custom DNN on pic.1072.jpg ---"
echo "Provided:"
./query data/olympus/pic.1072.jpg data/ResNet18_olym.csv 3 dnn
echo ""
echo "Custom:"
./query data/olympus/pic.1072.jpg data/my_dnn_features.csv 3 dnn

# ============================================
# EXTENSION: GUI
# ============================================

echo ""
echo "=== EXTENSION: GUI ==="
echo "Run interactively:"
echo "  ./gui_query data/olympus/ data/histogram_features.csv 4 histogram"
echo "  ./gui_query data/olympus/ data/my_dnn_features.csv 4 dnn"
echo "  ./gui_query data/olympus/ data/baseline_features.csv 4 baseline"

# ============================================
# VERIFICATION
# ============================================

echo ""
echo "=== VERIFICATION ==="

echo "Feature vector sizes:"
echo -n "Baseline: "
head -n 1 data/baseline_features.csv | awk -F',' '{print NF-1 " values"}'
echo -n "Histogram: "
head -n 1 data/histogram_features.csv | awk -F',' '{print NF-1 " values"}'
echo -n "Multi-histogram: "
head -n 1 data/multihistogram_features.csv | awk -F',' '{print NF-1 " values"}'
echo -n "Texture: "
head -n 1 data/texture_features.csv | awk -F',' '{print NF-1 " values"}'
echo -n "Custom: "
head -n 1 data/custom_features.csv | awk -F',' '{print NF-1 " values"}'
echo -n "DNN (provided): "
head -n 1 data/ResNet18_olym.csv | awk -F',' '{print NF-1 " values"}'
echo -n "DNN (custom): "
head -n 1 data/my_dnn_features.csv | awk -F',' '{print NF-1 " values"}'

echo ""
echo "Expected sizes:"
echo "  Baseline: 147 (7x7x3)"
echo "  Histogram: 256 (16x16)"
echo "  Multi-histogram: 128 (2x8x8)"
echo "  Texture: 272 (256 color + 16 texture)"
echo "  DNN: 512 (ResNet18)"
echo "  Custom: 209 (1 + 16 + 192)"

echo ""
echo "=== ALL TASKS COMPLETE ==="