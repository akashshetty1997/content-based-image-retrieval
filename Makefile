# Name: Akash Shridhar Shetty
# Date: February 2025
# File: Makefile
#
# Purpose:
# Build system for Content-Based Image Retrieval project
# Alternative to CMake - simpler for this project

# ========================================
# Configuration
# ========================================

# Compiler
CXX = g++

# C++ standard
CXXFLAGS = -std=c++17 -Wall -Wextra

# OpenCV flags (for M4 Mac with Homebrew)
# Adjust path if your OpenCV is installed elsewhere
OPENCV_CFLAGS = `pkg-config --cflags opencv4`
OPENCV_LIBS = `pkg-config --libs opencv4`

# Include directories
INCLUDES = -Iinclude

# Source files
UTILS_SOURCES = src/utils.cpp src/features.cpp src/distance.cpp
UTILS_OBJECTS = $(UTILS_SOURCES:.cpp=.o)

# Executables
EXTRACT_EXEC = extract_features
QUERY_EXEC = query
EMBEDDING_EXEC = compute_embeddings
GUI_EXEC = gui_query

# ========================================
# Targets
# ========================================

# Default target: build all programs
all: $(EXTRACT_EXEC) $(QUERY_EXEC) $(EMBEDDING_EXEC) $(GUI_EXEC)
	@echo "========================================="
	@echo "Build complete!"
	@echo "========================================="
	@echo "Executables created:"
	@echo "  - $(EXTRACT_EXEC)"
	@echo "  - $(QUERY_EXEC)"
	@echo "  - $(EMBEDDING_EXEC)"
	@echo "  - $(GUI_EXEC)"
	@echo ""
	@echo "Usage:"
	@echo "  ./$(EXTRACT_EXEC) <image_dir> <output_csv> <feature_type>"
	@echo "  ./$(QUERY_EXEC) <target_image> <feature_csv> <num_matches> <feature_type>"
	@echo "  ./$(EMBEDDING_EXEC) <model_path> <image_dir> <output_csv>"
	@echo "  ./$(GUI_EXEC) <image_dir> <feature_csv> <num_matches> <feature_type> [dnn_csv]"
	@echo "========================================="

# Build extract_features program
$(EXTRACT_EXEC): src/main_extract_features.o $(UTILS_OBJECTS)
	@echo "Linking $(EXTRACT_EXEC)..."
	$(CXX) $(CXXFLAGS) -o $@ $^ $(OPENCV_LIBS)
	@echo "✓ $(EXTRACT_EXEC) created"

# Build query program
$(QUERY_EXEC): src/main_query.o $(UTILS_OBJECTS)
	@echo "Linking $(QUERY_EXEC)..."
	$(CXX) $(CXXFLAGS) -o $@ $^ $(OPENCV_LIBS)
	@echo "✓ $(QUERY_EXEC) created"

# Build compute_embeddings program (Extension)
$(EMBEDDING_EXEC): src/embedding_extractor.o src/utils.o
	@echo "Linking $(EMBEDDING_EXEC)..."
	$(CXX) $(CXXFLAGS) -o $@ $^ $(OPENCV_LIBS)
	@echo "✓ $(EMBEDDING_EXEC) created"

# Build gui_query program (Extension)
$(GUI_EXEC): src/gui_query.o $(UTILS_OBJECTS)
	@echo "Linking $(GUI_EXEC)..."
	$(CXX) $(CXXFLAGS) -o $@ $^ $(OPENCV_LIBS)
	@echo "✓ $(GUI_EXEC) created"

# Compile .cpp files to .o files
%.o: %.cpp
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OPENCV_CFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f src/*.o $(EXTRACT_EXEC) $(QUERY_EXEC) $(EMBEDDING_EXEC) $(GUI_EXEC)
	@echo "✓ Clean complete"

# Clean and rebuild
rebuild: clean all

# Create data directories if they don't exist
setup:
	@echo "Setting up project directories..."
	mkdir -p data/olympus
	mkdir -p data
	mkdir -p results
	mkdir -p report
	@echo "✓ Directories created"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Download image database to data/olympus/"
	@echo "  2. Run: make"
	@echo "  3. Run: ./extract_features data/olympus/ data/baseline_features.csv baseline"

# Help target
help:
	@echo "========================================="
	@echo "Content-Based Image Retrieval - Makefile"
	@echo "========================================="
	@echo ""
	@echo "Available targets:"
	@echo "  make              - Build all programs (default)"
	@echo "  make all          - Build all programs"
	@echo "  make clean        - Remove compiled files"
	@echo "  make rebuild      - Clean and rebuild"
	@echo "  make setup        - Create project directories"
	@echo "  make help         - Show this help message"
	@echo ""
	@echo "Programs built:"
	@echo "  extract_features      - Extract features from images"
	@echo "  query                 - Query for similar images"
	@echo "  compute_embeddings    - Extract DNN embeddings (Extension)"
	@echo "  gui_query             - Visual GUI for retrieval (Extension)"
	@echo "========================================="

# Phony targets (not actual files)
.PHONY: all clean rebuild setup help