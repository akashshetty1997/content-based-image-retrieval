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

# ========================================
# Targets
# ========================================

# Default target: build both programs
all: $(EXTRACT_EXEC) $(QUERY_EXEC)
	@echo "========================================="
	@echo "Build complete!"
	@echo "========================================="
	@echo "Executables created:"
	@echo "  - $(EXTRACT_EXEC)"
	@echo "  - $(QUERY_EXEC)"
	@echo ""
	@echo "Usage:"
	@echo "  ./$(EXTRACT_EXEC) <image_dir> <output_csv>"
	@echo "  ./$(QUERY_EXEC) <target_image> <feature_csv> <num_matches>"
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

# Compile .cpp files to .o files
%.o: %.cpp
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OPENCV_CFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f src/*.o $(EXTRACT_EXEC) $(QUERY_EXEC)
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
	@echo "  3. Run: ./extract_features data/olympus/ data/baseline_features.csv"

# Help target
help:
	@echo "========================================="
	@echo "Content-Based Image Retrieval - Makefile"
	@echo "========================================="
	@echo ""
	@echo "Available targets:"
	@echo "  make              - Build both programs (default)"
	@echo "  make all          - Build both programs"
	@echo "  make clean        - Remove compiled files"
	@echo "  make rebuild      - Clean and rebuild"
	@echo "  make setup        - Create project directories"
	@echo "  make help         - Show this help message"
	@echo ""
	@echo "Programs built:"
	@echo "  extract_features  - Extract features from images"
	@echo "  query             - Query for similar images"
	@echo "========================================="

# Phony targets (not actual files)
.PHONY: all clean rebuild setup help