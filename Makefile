# Name: Akash Shridhar Shetty
# Date: February 2025
# File: Makefile
#
# Purpose:
# Build system for Content-Based Image Retrieval project

# ========================================
# Configuration
# ========================================

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra
OPENCV_CFLAGS = `pkg-config --cflags opencv4`
OPENCV_LIBS = `pkg-config --libs opencv4`
INCLUDES = -Iinclude

UTILS_SOURCES = src/utils.cpp src/features.cpp src/distance.cpp
UTILS_OBJECTS = $(UTILS_SOURCES:.cpp=.o)

EXTRACT_EXEC = extract_features
QUERY_EXEC = query
EMBEDDING_EXEC = compute_embeddings
GUI_EXEC = gui_query
COMPARE_EXEC = compare_embeddings

# ========================================
# Targets
# ========================================

all: $(EXTRACT_EXEC) $(QUERY_EXEC) $(EMBEDDING_EXEC) $(GUI_EXEC) $(COMPARE_EXEC)
	@echo "========================================="
	@echo "Build complete!"
	@echo "========================================="
	@echo "Executables created:"
	@echo "  - $(EXTRACT_EXEC)"
	@echo "  - $(QUERY_EXEC)"
	@echo "  - $(EMBEDDING_EXEC)"
	@echo "  - $(GUI_EXEC)"
	@echo "  - $(COMPARE_EXEC)"
	@echo "========================================="

$(EXTRACT_EXEC): src/main_extract_features.o $(UTILS_OBJECTS)
	@echo "Linking $(EXTRACT_EXEC)..."
	$(CXX) $(CXXFLAGS) -o $@ $^ $(OPENCV_LIBS)
	@echo "✓ $(EXTRACT_EXEC) created"

$(QUERY_EXEC): src/main_query.o $(UTILS_OBJECTS)
	@echo "Linking $(QUERY_EXEC)..."
	$(CXX) $(CXXFLAGS) -o $@ $^ $(OPENCV_LIBS)
	@echo "✓ $(QUERY_EXEC) created"

$(EMBEDDING_EXEC): src/embedding_extractor.o src/utils.o
	@echo "Linking $(EMBEDDING_EXEC)..."
	$(CXX) $(CXXFLAGS) -o $@ $^ $(OPENCV_LIBS)
	@echo "✓ $(EMBEDDING_EXEC) created"

$(GUI_EXEC): src/gui_query.o $(UTILS_OBJECTS)
	@echo "Linking $(GUI_EXEC)..."
	$(CXX) $(CXXFLAGS) -o $@ $^ $(OPENCV_LIBS)
	@echo "✓ $(GUI_EXEC) created"

$(COMPARE_EXEC): src/compare_embeddings.o src/utils.o src/distance.o
	@echo "Linking $(COMPARE_EXEC)..."
	$(CXX) $(CXXFLAGS) -o $@ $^ $(OPENCV_LIBS)
	@echo "✓ $(COMPARE_EXEC) created"

%.o: %.cpp
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OPENCV_CFLAGS) -c $< -o $@

clean:
	@echo "Cleaning build artifacts..."
	rm -f src/*.o $(EXTRACT_EXEC) $(QUERY_EXEC) $(EMBEDDING_EXEC) $(GUI_EXEC) $(COMPARE_EXEC)
	@echo "✓ Clean complete"

rebuild: clean all

setup:
	@echo "Setting up project directories..."
	mkdir -p data/olympus data results report
	@echo "✓ Directories created"

help:
	@echo "========================================="
	@echo "Content-Based Image Retrieval - Makefile"
	@echo "========================================="
	@echo ""
	@echo "Programs built:"
	@echo "  extract_features      - Extract features from images"
	@echo "  query                 - Query for similar images"
	@echo "  compute_embeddings    - Extract DNN embeddings (Extension)"
	@echo "  gui_query             - Visual GUI for retrieval (Extension)"
	@echo "  compare_embeddings    - Compare provided vs custom DNN (Extension)"
	@echo "========================================="

.PHONY: all clean rebuild setup help