/*
 * Name: Akash Shridhar Shetty, Skandhan Madhusudhana
 * Date: February 2025
 * File: features.h
 *
 * Purpose:
 * Header file for feature extraction functions used in content-based image retrieval.
 * For Task 1 (Baseline), we extract the center 7x7 square of pixels as a feature vector.
 */

#ifndef FEATURES_H
#define FEATURES_H

#include <opencv2/opencv.hpp>
#include <vector>

/**
 * Extract baseline feature: center 7x7 square as feature vector
 * 
 * @param src Source image (cv::Mat, BGR color image)
 * @param feature Output feature vector (std::vector<float>)
 * @return 0 on success, -1 on error
 * 
 * Implementation details:
 * What it does:
 *  1. Find the center of the image (centerRow, centerCol)
 *  2. Extract 7x7 square around center (3 pixels in each direction)
 *  3. For each of the 49 pixels:
 *      - Read Blue, Green, Red values
 *      - Add to feature vector in order: B,G,R,B,G,R,...
 *  4. Result: 7x7x3 = 147 values in feature vector
 * 
 * Visual representation:
 * 
 *     Image (e.g., 640x480)
 *     ┌─────────────────────┐
 *     │                     │
 *     │                     │
 *     │        7x7          │
 *     │       ┌───┐         │  ← We extract this tiny square
 *     │       │ X │         │     from the center
 *     │       └───┘         │
 *     │                     │
 *     │                     │
 *     └─────────────────────┘
 * 
 * Feature vector format (147 values):
 * [B₁,G₁,R₁, B₂,G₂,R₂, ..., B₄₉,G₄₉,R₄₉]
 * 
 * Where subscripts are pixel indices:
 *  1  2  3  4  5  6  7
 *  8  9 10 11 12 13 14
 * 15 16 17 18 19 20 21
 * 22 23 24 25 26 27 28  ← Row 4 = center row
 * 29 30 31 32 33 34 35
 * 36 37 38 39 40 41 42
 * 43 44 45 46 47 48 49
 *            ↑
 *         Pixel 25 = center pixel
 * 
 * Example:
 * Input: 640x480 image
 * Center: row=240, col=320
 * Extract: rows [237-243], cols [317-323]
 * Output: feature vector with 147 float values
 * 
 * Error handling:
 * - Returns -1 if image is too small (< 7x7)
 * - Returns -1 if image is empty
 */
int extractBaselineFeature(const cv::Mat &src, std::vector<float> &feature);

/**
 * Extract rg chromaticity histogram as feature vector
 * 
 * @param src Source image (cv::Mat, BGR color image)
 * @param feature Output feature vector (std::vector<float>)
 * @param binsPerChannel Number of bins for r and g (default: 16)
 * @return 0 on success, -1 on error
 * 
 * Implementation details:
 * What it does:
 *  1. For each pixel in image:
 *      - Compute rg chromaticity: r = R/(R+G+B), g = G/(R+G+B)
 *      - Determine which bin it falls into
 *      - Increment that bin's count
 *  2. Normalize histogram (divide by total pixels)
 *  3. Flatten 2D histogram into 1D feature vector
 * 
 * With binsPerChannel=16:
 *  - Total bins: 16 × 16 = 256
 *  - Feature vector size: 256 floats
 *  - Each value is normalized (percentage of pixels in that bin)
 * 
 * Example:
 *  Input: 640×480 image (307,200 pixels)
 *  Process: Compute r,g for each pixel, bin them
 *  Output: [0.012, 0.045, 0.003, ...] (256 normalized values)
 * 
 * Why rg chromaticity?
 *  - Lighting-invariant (bright red and dark red have same r,g)
 *  - Works better than RGB for image matching
 *  - Only uses 2 channels (r,g) because r+g+b=1, so b is redundant
 */
int extractRGChromaticityHistogram(const cv::Mat &src, 
                                    std::vector<float> &feature,
                                    int binsPerChannel = 16);


/**
 * Extract multi-histogram feature: top and bottom halves
 * 
 * @param src Source image (cv::Mat, BGR color image)
 * @param feature Output feature vector (std::vector<float>)
 * @param binsPerChannel Number of bins for r and g (default: 8)
 * @return 0 on success, -1 on error
 * 
 * Implementation details:
 * What it does:
 *  1. Split image into top and bottom halves
 *  2. Compute rg chromaticity histogram for top half
 *  3. Compute rg chromaticity histogram for bottom half
 *  4. Concatenate both histograms into single feature vector
 * 
 * With binsPerChannel=8:
 *  - Top histogram: 8 × 8 = 64 bins
 *  - Bottom histogram: 8 × 8 = 64 bins
 *  - Total feature vector: 64 + 64 = 128 values
 * 
 * Why split top/bottom?
 *  - Captures spatial layout (sky vs ground, water vs horizon)
 *  - Better than single histogram which loses spatial info
 *  - More discriminative for scene matching
 * 
 * Example:
 *  Beach scene: top = blue sky, bottom = yellow sand
 *  Mountain scene: top = blue sky, bottom = green trees
 *  → Different bottom histograms → won't match ✓
 * 
 * Feature vector format:
 *  [top_bin0, top_bin1, ..., top_bin63, bottom_bin0, ..., bottom_bin63]
 */
int extractMultiHistogram(const cv::Mat &src, 
                          std::vector<float> &feature,
                          int binsPerChannel = 8);


/**
 * Extract combined texture and color feature
 * 
 * @param src Source image (cv::Mat, BGR color image)
 * @param feature Output feature vector (std::vector<float>)
 * @param colorBins Number of bins for color histogram (default: 16)
 * @param textureBins Number of bins for texture histogram (default: 16)
 * @return 0 on success, -1 on error
 * 
 * Implementation details:
 * What it does:
 *  1. Extract whole-image rg chromaticity histogram (color)
 *  2. Compute Sobel gradient magnitude
 *  3. Extract histogram of gradient magnitudes (texture)
 *  4. Concatenate both histograms into single feature vector
 * 
 * With colorBins=16, textureBins=16:
 *  - Color histogram: 16 × 16 = 256 bins (rg chromaticity)
 *  - Texture histogram: 16 bins (gradient magnitude, single channel)
 *  - Total feature vector: 256 + 16 = 272 values
 * 
 * Why combine color and texture?
 *  - Color alone: matches similar colors regardless of patterns
 *  - Texture alone: matches similar patterns regardless of colors
 *  - Combined: matches images with similar colors AND patterns
 * 
 * Example:
 *  Brick wall (red, high texture) vs Red car (red, smooth)
 *  - Color only: would match (both red)
 *  - Texture only: would not match (different patterns)
 *  - Combined: would not match (different texture) ✓
 * 
 * Feature vector format:
 *  [color_bin0, ..., color_bin255, texture_bin0, ..., texture_bin15]
 */
int extractTextureColorFeature(const cv::Mat &src, 
                                std::vector<float> &feature,
                                int colorBins = 16,
                                int textureBins = 16);

// Helper function declarations (you already have these from Project 1)
int sobelX3x3(cv::Mat &src, cv::Mat &dst);
int sobelY3x3(cv::Mat &src, cv::Mat &dst);
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);

/**
 * Extract custom blue scene feature
 * 
 * @param src Source image (cv::Mat, BGR color image)
 * @param feature Output feature vector (std::vector<float>)
 * @return 0 on success, -1 on error
 * 
 * Custom feature for detecting blue/water scenes
 * 
 * Feature components:
 *  1. Blue dominance (1 value): percentage of pixels with blue hue
 *  2. Texture smoothness (16 values): gradient magnitude histogram
 *  3. Spatial layout (3 × 64 = 192 values): rg histograms for top/middle/bottom
 *  4. Total: 1 + 16 + 192 = 209 values
 * 
 * Note: DNN features will be loaded separately and combined at distance computation
 * 
 * Designed for: Finding images with blue water/sky scenes
 */
int extractCustomBlueSceneFeature(const cv::Mat &src, 
                                   std::vector<float> &feature);

#endif // FEATURES_H
