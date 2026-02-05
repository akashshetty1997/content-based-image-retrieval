/*
 * Name: Akash Shridhar Shetty
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

#endif // FEATURES_H
