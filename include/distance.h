/*
 * Name: Akash Shridhar Shetty
 * Date: February 2025
 * File: distance.h
 *
 * Purpose:
 * Header file for distance metric functions used in content-based image retrieval.
 * For Task 1 (Baseline), we use Sum of Squared Differences (SSD) to compare features.
 */

#ifndef DISTANCE_H
#define DISTANCE_H

#include <vector>

/**
 * Sum of Squared Differences (SSD) distance metric
 *
 * @param feature1 First feature vector
 * @param feature2 Second feature vector
 * @return Distance value (lower = more similar, 0 = identical)
 *
 * Implementation details:
 * What it does:
 *  1. Check that both feature vectors have same length
 *  2. For each element i:
 *      - Calculate difference: diff = feature1[i] - feature2[i]
 *      - Square it: squared = diff * diff
 *      - Add to running total: sum += squared
 *  3. Return total sum
 *
 * Mathematical formula:
 *  SSD = Σ(feature1[i] - feature2[i])²
 *
 *  Where:
 *  - Σ means "sum of"
 *  - i goes from 0 to 146 (for 147-element feature vectors)
 *  - ² means "squared"
 *
 * Visual example with 3-element vectors:
 *
 *  feature1 = [120, 130, 125]
 *  feature2 = [121, 131, 124]
 *
 *  Differences:
 *   i=0: 120 - 121 = -1  →  (-1)² = 1
 *   i=1: 130 - 131 = -1  →  (-1)² = 1
 *   i=2: 125 - 124 =  1  →  ( 1)² = 1
 *
 *  SSD = 1 + 1 + 1 = 3  ← Very similar! (small distance)
 *
 * Real example with 147-element vectors:
 *
 *  Target image center:  [120, 130, 125, 128, ...]  (147 values)
 *  Image A center:       [121, 131, 124, 127, ...]  (147 values)
 *  Image B center:       [200, 50,  90,  180, ...]  (147 values)
 *
 *  SSD(Target, Image A) = small number  (e.g., 1234)  ← Similar colors
 *  SSD(Target, Image B) = large number  (e.g., 98765) ← Very different colors
 *
 *  Result: Image A is more similar to Target ✓
 *
 * Properties of SSD:
 *  - Always non-negative (≥ 0)
 *  - Zero means identical: SSD(X, X) = 0
 *  - Symmetric: SSD(A, B) = SSD(B, A)
 *  - Larger values = more different
 *  - Smaller values = more similar
 *
 * Why we square the differences:
 *  1. Makes all values positive (no negatives cancel out positives)
 *  2. Emphasizes large differences more than small ones
 *  3. Standard mathematical distance metric
 *
 * Example without squaring (wrong):
 *  diff1 = 120 - 121 = -1
 *  diff2 = 130 - 129 =  1
 *  sum = -1 + 1 = 0  ← Wrong! They cancel out
 *
 * Example with squaring (correct):
 *  diff1 = 120 - 121 = -1  →  1
 *  diff2 = 130 - 129 =  1  →  1
 *  sum = 1 + 1 = 2  ← Correct! Shows actual difference
 *
 * Error handling:
 *  - Returns -1.0f if feature vectors have different lengths
 *  - This shouldn't happen if code is correct, but we check anyway
 *
 * Performance note:
 *  - Simple linear scan: O(n) where n = feature length
 *  - For 147 elements: 147 subtractions, 147 multiplications, 147 additions
 *  - Very fast even for large databases (millions of comparisons per second)
 */
float distanceSSD(const std::vector<float> &feature1,
                  const std::vector<float> &feature2);

#endif // DISTANCE_H