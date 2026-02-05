/*
 * Name: Akash Shridhar Shetty
 * Date: February 2025
 * File: distance.cpp
 *
 * Purpose:
 * Implementation of distance metric functions for content-based image retrieval.
 * Task 1 (Baseline): Sum of Squared Differences (SSD) for comparing feature vectors.
 */

#include "distance.h"
#include <iostream>
#include <cmath>

/**
 * Sum of Squared Differences (SSD) distance metric
 *
 * @param feature1 First feature vector
 * @param feature2 Second feature vector
 * @return Distance value (lower = more similar, 0 = identical)
 *
 * Implementation details:
 * What it does:
 *  1. Verify both vectors have the same length
 *  2. Initialize sum to 0
 *  3. Loop through all elements:
 *      - Compute difference
 *      - Square the difference
 *      - Add to sum
 *  4. Return the sum
 *
 * Algorithm:
 *  sum = 0
 *  for i = 0 to n-1:
 *      diff = feature1[i] - feature2[i]
 *      sum = sum + (diff * diff)
 *  return sum
 *
 * Example execution with small vectors:
 *  feature1 = [120, 130, 125]
 *  feature2 = [121, 131, 124]
 *
 *  Iteration 0:
 *    diff = 120 - 121 = -1
 *    squared = (-1) * (-1) = 1
 *    sum = 0 + 1 = 1
 *
 *  Iteration 1:
 *    diff = 130 - 131 = -1
 *    squared = (-1) * (-1) = 1
 *    sum = 1 + 1 = 2
 *
 *  Iteration 2:
 *    diff = 125 - 124 = 1
 *    squared = 1 * 1 = 1
 *    sum = 2 + 1 = 3
 *
 *  Final result: SSD = 3.0
 *
 * Real example with 147-element vectors:
 *  - Target image center: 147 values
 *  - Database image center: 147 values
 *  - Compute 147 squared differences
 *  - Sum them up
 *  - Typical range: 0 to ~1,000,000
 *
 * Special cases:
 *  - Identical vectors: SSD = 0 (all differences are 0)
 *  - Very similar: SSD = small value (e.g., 100-5000)
 *  - Different: SSD = large value (e.g., 50000-200000)
 *  - Very different: SSD = very large value (e.g., 500000+)
 */
float distanceSSD(const std::vector<float> &feature1,
                  const std::vector<float> &feature2)
{
    // === Step 1: Validate input ===

    // Check if both feature vectors have the same length
    if (feature1.size() != feature2.size())
    {
        std::cerr << "Error: Feature vectors have different sizes: "
                  << feature1.size() << " vs " << feature2.size() << std::endl;
        return -1.0f; // Return negative value to indicate error
    }

    // Check if vectors are empty
    if (feature1.empty())
    {
        std::cerr << "Error: Feature vectors are empty" << std::endl;
        return -1.0f;
    }

    // === Step 2: Initialize accumulator ===

    // This will hold the sum of all squared differences
    float sum = 0.0f;

    // === Step 3: Compute SSD ===

    // Get the size once (more efficient than calling .size() repeatedly)
    size_t n = feature1.size();

    // Loop through all elements in the feature vectors
    for (size_t i = 0; i < n; i++)
    {
        // Calculate the difference between corresponding elements
        float diff = feature1[i] - feature2[i];

        // Square the difference and add to sum
        // Using diff * diff instead of pow(diff, 2) for performance
        sum += diff * diff;
    }

    // === Step 4: Return result ===

    return sum;
}