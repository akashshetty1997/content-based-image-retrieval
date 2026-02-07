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

/**
 * Histogram Intersection distance metric
 */
float distanceHistogramIntersection(const std::vector<float> &feature1,
                                     const std::vector<float> &feature2)
{
    // === Step 1: Validate input ===
    
    if (feature1.size() != feature2.size())
    {
        std::cerr << "Error: Histogram feature vectors have different sizes: "
                  << feature1.size() << " vs " << feature2.size() << std::endl;
        return -1.0f;
    }
    
    if (feature1.empty())
    {
        std::cerr << "Error: Histogram feature vectors are empty" << std::endl;
        return -1.0f;
    }
    
    // === Step 2: Compute histogram intersection ===
    
    float intersection = 0.0f;
    
    // For each bin, take the minimum value
    for (size_t i = 0; i < feature1.size(); i++)
    {
        // Take minimum of the two histogram bins
        float minVal = std::min(feature1[i], feature2[i]);
        intersection += minVal;
    }
    
    // === Step 3: Convert intersection to distance ===
    
    // Intersection ranges from 0 (no overlap) to 1 (identical)
    // Distance ranges from 1 (no overlap) to 0 (identical)
    float distance = 1.0f - intersection;
    
    return distance;
}


/**
 * Multi-histogram distance metric with weighted combination
 */
float distanceMultiHistogram(const std::vector<float> &feature1,
                              const std::vector<float> &feature2,
                              int numHistograms,
                              const std::vector<float> &weights)
{
    // === Step 1: Validate input ===
    
    if (feature1.size() != feature2.size())
    {
        std::cerr << "Error: Multi-histogram feature vectors have different sizes: "
                  << feature1.size() << " vs " << feature2.size() << std::endl;
        return -1.0f;
    }
    
    if (feature1.empty())
    {
        std::cerr << "Error: Multi-histogram feature vectors are empty" << std::endl;
        return -1.0f;
    }
    
    // Check if weights vector has correct size
    if (weights.size() != static_cast<size_t>(numHistograms))
    {
        std::cerr << "Error: Number of weights (" << weights.size() 
                  << ") doesn't match number of histograms (" << numHistograms << ")" << std::endl;
        return -1.0f;
    }
    
    // Verify weights sum to 1.0 (approximately)
    float weightSum = 0.0f;
    for (float w : weights)
    {
        weightSum += w;
    }
    if (std::abs(weightSum - 1.0f) > 0.001f)
    {
        std::cerr << "Warning: Weights do not sum to 1.0 (sum = " << weightSum << ")" << std::endl;
    }
    
    // === Step 2: Calculate histogram size ===
    
    // Each histogram has equal size
    size_t histogramSize = feature1.size() / numHistograms;
    
    if (feature1.size() % numHistograms != 0)
    {
        std::cerr << "Error: Feature vector size (" << feature1.size() 
                  << ") not evenly divisible by number of histograms (" << numHistograms << ")" << std::endl;
        return -1.0f;
    }
    
    // === Step 3: Compute distance for each histogram pair ===
    
    float totalDistance = 0.0f;
    
    for (int h = 0; h < numHistograms; h++)
    {
        // Extract the h-th histogram from both feature vectors
        size_t startIdx = h * histogramSize;
        size_t endIdx = startIdx + histogramSize;
        
        // Create sub-vectors for this histogram
        std::vector<float> hist1(feature1.begin() + startIdx, feature1.begin() + endIdx);
        std::vector<float> hist2(feature2.begin() + startIdx, feature2.begin() + endIdx);
        
        // Compute histogram intersection distance for this pair
        float dist = distanceHistogramIntersection(hist1, hist2);
        
        if (dist < 0)
        {
            std::cerr << "Error: Failed to compute distance for histogram " << h << std::endl;
            return -1.0f;
        }
        
        // Add weighted distance to total
        totalDistance += weights[h] * dist;
    }
    
    return totalDistance;
}

/**
 * Texture-Color distance metric
 */
float distanceTextureColor(const std::vector<float> &feature1,
                            const std::vector<float> &feature2,
                            int colorSize,
                            int textureSize,
                            float colorWeight,
                            float textureWeight)
{
    // === Step 1: Validate input ===
    
    int expectedSize = colorSize + textureSize;
    
    if (feature1.size() != static_cast<size_t>(expectedSize) ||
        feature2.size() != static_cast<size_t>(expectedSize))
    {
        std::cerr << "Error: Feature vectors have unexpected size. "
                  << "Expected " << expectedSize 
                  << ", got " << feature1.size() << " and " << feature2.size() << std::endl;
        return -1.0f;
    }
    
    // Verify weights sum to 1.0 (approximately)
    float weightSum = colorWeight + textureWeight;
    if (std::abs(weightSum - 1.0f) > 0.001f)
    {
        std::cerr << "Warning: Weights do not sum to 1.0 (sum = " << weightSum << ")" << std::endl;
    }
    
    // === Step 2: Extract color histograms ===
    
    std::vector<float> color1(feature1.begin(), feature1.begin() + colorSize);
    std::vector<float> color2(feature2.begin(), feature2.begin() + colorSize);
    
    // === Step 3: Extract texture histograms ===
    
    std::vector<float> texture1(feature1.begin() + colorSize, feature1.end());
    std::vector<float> texture2(feature2.begin() + colorSize, feature2.end());
    
    // === Step 4: Compute color distance ===
    
    float colorDist = distanceHistogramIntersection(color1, color2);
    
    if (colorDist < 0)
    {
        std::cerr << "Error: Failed to compute color histogram distance" << std::endl;
        return -1.0f;
    }
    
    // === Step 5: Compute texture distance ===
    
    float textureDist = distanceHistogramIntersection(texture1, texture2);
    
    if (textureDist < 0)
    {
        std::cerr << "Error: Failed to compute texture histogram distance" << std::endl;
        return -1.0f;
    }
    
    // === Step 6: Combine with weighted sum ===
    
    float totalDistance = colorWeight * colorDist + textureWeight * textureDist;
    
    return totalDistance;
}


/**
 * Cosine distance metric
 */
float distanceCosine(const std::vector<float> &feature1,
                     const std::vector<float> &feature2)
{
    // === Step 1: Validate input ===
    
    if (feature1.size() != feature2.size())
    {
        std::cerr << "Error: Feature vectors have different sizes: "
                  << feature1.size() << " vs " << feature2.size() << std::endl;
        return -1.0f;
    }
    
    if (feature1.empty())
    {
        std::cerr << "Error: Feature vectors are empty" << std::endl;
        return -1.0f;
    }
    
    // === Step 2: Compute dot product ===
    
    float dotProduct = 0.0f;
    
    for (size_t i = 0; i < feature1.size(); i++)
    {
        dotProduct += feature1[i] * feature2[i];
    }
    
    // === Step 3: Compute L2-norms (magnitudes) ===
    
    float norm1 = 0.0f;
    float norm2 = 0.0f;
    
    for (size_t i = 0; i < feature1.size(); i++)
    {
        norm1 += feature1[i] * feature1[i];
        norm2 += feature2[i] * feature2[i];
    }
    
    norm1 = sqrt(norm1);
    norm2 = sqrt(norm2);
    
    // === Step 4: Handle zero-length vectors ===
    
    if (norm1 < 1e-10f || norm2 < 1e-10f)
    {
        std::cerr << "Warning: One or both vectors have near-zero length" << std::endl;
        return 1.0f;  // Maximum distance
    }
    
    // === Step 5: Compute cosine similarity ===
    
    float cosineSimilarity = dotProduct / (norm1 * norm2);
    
    // Clamp to [-1, 1] to handle floating-point errors
    if (cosineSimilarity > 1.0f) cosineSimilarity = 1.0f;
    if (cosineSimilarity < -1.0f) cosineSimilarity = -1.0f;
    
    // === Step 6: Convert to distance ===
    
    float cosineDistance = 1.0f - cosineSimilarity;
    
    return cosineDistance;
}


/**
 * Custom distance metric for blue scene detection
 */
float distanceCustomBlueScene(const std::vector<float> &customFeature1,
                               const std::vector<float> &customFeature2,
                               const std::vector<float> &dnnFeature1,
                               const std::vector<float> &dnnFeature2)
{
    // === Step 1: Validate inputs ===
    
    if (customFeature1.size() != 209 || customFeature2.size() != 209)
    {
        std::cerr << "Error: Custom features must be 209 values. Got: "
                  << customFeature1.size() << " and " << customFeature2.size() << std::endl;
        return -1.0f;
    }
    
    if (dnnFeature1.size() != 512 || dnnFeature2.size() != 512)
    {
        std::cerr << "Error: DNN features must be 512 values. Got: "
                  << dnnFeature1.size() << " and " << dnnFeature2.size() << std::endl;
        return -1.0f;
    }
    
    // === Step 2: Component 1 - Blue dominance distance (1 value) ===
    
    float blueDom1 = customFeature1[0];
    float blueDom2 = customFeature2[0];
    
    // Absolute difference, normalized to [0, 1]
    float blueDist = std::abs(blueDom1 - blueDom2);
    
    // === Step 3: Component 2 - Texture distance (16 values) ===
    
    std::vector<float> texture1(customFeature1.begin() + 1, 
                                 customFeature1.begin() + 17);
    std::vector<float> texture2(customFeature2.begin() + 1, 
                                 customFeature2.begin() + 17);
    
    float textureDist = distanceHistogramIntersection(texture1, texture2);
    
    if (textureDist < 0)
    {
        std::cerr << "Error: Failed to compute texture distance" << std::endl;
        return -1.0f;
    }
    
    // === Step 4: Component 3 - Spatial layout distance (192 values = 3×64) ===
    
    std::vector<float> spatial1(customFeature1.begin() + 17, customFeature1.end());
    std::vector<float> spatial2(customFeature2.begin() + 17, customFeature2.end());
    
    // Treat as 3 histograms of 64 bins each
    std::vector<float> spatialWeights = {0.33f, 0.34f, 0.33f}; // Equal weights for 3 regions
    float spatialDist = distanceMultiHistogram(spatial1, spatial2, 3, spatialWeights);
    
    if (spatialDist < 0)
    {
        std::cerr << "Error: Failed to compute spatial distance" << std::endl;
        return -1.0f;
    }
    
    // === Step 5: Component 4 - DNN semantic distance ===
    
    float dnnDist = distanceCosine(dnnFeature1, dnnFeature2);
    
    if (dnnDist < 0)
    {
        std::cerr << "Error: Failed to compute DNN distance" << std::endl;
        return -1.0f;
    }
    
    // === Step 6: Weighted combination ===
    
    float blueWeight = 0.4f;      // 40% - most important for blue scenes
    float textureWeight = 0.2f;   // 20% - smooth textures
    float spatialWeight = 0.2f;   // 20% - spatial layout
    float dnnWeight = 0.2f;       // 20% - semantic similarity
    
    float totalDistance = blueWeight * blueDist +
                         textureWeight * textureDist +
                         spatialWeight * spatialDist +
                         dnnWeight * dnnDist;
    
    return totalDistance;
}