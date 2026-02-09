/*
 * Name: Akash Shridhar Shetty, Skandhan Madhusudhana
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


/**
 * Histogram Intersection distance metric
 *
 * @param feature1 First histogram (normalized)
 * @param feature2 Second histogram (normalized)
 * @return Distance value in range [0, 1] (0 = identical, 1 = completely different)
 *
 * Implementation details:
 * What it does:
 *  1. Verify both histograms have same size
 *  2. For each bin:
 *      - Take minimum of the two values
 *      - Add to intersection sum
 *  3. Distance = 1 - intersection
 *
 * Mathematical formula:
 *  intersection = Σ min(H1[i], H2[i])
 *  distance = 1 - intersection
 *
 *  Where:
 *  - Σ means "sum of"
 *  - min(a,b) takes the smaller value
 *  - H1, H2 are normalized histograms (sum to 1.0)
 *
 * Visual example with 3-bin histograms:
 *
 *  H1 = [0.3, 0.2, 0.5]  (normalized, sums to 1.0)
 *  H2 = [0.3, 0.1, 0.6]  (normalized, sums to 1.0)
 *
 *  Intersection:
 *   Bin 0: min(0.3, 0.3) = 0.3
 *   Bin 1: min(0.2, 0.1) = 0.1
 *   Bin 2: min(0.5, 0.6) = 0.5
 *   Sum = 0.3 + 0.1 + 0.5 = 0.9
 *
 *  Distance = 1 - 0.9 = 0.1  ← Very similar!
 *
 * Real example with 256-bin histograms:
 *
 *  Target image histogram:  [0.012, 0.045, 0.003, ...]  (256 values)
 *  Database image A:        [0.011, 0.046, 0.004, ...]  (256 values)
 *  Database image B:        [0.001, 0.002, 0.098, ...]  (256 values)
 *
 *  distance(Target, Image A) = 0.05  ← Similar colors
 *  distance(Target, Image B) = 0.85  ← Very different colors
 *
 *  Result: Image A is more similar to Target ✓
 *
 * Properties:
 *  - Range: [0, 1]
 *  - Zero means identical histograms
 *  - One means completely different (no overlap)
 *  - Works well for comparing color distributions
 *
 * Why histogram intersection?
 *  - Designed specifically for histograms
 *  - More robust than SSD for color matching
 *  - Handles lighting variations better
 */
float distanceHistogramIntersection(const std::vector<float> &feature1,
                                     const std::vector<float> &feature2);


/**
 * Multi-histogram distance metric with weighted combination
 *
 * @param feature1 First multi-histogram (concatenated histograms)
 * @param feature2 Second multi-histogram (concatenated histograms)
 * @param numHistograms Number of histograms concatenated (default: 2)
 * @param weights Weight for each histogram (default: equal weights)
 * @return Distance value (lower = more similar)
 *
 * Implementation details:
 * What it does:
 *  1. Split concatenated feature vector into individual histograms
 *  2. Compute histogram intersection for each pair
 *  3. Combine distances using weighted average
 *
 * For 2 histograms with equal weights:
 *  feature1 = [top1_hist (64 values), bottom1_hist (64 values)]
 *  feature2 = [top2_hist (64 values), bottom2_hist (64 values)]
 *
 *  dist_top = histogram_intersection(top1, top2)
 *  dist_bottom = histogram_intersection(bottom1, bottom2)
 *  
 *  final_distance = 0.5 * dist_top + 0.5 * dist_bottom
 *
 * With weights [0.4, 0.6] (more weight on bottom):
 *  final_distance = 0.4 * dist_top + 0.6 * dist_bottom
 *
 * Example:
 *  Image A vs Image B:
 *    Top halves very similar: dist_top = 0.1
 *    Bottom halves different: dist_bottom = 0.8
 *    
 *  Equal weights:
 *    distance = 0.5 * 0.1 + 0.5 * 0.8 = 0.45
 *
 * Properties:
 *  - Range: [0, 1] (same as histogram intersection)
 *  - Zero means all histograms identical
 *  - Can weight different regions differently
 */
float distanceMultiHistogram(const std::vector<float> &feature1,
                              const std::vector<float> &feature2,
                              int numHistograms = 2,
                              const std::vector<float> &weights = {0.5f, 0.5f});


/**
 * Texture-Color distance metric
 * Handles two histograms of different sizes with weighted combination
 *
 * @param feature1 First feature vector [color_hist, texture_hist]
 * @param feature2 Second feature vector [color_hist, texture_hist]
 * @param colorSize Size of color histogram (default: 256)
 * @param textureSize Size of texture histogram (default: 16)
 * @param colorWeight Weight for color (default: 0.5)
 * @param textureWeight Weight for texture (default: 0.5)
 * @return Distance value (lower = more similar)
 *
 * Implementation details:
 * What it does:
 *  1. Split feature vector into color and texture parts
 *  2. Compute histogram intersection for color part
 *  3. Compute histogram intersection for texture part
 *  4. Combine with weighted sum: colorWeight * colorDist + textureWeight * textureDist
 *
 * Example:
 *  feature1 = [color_hist (256 values), texture_hist (16 values)]
 *  feature2 = [color_hist (256 values), texture_hist (16 values)]
 *
 *  colorDist = histogram_intersection(color1, color2)
 *  textureDist = histogram_intersection(texture1, texture2)
 *  
 *  final = 0.5 * colorDist + 0.5 * textureDist
 */
float distanceTextureColor(const std::vector<float> &feature1,
                            const std::vector<float> &feature2,
                            int colorSize = 256,
                            int textureSize = 16,
                            float colorWeight = 0.5f,
                            float textureWeight = 0.5f);



/**
 * Cosine distance metric
 * 
 * @param feature1 First feature vector
 * @param feature2 Second feature vector
 * @return Distance value in range [0, 2] (0 = identical, 2 = opposite)
 * 
 * Implementation details:
 * What it does:
 *  1. Normalize both vectors by their length (L2-norm)
 *  2. Compute dot product of normalized vectors (this is cosine similarity)
 *  3. Convert to distance: distance = 1 - cosine_similarity
 * 
 * Mathematical formula:
 *  cosine_similarity = (v1 · v2) / (||v1|| × ||v2||)
 *  cosine_distance = 1 - cosine_similarity
 * 
 *  Where:
 *  - v1 · v2 = dot product = Σ(v1[i] × v2[i])
 *  - ||v1|| = L2-norm = sqrt(Σ(v1[i]²))
 *  - ||v2|| = L2-norm = sqrt(Σ(v2[i]²))
 * 
 * Visual example with 3D vectors:
 *  v1 = [1.0, 2.0, 3.0]
 *  v2 = [1.0, 2.0, 3.0]  (same direction)
 *  
 *  ||v1|| = sqrt(1² + 2² + 3²) = sqrt(14) ≈ 3.742
 *  ||v2|| = sqrt(14) ≈ 3.742
 *  
 *  v1_norm = [1/3.742, 2/3.742, 3/3.742] = [0.267, 0.535, 0.802]
 *  v2_norm = [0.267, 0.535, 0.802]
 *  
 *  dot_product = 0.267×0.267 + 0.535×0.535 + 0.802×0.802 = 1.0
 *  distance = 1 - 1.0 = 0.0  ← Identical!
 * 
 * Real example with 512D vectors:
 *  Target image DNN vector:  [0.123, 0.456, ..., 0.234]  (512 values)
 *  Database image A vector:  [0.124, 0.455, ..., 0.235]  (512 values)
 *  Database image B vector:  [0.987, 0.123, ..., 0.876]  (512 values)
 *  
 *  distance(Target, Image A) = 0.05  ← Similar semantic content
 *  distance(Target, Image B) = 0.85  ← Very different semantic content
 * 
 * Properties:
 *  - Range: [0, 2] theoretically, but typically [0, 1] for real data
 *  - 0 = vectors point in same direction (identical/very similar)
 *  - 1 = vectors are perpendicular (uncorrelated)
 *  - 2 = vectors point in opposite directions (rare in practice)
 *  - Scale-invariant: only direction matters, not magnitude
 * 
 * Why cosine distance for DNN embeddings?
 *  - High-dimensional spaces: Euclidean distance suffers from curse of dimensionality
 *  - DNN embeddings encode semantic similarity as directional similarity
 *  - Magnitude of embedding vectors is less meaningful than direction
 *  - Works better than SSD for comparing learned representations
 */
float distanceCosine(const std::vector<float> &feature1,
                     const std::vector<float> &feature2);



/**
 * Custom distance metric for blue scene detection
 * Combines custom features with DNN embeddings
 *
 * @param customFeature1 First custom feature vector (209 values)
 * @param customFeature2 Second custom feature vector (209 values)
 * @param dnnFeature1 First DNN embedding (512 values)
 * @param dnnFeature2 Second DNN embedding (512 values)
 * @return Distance value (lower = more similar blue scenes)
 *
 * Feature breakdown:
 *  - Blue dominance: 1 value (compare using absolute difference)
 *  - Texture: 16 values (histogram intersection)
 *  - Spatial: 192 values (3 histograms of 64 bins each)
 *  - DNN: 512 values (cosine distance)
 *
 * Weights:
 *  - Blue dominance: 40% (most important for blue scenes)
 *  - Texture: 20% (smooth water/sky)
 *  - Spatial layout: 20% (where is the blue?)
 *  - DNN semantics: 20% (general similarity)
 */
float distanceCustomBlueScene(const std::vector<float> &customFeature1,
                               const std::vector<float> &customFeature2,
                               const std::vector<float> &dnnFeature1,
                               const std::vector<float> &dnnFeature2);
                               
#endif // DISTANCE_H