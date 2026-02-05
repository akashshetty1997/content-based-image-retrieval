/*
 * Name: Akash Shridhar Shetty
 * Date: February 2025
 * File: features.cpp
 *
 * Purpose:
 * Implementation of feature extraction functions for content-based image retrieval.
 * Task 1 (Baseline): Extract center 7x7 square as a 147-element feature vector.
 */

#include "features.h"
#include <iostream>

/**
 * Extract baseline feature: center 7x7 square as feature vector
 * 
 * @param src Source image (cv::Mat, BGR color image)
 * @param feature Output feature vector (std::vector<float>)
 * @return 0 on success, -1 on error
 * 
 * Implementation details:
 * What it does:
 *  1. Validate image (not empty, large enough for 7x7)
 *  2. Calculate center position
 *  3. Loop through 7x7 region around center
 *  4. For each pixel, extract B,G,R values and add to feature vector
 *  5. Result: 7x7x3 = 147-element feature vector
 * 
 * Algorithm:
 *  centerRow = src.rows / 2
 *  centerCol = src.cols / 2
 *  
 *  For row from (centerRow - 3) to (centerRow + 3):  // 7 rows
 *      For col from (centerCol - 3) to (centerCol + 3):  // 7 cols
 *          pixel = src(row, col)
 *          feature.push_back(pixel.blue)
 *          feature.push_back(pixel.green)
 *          feature.push_back(pixel.red)
 * 
 * Example execution:
 *  Image size: 640x480
 *  Center: (240, 320)
 *  Extract rows: [237, 238, 239, 240, 241, 242, 243]
 *  Extract cols: [317, 318, 319, 320, 321, 322, 323]
 *  
 *  Pixel order (reading left-to-right, top-to-bottom):
 *  (237,317) → (237,318) → ... → (237,323)  ← Row 1
 *  (238,317) → (238,318) → ... → (238,323)  ← Row 2
 *  ...
 *  (243,317) → (243,318) → ... → (243,323)  ← Row 7
 *  
 *  Total: 49 pixels × 3 channels = 147 values
 */
int extractBaselineFeature(const cv::Mat &src, std::vector<float> &feature)
{
    // Clear any existing feature data
    feature.clear();
    
    // === Step 1: Validate input image ===
    
    // Check if image is empty
    if (src.empty())
    {
        std::cerr << "Error: Source image is empty" << std::endl;
        return -1;
    }
    
    // Check if image is large enough for 7x7 extraction
    if (src.rows < 7 || src.cols < 7)
    {
        std::cerr << "Error: Image too small for 7x7 extraction. Size: " 
                  << src.cols << "x" << src.rows << std::endl;
        return -1;
    }
    
    // Check if image is color (3 channels)
    if (src.channels() != 3)
    {
        std::cerr << "Error: Image must be 3-channel color (BGR)" << std::endl;
        return -1;
    }
    
    // === Step 2: Calculate center position ===
    
    int centerRow = src.rows / 2;  // Integer division (e.g., 480/2 = 240)
    int centerCol = src.cols / 2;  // Integer division (e.g., 640/2 = 320)
    
    // === Step 3: Define extraction region ===
    
    // We extract a 7x7 region centered at (centerRow, centerCol)
    // This means 3 pixels in each direction from center
    int startRow = centerRow - 3;  // Start 3 rows above center
    int endRow = centerRow + 3;    // End 3 rows below center (inclusive)
    
    int startCol = centerCol - 3;  // Start 3 cols left of center
    int endCol = centerCol + 3;    // End 3 cols right of center (inclusive)
    
    // Reserve space in feature vector for efficiency
    // We know we'll have exactly 7 * 7 * 3 = 147 values
    feature.reserve(147);
    
    // === Step 4: Extract pixel values ===
    
    // Loop through each row in the 7x7 region
    for (int row = startRow; row <= endRow; row++)
    {
        // Get pointer to current row for faster access
        const cv::Vec3b *rowPtr = src.ptr<cv::Vec3b>(row);
        
        // Loop through each column in the 7x7 region
        for (int col = startCol; col <= endCol; col++)
        {
            // Get the pixel at (row, col)
            // cv::Vec3b stores [Blue, Green, Red] in that order
            cv::Vec3b pixel = rowPtr[col];
            
            // Extract Blue, Green, Red channels
            unsigned char blue = pixel[0];
            unsigned char green = pixel[1];
            unsigned char red = pixel[2];
            
            // Add to feature vector in B,G,R order
            // Convert unsigned char (0-255) to float for consistency
            feature.push_back(static_cast<float>(blue));
            feature.push_back(static_cast<float>(green));
            feature.push_back(static_cast<float>(red));
        }
    }
    
    // === Step 5: Verify result ===
    
    // Double-check we got exactly 147 values
    if (feature.size() != 147)
    {
        std::cerr << "Error: Expected 147 features, got " << feature.size() << std::endl;
        return -1;
    }
    
    return 0;
}
