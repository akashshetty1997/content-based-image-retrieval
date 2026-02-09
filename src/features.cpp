/*
 * Name: Akash Shridhar Shetty, Skandhan Madhusudhana
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

/**
 * Extract rg chromaticity histogram as feature vector
 */
int extractRGChromaticityHistogram(const cv::Mat &src, 
                                    std::vector<float> &feature,
                                    int binsPerChannel)
{
    // Clear any existing feature data
    feature.clear();
    
    // === Step 1: Validate input ===
    
    if (src.empty())
    {
        std::cerr << "Error: Source image is empty" << std::endl;
        return -1;
    }
    
    if (src.channels() != 3)
    {
        std::cerr << "Error: Image must be 3-channel color (BGR)" << std::endl;
        return -1;
    }
    
    // === Step 2: Initialize 2D histogram ===
    
    // Create 2D histogram with zeros
    // histogram[r_bin][g_bin] = count
    std::vector<std::vector<float>> histogram(binsPerChannel, 
                                               std::vector<float>(binsPerChannel, 0.0f));
    
    int totalPixels = 0;
    
    // === Step 3: Compute histogram ===
    
    // Loop through every pixel in the image
    for (int row = 0; row < src.rows; row++)
    {
        // Get pointer to current row for faster access
        const cv::Vec3b *rowPtr = src.ptr<cv::Vec3b>(row);
        
        for (int col = 0; col < src.cols; col++)
        {
            // Get BGR values
            cv::Vec3b pixel = rowPtr[col];
            float b = static_cast<float>(pixel[0]);
            float g = static_cast<float>(pixel[1]);
            float r = static_cast<float>(pixel[2]);
            
            // Compute sum for normalization
            float sum = r + g + b;
            
            // Skip black or near-black pixels to avoid division by zero
            if (sum < 1.0f)
                continue;
            
            // Compute rg chromaticity
            float r_chrom = r / sum;  // Range: [0, 1]
            float g_chrom = g / sum;  // Range: [0, 1]
            
            // Determine which bin this pixel falls into
            int r_bin = static_cast<int>(r_chrom * binsPerChannel);
            int g_bin = static_cast<int>(g_chrom * binsPerChannel);
            
            // Clamp to valid range (handle edge case where value = 1.0)
            if (r_bin >= binsPerChannel) r_bin = binsPerChannel - 1;
            if (g_bin >= binsPerChannel) g_bin = binsPerChannel - 1;
            
            // Increment bin count
            histogram[r_bin][g_bin] += 1.0f;
            totalPixels++;
        }
    }
    
    // === Step 4: Normalize histogram ===
    
    // Convert counts to percentages
    for (int r_bin = 0; r_bin < binsPerChannel; r_bin++)
    {
        for (int g_bin = 0; g_bin < binsPerChannel; g_bin++)
        {
            if (totalPixels > 0)
            {
                histogram[r_bin][g_bin] /= totalPixels;
            }
        }
    }
    
    // === Step 5: Flatten 2D histogram into 1D feature vector ===
    
    // Reserve space for efficiency
    feature.reserve(binsPerChannel * binsPerChannel);
    
    // Read histogram row by row
    for (int r_bin = 0; r_bin < binsPerChannel; r_bin++)
    {
        for (int g_bin = 0; g_bin < binsPerChannel; g_bin++)
        {
            feature.push_back(histogram[r_bin][g_bin]);
        }
    }
    
    // === Step 6: Verify result ===
    
    int expectedSize = binsPerChannel * binsPerChannel;
    if (feature.size() != static_cast<size_t>(expectedSize))
    {
        std::cerr << "Error: Expected " << expectedSize 
                  << " features, got " << feature.size() << std::endl;
        return -1;
    }
    
    return 0;
}

/**
 * Extract multi-histogram feature: top and bottom halves
 */
int extractMultiHistogram(const cv::Mat &src, 
                          std::vector<float> &feature,
                          int binsPerChannel)
{
    // Clear any existing feature data
    feature.clear();
    
    // === Step 1: Validate input ===
    
    if (src.empty())
    {
        std::cerr << "Error: Source image is empty" << std::endl;
        return -1;
    }
    
    if (src.channels() != 3)
    {
        std::cerr << "Error: Image must be 3-channel color (BGR)" << std::endl;
        return -1;
    }
    
    // === Step 2: Split image into top and bottom halves ===
    
    int midRow = src.rows / 2;  // Split at middle row
    
    // Define regions using OpenCV Rect (x, y, width, height)
    cv::Rect topRegion(0, 0, src.cols, midRow);           // Top half
    cv::Rect bottomRegion(0, midRow, src.cols, src.rows - midRow);  // Bottom half
    
    // Extract regions (these are views, not copies)
    cv::Mat topHalf = src(topRegion);
    cv::Mat bottomHalf = src(bottomRegion);
    
    // === Step 3: Compute histogram for top half ===
    
    std::vector<float> topHistogram;
    if (extractRGChromaticityHistogram(topHalf, topHistogram, binsPerChannel) != 0)
    {
        std::cerr << "Error: Failed to extract histogram from top half" << std::endl;
        return -1;
    }
    
    // === Step 4: Compute histogram for bottom half ===
    
    std::vector<float> bottomHistogram;
    if (extractRGChromaticityHistogram(bottomHalf, bottomHistogram, binsPerChannel) != 0)
    {
        std::cerr << "Error: Failed to extract histogram from bottom half" << std::endl;
        return -1;
    }
    
    // === Step 5: Concatenate histograms ===
    
    // Reserve space for efficiency
    int expectedSize = 2 * binsPerChannel * binsPerChannel;
    feature.reserve(expectedSize);
    
    // Add top histogram
    feature.insert(feature.end(), topHistogram.begin(), topHistogram.end());
    
    // Add bottom histogram
    feature.insert(feature.end(), bottomHistogram.begin(), bottomHistogram.end());
    
    // === Step 6: Verify result ===
    
    if (feature.size() != static_cast<size_t>(expectedSize))
    {
        std::cerr << "Error: Expected " << expectedSize 
                  << " features, got " << feature.size() << std::endl;
        return -1;
    }
    
    return 0;
}

/**
 * 3x3 Sobel X Filter - detects vertical edges (positive right)
 * @param src Source color image (cv::Mat)
 * @param dst Destination gradient image (cv::Mat, type CV_16SC3)
 * @return 0 on success
 * Implementation details:
 * What it does:
    - Implements 3x3 Sobel X filter as separable 1x3 filters
    - Horizontal: [-1, 0, 1] (derivative)
    - Vertical: [1, 2, 1] (smoothing)
    - Output is signed short (CV_16SC3) to handle negative values
    - Processes each color channel separately
    - Detects vertical edges (changes in horizontal direction)

   Example:
    - Apply [1, 2, 1] vertically first (smooth)
    - Then apply [-1, 0, 1] horizontally (gradient)
    - Result: strong response at vertical edges
 */
int sobelX3x3(cv::Mat &src, cv::Mat &dst)
{
    // Create temporary image to store intermediate vertical smoothing result
    cv::Mat temp;
    temp.create(src.size(), CV_16SC3);

    // Vertical smoothing kernel: [1, 2, 1]
    int vKernel[3] = {1, 2, 1};

    // Horizontal gradient kernel: [-1, 0, 1]
    int hKernel[3] = {-1, 0, 1};

    // Step 1: Apply vertical smoothing [1, 2, 1]
    for (int i = 1; i < src.rows - 1; i++)
    {
        cv::Vec3b *srcRowPrev = src.ptr<cv::Vec3b>(i - 1);
        cv::Vec3b *srcRowCurr = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *srcRowNext = src.ptr<cv::Vec3b>(i + 1);
        cv::Vec3s *tempRow = temp.ptr<cv::Vec3s>(i);

        for (int j = 0; j < src.cols; j++)
        {
            // Apply vertical kernel to each channel
            for (int c = 0; c < 3; c++)
            {
                short sum = vKernel[0] * srcRowPrev[j][c] +
                            vKernel[1] * srcRowCurr[j][c] +
                            vKernel[2] * srcRowNext[j][c];
                tempRow[j][c] = sum / 4; // Normalize by sum of kernel (4)
            }
        }
    }

    // Handle boundary rows (copy from source)
    for (int j = 0; j < src.cols; j++)
    {
        cv::Vec3b *srcTop = src.ptr<cv::Vec3b>(0);
        cv::Vec3b *srcBottom = src.ptr<cv::Vec3b>(src.rows - 1);
        cv::Vec3s *tempTop = temp.ptr<cv::Vec3s>(0);
        cv::Vec3s *tempBottom = temp.ptr<cv::Vec3s>(temp.rows - 1);

        for (int c = 0; c < 3; c++)
        {
            tempTop[j][c] = srcTop[j][c];
            tempBottom[j][c] = srcBottom[j][c];
        }
    }

    // Step 2: Apply horizontal gradient [-1, 0, 1]
    dst.create(src.size(), CV_16SC3);

    for (int i = 0; i < temp.rows; i++)
    {
        cv::Vec3s *tempRow = temp.ptr<cv::Vec3s>(i);
        cv::Vec3s *dstRow = dst.ptr<cv::Vec3s>(i);

        for (int j = 1; j < temp.cols - 1; j++)
        {
            // Apply horizontal gradient kernel to each channel
            for (int c = 0; c < 3; c++)
            {
                short gradient = hKernel[0] * tempRow[j - 1][c] +
                                 hKernel[1] * tempRow[j][c] +
                                 hKernel[2] * tempRow[j + 1][c];
                dstRow[j][c] = gradient;
            }
        }

        // Handle boundary columns
        for (int c = 0; c < 3; c++)
        {
            dstRow[0][c] = 0;
            dstRow[temp.cols - 1][c] = 0;
        }
    }

    return 0;
}

/**
 * 3x3 Sobel Y Filter - detects horizontal edges (positive up)
 * @param src Source color image (cv::Mat)
 * @param dst Destination gradient image (cv::Mat, type CV_16SC3)
 * @return 0 on success
 * Implementation details:
 * What it does:
    - Implements 3x3 Sobel Y filter as separable 1x3 filters
    - Horizontal: [1, 2, 1] (smoothing)
    - Vertical: [1, 0, -1] (derivative, positive up)
    - Output is signed short (CV_16SC3) to handle negative values
    - Processes each color channel separately
    - Detects horizontal edges (changes in vertical direction)

   Example:
    - Apply [1, 2, 1] horizontally first (smooth)
    - Then apply [1, 0, -1] vertically (gradient, positive up)
    - Result: strong response at horizontal edges
 */
int sobelY3x3(cv::Mat &src, cv::Mat &dst)
{
    // Create temporary image to store intermediate horizontal smoothing result
    cv::Mat temp;
    temp.create(src.size(), CV_16SC3);

    // Horizontal smoothing kernel: [1, 2, 1]
    int hKernel[3] = {1, 2, 1};

    // Vertical gradient kernel: [1, 0, -1] (positive up)
    int vKernel[3] = {1, 0, -1};

    // Step 1: Apply horizontal smoothing [1, 2, 1]
    for (int i = 0; i < src.rows; i++)
    {
        cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);
        cv::Vec3s *tempRow = temp.ptr<cv::Vec3s>(i);

        for (int j = 1; j < src.cols - 1; j++)
        {
            // Apply horizontal kernel to each channel
            for (int c = 0; c < 3; c++)
            {
                short sum = hKernel[0] * srcRow[j - 1][c] +
                            hKernel[1] * srcRow[j][c] +
                            hKernel[2] * srcRow[j + 1][c];
                tempRow[j][c] = sum / 4; // Normalize by sum of kernel (4)
            }
        }

        // Handle boundary columns
        for (int c = 0; c < 3; c++)
        {
            tempRow[0][c] = srcRow[0][c];
            tempRow[src.cols - 1][c] = srcRow[src.cols - 1][c];
        }
    }

    // Step 2: Apply vertical gradient [1, 0, -1]
    dst.create(src.size(), CV_16SC3);

    for (int i = 1; i < temp.rows - 1; i++)
    {
        cv::Vec3s *tempRowPrev = temp.ptr<cv::Vec3s>(i - 1);
        cv::Vec3s *tempRowCurr = temp.ptr<cv::Vec3s>(i);
        cv::Vec3s *tempRowNext = temp.ptr<cv::Vec3s>(i + 1);
        cv::Vec3s *dstRow = dst.ptr<cv::Vec3s>(i);

        for (int j = 0; j < temp.cols; j++)
        {
            // Apply vertical gradient kernel to each channel
            for (int c = 0; c < 3; c++)
            {
                short gradient = vKernel[0] * tempRowPrev[j][c] +
                                 vKernel[1] * tempRowCurr[j][c] +
                                 vKernel[2] * tempRowNext[j][c];
                dstRow[j][c] = gradient;
            }
        }
    }

    // Handle boundary rows
    for (int j = 0; j < temp.cols; j++)
    {
        cv::Vec3s *dstTop = dst.ptr<cv::Vec3s>(0);
        cv::Vec3s *dstBottom = dst.ptr<cv::Vec3s>(dst.rows - 1);

        for (int c = 0; c < 3; c++)
        {
            dstTop[j][c] = 0;
            dstBottom[j][c] = 0;
        }
    }

    return 0;
}

/**
 * Gradient Magnitude - computes magnitude from Sobel X and Y gradients
 * @param sx Sobel X gradient image (cv::Mat, type CV_16SC3)
 * @param sy Sobel Y gradient image (cv::Mat, type CV_16SC3)
 * @param dst Destination magnitude image (cv::Mat, type CV_8UC3)
 * @return 0 on success
 * Implementation details:
 * What it does:
    - Computes Euclidean distance: magnitude = sqrt(sx² + sy²)
    - Combines X and Y gradients to show overall edge strength
    - Processes each color channel independently
    - Output is unsigned char (CV_8UC3) suitable for display
    - Values are clamped to [0, 255] range

   Example:
    - If sx = 100 and sy = 100 for a pixel
    - magnitude = sqrt(100² + 100²) = sqrt(20000) ≈ 141
    - Result: shows strong edge regardless of direction
 */
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst)
{
    // Ensure both input images have the same size
    if (sx.size() != sy.size() || sx.type() != CV_16SC3 || sy.type() != CV_16SC3)
    {
        return -1;
    }

    // Create destination image as unsigned char
    dst.create(sx.size(), CV_8UC3);

    // Process each pixel
    for (int i = 0; i < sx.rows; i++)
    {
        cv::Vec3s *sxRow = sx.ptr<cv::Vec3s>(i);
        cv::Vec3s *syRow = sy.ptr<cv::Vec3s>(i);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j < sx.cols; j++)
        {
            // Compute magnitude for each channel
            for (int c = 0; c < 3; c++)
            {
                // Get gradient values
                short gx = sxRow[j][c];
                short gy = syRow[j][c];

                // Compute Euclidean magnitude: sqrt(gx² + gy²)
                float mag = sqrt(gx * gx + gy * gy);

                // Clamp to [0, 255] range and convert to unsigned char
                if (mag > 255)
                    mag = 255;

                dstRow[j][c] = (unsigned char)mag;
            }
        }
    }

    return 0;
}

/**
 * Extract histogram of gradient magnitudes (texture feature)
 * Helper function for extractTextureColorFeature
 */
int extractGradientMagnitudeHistogram(const cv::Mat &src, 
                                       std::vector<float> &feature,
                                       int bins)
{
    feature.clear();
    
    // === Step 1: Compute Sobel gradients ===
    
    cv::Mat sobelX, sobelY;
    
    if (sobelX3x3(const_cast<cv::Mat&>(src), sobelX) != 0)
    {
        std::cerr << "Error: Failed to compute Sobel X" << std::endl;
        return -1;
    }
    
    if (sobelY3x3(const_cast<cv::Mat&>(src), sobelY) != 0)
    {
        std::cerr << "Error: Failed to compute Sobel Y" << std::endl;
        return -1;
    }
    
    // === Step 2: Compute gradient magnitude ===
    
    cv::Mat mag;
    if (magnitude(sobelX, sobelY, mag) != 0)
    {
        std::cerr << "Error: Failed to compute magnitude" << std::endl;
        return -1;
    }
    
    // === Step 3: Convert to grayscale for histogram ===
    
    // Convert BGR magnitude to single-channel grayscale
    cv::Mat magGray;
    cv::cvtColor(mag, magGray, cv::COLOR_BGR2GRAY);
    
    // === Step 4: Build histogram of gradient magnitudes ===
    
    std::vector<float> histogram(bins, 0.0f);
    int totalPixels = 0;
    
    for (int i = 0; i < magGray.rows; i++)
    {
        unsigned char *row = magGray.ptr<unsigned char>(i);
        
        for (int j = 0; j < magGray.cols; j++)
        {
            // Get magnitude value (0-255)
            unsigned char value = row[j];
            
            // Determine which bin (0 to bins-1)
            int bin = (value * bins) / 256;
            
            // Clamp to valid range
            if (bin >= bins) bin = bins - 1;
            
            histogram[bin] += 1.0f;
            totalPixels++;
        }
    }
    
    // === Step 5: Normalize histogram ===
    
    for (int i = 0; i < bins; i++)
    {
        if (totalPixels > 0)
        {
            histogram[i] /= totalPixels;
        }
    }
    
    // === Step 6: Copy to feature vector ===
    
    feature = histogram;
    
    return 0;
}


/**
 * Extract combined texture and color feature
 */
int extractTextureColorFeature(const cv::Mat &src, 
                                std::vector<float> &feature,
                                int colorBins,
                                int textureBins)
{
    feature.clear();
    
    // === Step 1: Validate input ===
    
    if (src.empty())
    {
        std::cerr << "Error: Source image is empty" << std::endl;
        return -1;
    }
    
    if (src.channels() != 3)
    {
        std::cerr << "Error: Image must be 3-channel color (BGR)" << std::endl;
        return -1;
    }
    
    // === Step 2: Extract color histogram ===
    
    std::vector<float> colorFeature;
    if (extractRGChromaticityHistogram(src, colorFeature, colorBins) != 0)
    {
        std::cerr << "Error: Failed to extract color histogram" << std::endl;
        return -1;
    }
    
    // === Step 3: Extract texture histogram ===
    
    std::vector<float> textureFeature;
    if (extractGradientMagnitudeHistogram(src, textureFeature, textureBins) != 0)
    {
        std::cerr << "Error: Failed to extract texture histogram" << std::endl;
        return -1;
    }
    
    // === Step 4: Concatenate features ===
    
    int expectedSize = (colorBins * colorBins) + textureBins;
    feature.reserve(expectedSize);
    
    // Add color histogram
    feature.insert(feature.end(), colorFeature.begin(), colorFeature.end());
    
    // Add texture histogram
    feature.insert(feature.end(), textureFeature.begin(), textureFeature.end());
    
    // === Step 5: Verify result ===
    
    if (feature.size() != static_cast<size_t>(expectedSize))
    {
        std::cerr << "Error: Expected " << expectedSize 
                  << " features, got " << feature.size() << std::endl;
        return -1;
    }
    
    return 0;
}

/**
 * Helper: Calculate blue dominance in image
 */
float calculateBlueDominance(const cv::Mat &src)
{
    if (src.empty() || src.channels() != 3)
    {
        return 0.0f;
    }
    
    // Convert to HSV for better color detection
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
    
    int bluePixels = 0;
    int totalPixels = src.rows * src.cols;
    
    // Blue hue range in OpenCV HSV: approximately 100-130 (out of 180)
    // High saturation (> 30) to avoid grayish blues
    for (int i = 0; i < hsv.rows; i++)
    {
        cv::Vec3b *row = hsv.ptr<cv::Vec3b>(i);
        
        for (int j = 0; j < hsv.cols; j++)
        {
            unsigned char hue = row[j][0];
            unsigned char saturation = row[j][1];
            unsigned char value = row[j][2];
            
            // Check if pixel is blue
            // Hue: 100-130 (blue range in OpenCV)
            // Saturation: > 30 (not too gray)
            // Value: > 50 (not too dark)
            if (hue >= 100 && hue <= 130 && saturation > 30 && value > 50)
            {
                bluePixels++;
            }
        }
    }
    
    // Return percentage of blue pixels
    return static_cast<float>(bluePixels) / static_cast<float>(totalPixels);
}

/**
 * Extract custom blue scene feature
 */
int extractCustomBlueSceneFeature(const cv::Mat &src, 
                                   std::vector<float> &feature)
{
    feature.clear();
    
    // === Step 1: Validate input ===
    
    if (src.empty())
    {
        std::cerr << "Error: Source image is empty" << std::endl;
        return -1;
    }
    
    if (src.channels() != 3)
    {
        std::cerr << "Error: Image must be 3-channel color (BGR)" << std::endl;
        return -1;
    }
    
    // === Step 2: Component 1 - Blue dominance (1 value) ===
    
    float blueDominance = calculateBlueDominance(src);
    feature.push_back(blueDominance);
    
    // === Step 3: Component 2 - Texture smoothness (16 values) ===
    
    std::vector<float> textureFeature;
    if (extractGradientMagnitudeHistogram(src, textureFeature, 16) != 0)
    {
        std::cerr << "Error: Failed to extract texture histogram" << std::endl;
        return -1;
    }
    
    feature.insert(feature.end(), textureFeature.begin(), textureFeature.end());
    
    // === Step 4: Component 3 - Spatial layout (3 regions × 64 bins = 192 values) ===
    
    // Divide image into top (sky), middle (horizon), bottom (foreground/water)
    int regionHeight = src.rows / 3;
    
    // Top region (sky)
    cv::Rect topRect(0, 0, src.cols, regionHeight);
    cv::Mat topRegion = src(topRect);
    
    std::vector<float> topHist;
    if (extractRGChromaticityHistogram(topRegion, topHist, 8) != 0)
    {
        std::cerr << "Error: Failed to extract top region histogram" << std::endl;
        return -1;
    }
    feature.insert(feature.end(), topHist.begin(), topHist.end());
    
    // Middle region (horizon/transition)
    cv::Rect middleRect(0, regionHeight, src.cols, regionHeight);
    cv::Mat middleRegion = src(middleRect);
    
    std::vector<float> middleHist;
    if (extractRGChromaticityHistogram(middleRegion, middleHist, 8) != 0)
    {
        std::cerr << "Error: Failed to extract middle region histogram" << std::endl;
        return -1;
    }
    feature.insert(feature.end(), middleHist.begin(), middleHist.end());
    
    // Bottom region (foreground/water)
    cv::Rect bottomRect(0, 2 * regionHeight, src.cols, src.rows - 2 * regionHeight);
    cv::Mat bottomRegion = src(bottomRect);
    
    std::vector<float> bottomHist;
    if (extractRGChromaticityHistogram(bottomRegion, bottomHist, 8) != 0)
    {
        std::cerr << "Error: Failed to extract bottom region histogram" << std::endl;
        return -1;
    }
    feature.insert(feature.end(), bottomHist.begin(), bottomHist.end());
    
    // === Step 5: Verify result ===
    
    int expectedSize = 1 + 16 + (3 * 8 * 8); // 1 + 16 + 192 = 209
    if (feature.size() != static_cast<size_t>(expectedSize))
    {
        std::cerr << "Error: Expected " << expectedSize 
                  << " features, got " << feature.size() << std::endl;
        return -1;
    }
    
    return 0;
}