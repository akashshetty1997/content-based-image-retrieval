/*
 * Name: Akash Shridhar Shetty, Skandhan Madhusudhana
 * Date: February 2025
 * File: main_extract_features.cpp
 *
 * Purpose:
 * Program 1: Extract features from all images in a directory and save to CSV file.
 * This is run ONCE to build the feature database, then can be reused for many queries.
 *
 * Usage:
 *   ./extract_features <image_directory> <output_csv> <feature_type>
 *
 * Example:
 *   ./extract_features data/olympus/ data/baseline_features.csv baseline
 *   ./extract_features data/olympus/ data/histogram_features.csv histogram
 *
 * What it does:
 *   1. Read all image filenames from directory
 *   2. For each image:
 *      - Load the image
 *      - Extract features based on feature type
 *      - Store in memory
 *   3. Write all features to CSV file
 *
 * Output CSV format:
 *   pic.0001.jpg,120.5,130.2,125.8,...,118.3
 *   pic.0002.jpg,115.1,128.9,130.5,...,122.7
 *   ...
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include "features.h"
#include "utils.h"

/**
 * Main function: Extract features from all images and save to CSV
 */
int main(int argc, char *argv[])
{
    // === Step 1: Parse command line arguments ===

    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <image_directory> <output_csv> <feature_type>" << std::endl;
        std::cerr << "\nFeature types:" << std::endl;
        std::cerr << "  baseline       - 7x7 center square (Task 1)" << std::endl;
        std::cerr << "  histogram      - rg chromaticity histogram (Task 2)" << std::endl;
        std::cerr << "  multihistogram - top/bottom histograms (Task 3)" << std::endl;
        std::cerr << "  texture        - color + texture histograms (Task 4)" << std::endl;
        std::cerr << "  dnn            - NOT NEEDED (features provided by assignment)" << std::endl;
        std::cerr << "  custom         - custom blue scene detector (Task 7)" << std::endl;
        std::cerr << "\nExamples:" << std::endl;
        std::cerr << "  " << argv[0] << " data/olympus/ data/baseline_features.csv baseline" << std::endl;
        std::cerr << "  " << argv[0] << " data/olympus/ data/histogram_features.csv histogram" << std::endl;
        std::cerr << "  " << argv[0] << " data/olympus/ data/multihistogram_features.csv multihistogram" << std::endl;
        std::cerr << "  " << argv[0] << " data/olympus/ data/texture_features.csv texture" << std::endl;
        return -1;
    }

    std::string imageDir = argv[1];     // e.g., "data/olympus/"
    std::string outputCSV = argv[2];    // e.g., "data/histogram_features.csv"
    std::string featureType = argv[3];  // e.g., "histogram"

    // Validate feature type
    if (featureType != "baseline" && featureType != "histogram" && 
        featureType != "multihistogram" && featureType != "texture" && featureType != "dnn" && featureType != "custom")
    {
        std::cerr << "Error: Invalid feature type: " << featureType << std::endl;
        std::cerr << "Valid types: baseline, histogram, multihistogram, texture, dnn, custom" << std::endl;
        return -1;
    }

    // Check if user tried to extract DNN features
    if (featureType == "dnn")
    {
        std::cerr << "\nError: DNN features are pre-computed by the assignment." << std::endl;
        std::cerr << "You should use the provided CSV file directly with the query program." << std::endl;
        std::cerr << "No need to run feature extraction for DNN embeddings." << std::endl;
        return -1;
    }

    std::cout << "========================================" << std::endl;
    std::cout << "Feature Extraction Program" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Image directory: " << imageDir << std::endl;
    std::cout << "Output CSV: " << outputCSV << std::endl;
    std::cout << "Feature type: " << featureType << std::endl;
    std::cout << "========================================\n"
              << std::endl;

    // === Step 2: Get all image filenames from directory ===

    std::vector<std::string> filenames;

    std::cout << "Reading image filenames from directory..." << std::endl;

    if (getImageFilenames(imageDir, filenames) != 0)
    {
        std::cerr << "Error: Failed to read image filenames" << std::endl;
        return -1;
    }

    if (filenames.empty())
    {
        std::cerr << "Error: No images found in directory" << std::endl;
        return -1;
    }

    std::cout << "Found " << filenames.size() << " images\n"
              << std::endl;

    // === Step 3: Extract features from each image ===

    std::vector<FeatureData> allFeatures;
    allFeatures.reserve(filenames.size()); // Reserve space for efficiency

    int successCount = 0;
    int failCount = 0;

    std::cout << "Extracting features from images..." << std::endl;
    std::cout << "Progress: 0/" << filenames.size() << std::flush;

    for (size_t i = 0; i < filenames.size(); i++)
    {
        const std::string &filename = filenames[i];

        // Construct full path to image
        std::string fullPath = imageDir;
        if (fullPath.back() != '/')
        {
            fullPath += '/';
        }
        fullPath += filename;

        // Load the image
        cv::Mat image = cv::imread(fullPath);

        // Check if image loaded successfully
        if (image.empty())
        {
            std::cerr << "\nWarning: Failed to load image: " << filename << std::endl;
            failCount++;
            continue;
        }

        // Extract features based on type
        std::vector<float> feature;
        int result;

        if (featureType == "baseline")
        {
            result = extractBaselineFeature(image, feature);
        }
        else if (featureType == "histogram")
        {
            result = extractRGChromaticityHistogram(image, feature);
        }
        else if (featureType == "multihistogram")
        {
            result = extractMultiHistogram(image, feature);
        }
        else if (featureType == "texture")
        {
            result = extractTextureColorFeature(image, feature);
        }
        else if (featureType == "custom")
        {
            result = extractCustomBlueSceneFeature(image, feature);
        }
        else
        {
            std::cerr << "\nError: Unknown feature type: " << featureType << std::endl;
            failCount++;
            continue;
        }
        
        if (result != 0)
        {
            std::cerr << "\nWarning: Failed to extract features from: " << filename << std::endl;
            failCount++;
            continue;
        }

        // Store filename and feature vector
        FeatureData data;
        data.filename = filename;
        data.feature = feature;
        allFeatures.push_back(data);

        successCount++;

        // Update progress every 50 images
        if ((i + 1) % 50 == 0 || (i + 1) == filenames.size())
        {
            std::cout << "\rProgress: " << (i + 1) << "/" << filenames.size() << std::flush;
        }
    }

    std::cout << "\n"
              << std::endl;

    // === Step 4: Report extraction results ===

    std::cout << "========================================" << std::endl;
    std::cout << "Extraction Summary:" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Total images found: " << filenames.size() << std::endl;
    std::cout << "Successfully extracted: " << successCount << std::endl;
    std::cout << "Failed: " << failCount << std::endl;
    if (successCount > 0)
    {
        std::cout << "Feature vector size: " << allFeatures[0].feature.size() << " values" << std::endl;
    }
    std::cout << "========================================\n"
              << std::endl;

    if (allFeatures.empty())
    {
        std::cerr << "Error: No features extracted successfully" << std::endl;
        return -1;
    }

    // === Step 5: Write features to CSV file ===

    std::cout << "Writing features to CSV file..." << std::endl;

    if (writeFeaturesToCSV(outputCSV, allFeatures) != 0)
    {
        std::cerr << "Error: Failed to write features to CSV" << std::endl;
        return -1;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Feature extraction completed successfully!" << std::endl;
    std::cout << "Feature database saved to: " << outputCSV << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}