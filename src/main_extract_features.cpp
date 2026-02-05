/*
 * Name: Akash Shridhar Shetty
 * Date: February 2025
 * File: main_extract_features.cpp
 *
 * Purpose:
 * Program 1: Extract features from all images in a directory and save to CSV file.
 * This is run ONCE to build the feature database, then can be reused for many queries.
 *
 * Usage:
 *   ./extract_features <image_directory> <output_csv>
 *
 * Example:
 *   ./extract_features data/olympus/ data/baseline_features.csv
 *
 * What it does:
 *   1. Read all image filenames from directory
 *   2. For each image:
 *      - Load the image
 *      - Extract baseline feature (center 7x7 square)
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

    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <image_directory> <output_csv>" << std::endl;
        std::cerr << "Example: " << argv[0] << " data/olympus/ data/baseline_features.csv" << std::endl;
        return -1;
    }

    std::string imageDir = argv[1];  // e.g., "data/olympus/"
    std::string outputCSV = argv[2]; // e.g., "data/baseline_features.csv"

    std::cout << "========================================" << std::endl;
    std::cout << "Feature Extraction Program" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Image directory: " << imageDir << std::endl;
    std::cout << "Output CSV: " << outputCSV << std::endl;
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

        // Extract baseline feature (center 7x7 square)
        std::vector<float> feature;
        if (extractBaselineFeature(image, feature) != 0)
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