/*
 * Name: Akash Shridhar Shetty
 * Date: February 2025
 * File: main_query.cpp
 *
 * Purpose:
 * Program 2: Query the feature database to find similar images.
 * This is run MANY times with different target images to find matches.
 * 
 * Usage:
 *   ./query <target_image> <feature_csv> <num_matches>
 * 
 * Example:
 *   ./query data/olympus/pic.1016.jpg data/baseline_features.csv 3
 * 
 * What it does:
 *   1. Load target image and extract its features
 *   2. Load all features from CSV database
 *   3. Compare target to every database image using SSD distance
 *   4. Sort results by distance (ascending)
 *   5. Display top N matches
 * 
 * Expected output for pic.1016.jpg:
 *   Top 3 matches:
 *   1. pic.1016.jpg  (distance: 0.000000)
 *   2. pic.0986.jpg  (distance: 1234.567890)
 *   3. pic.0641.jpg  (distance: 2345.678901)
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include "features.h"
#include "distance.h"
#include "utils.h"

/**
 * Main function: Query feature database to find similar images
 */
int main(int argc, char *argv[])
{
    // === Step 1: Parse command line arguments ===
    
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <target_image> <feature_csv> <num_matches>" << std::endl;
        std::cerr << "Example: " << argv[0] << " data/olympus/pic.1016.jpg data/baseline_features.csv 3" << std::endl;
        return -1;
    }
    
    std::string targetImagePath = argv[1];  // e.g., "data/olympus/pic.1016.jpg"
    std::string featureCSV = argv[2];       // e.g., "data/baseline_features.csv"
    int numMatches = std::stoi(argv[3]);    // e.g., 3
    
    std::cout << "========================================" << std::endl;
    std::cout << "Image Retrieval Query" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Target image: " << targetImagePath << std::endl;
    std::cout << "Feature database: " << featureCSV << std::endl;
    std::cout << "Number of matches: " << numMatches << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // === Step 2: Load and extract features from target image ===
    
    std::cout << "Loading target image..." << std::endl;
    
    cv::Mat targetImage = cv::imread(targetImagePath);
    
    if (targetImage.empty())
    {
        std::cerr << "Error: Failed to load target image: " << targetImagePath << std::endl;
        return -1;
    }
    
    std::cout << "Target image size: " << targetImage.cols << "x" << targetImage.rows << std::endl;
    
    std::cout << "Extracting features from target image..." << std::endl;
    
    std::vector<float> targetFeature;
    if (extractBaselineFeature(targetImage, targetFeature) != 0)
    {
        std::cerr << "Error: Failed to extract features from target image" << std::endl;
        return -1;
    }
    
    std::cout << "Target feature size: " << targetFeature.size() << " values" << std::endl;
    std::cout << std::endl;
    
    // === Step 3: Load feature database from CSV ===
    
    std::cout << "Loading feature database from CSV..." << std::endl;
    
    std::vector<FeatureData> database;
    if (readFeaturesFromCSV(featureCSV, database) != 0)
    {
        std::cerr << "Error: Failed to load feature database" << std::endl;
        return -1;
    }
    
    if (database.empty())
    {
        std::cerr << "Error: Feature database is empty" << std::endl;
        return -1;
    }
    
    std::cout << "Loaded " << database.size() << " feature vectors from database" << std::endl;
    std::cout << std::endl;
    
    // === Step 4: Compare target to all database images ===
    
    std::cout << "Computing distances to all database images..." << std::endl;
    
    std::vector<MatchResult> results;
    results.reserve(database.size());  // Reserve space for efficiency
    
    for (size_t i = 0; i < database.size(); i++)
    {
        // Compute SSD distance between target and database image
        float dist = distanceSSD(targetFeature, database[i].feature);
        
        // Check for error (negative distance indicates error)
        if (dist < 0)
        {
            std::cerr << "Warning: Error computing distance for " << database[i].filename << std::endl;
            continue;
        }
        
        // Store result
        MatchResult match;
        match.filename = database[i].filename;
        match.distance = dist;
        results.push_back(match);
        
        // Show progress for large databases
        if ((i + 1) % 100 == 0)
        {
            std::cout << "\rProgress: " << (i + 1) << "/" << database.size() << std::flush;
        }
    }
    
    if ((database.size() >= 100))
    {
        std::cout << "\rProgress: " << database.size() << "/" << database.size() << std::endl;
    }
    
    std::cout << "Computed " << results.size() << " distances" << std::endl;
    std::cout << std::endl;
    
    // === Step 5: Sort results by distance (ascending) ===
    
    std::cout << "Sorting results by distance..." << std::endl;
    
    // Sort using the comparison operator defined in MatchResult
    // This sorts in ascending order (smallest distance first)
    std::sort(results.begin(), results.end());
    
    std::cout << "Sorting complete" << std::endl;
    
    // === Step 6: Display top N matches ===
    
    printTopMatches(results, numMatches);
    
    // === Step 7: Verify expected results for pic.1016.jpg ===
    
    // Extract just the filename from the full path for comparison
    std::string targetFilename = targetImagePath;
    size_t lastSlash = targetFilename.find_last_of("/\\");
    if (lastSlash != std::string::npos)
    {
        targetFilename = targetFilename.substr(lastSlash + 1);
    }
    
    // Check if this is pic.1016.jpg and verify expected matches
    if (targetFilename == "pic.1016.jpg" && results.size() >= 4)
    {
        std::cout << "Verification for pic.1016.jpg:" << std::endl;
        std::cout << "Expected top 4 matches: pic.1016.jpg, pic.0986.jpg, pic.0641.jpg, pic.0547.jpg" << std::endl;
        std::cout << "Actual top 4 matches: " 
                  << results[0].filename << ", "
                  << results[1].filename << ", "
                  << results[2].filename << ", "
                  << results[3].filename << std::endl;
        
        // Check if matches are correct
        bool match1 = (results[0].filename == "pic.1016.jpg");  // Should be 0 distance
        bool match2 = (results[1].filename == "pic.0986.jpg");
        bool match3 = (results[2].filename == "pic.0641.jpg");
        bool match4 = (results[3].filename == "pic.0547.jpg");
        
        if (match1 && match2 && match3 && match4)
        {
            std::cout << "✓ Results match expected output!" << std::endl;
        }
        else
        {
            std::cout << "✗ Results do not match expected output" << std::endl;
        }
        std::cout << std::endl;
    }
    
    // === Step 8: Success message ===
    
    std::cout << "========================================" << std::endl;
    std::cout << "Query completed successfully!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}