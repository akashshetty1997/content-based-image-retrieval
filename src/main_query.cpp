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
 *   ./query <target_image> <feature_csv> <num_matches> <feature_type> [dnn_csv]
 * 
 * Example:
 *   ./query data/olympus/pic.1016.jpg data/baseline_features.csv 3 baseline
 *   ./query data/olympus/pic.0164.jpg data/histogram_features.csv 3 histogram
 *   ./query data/olympus/pic.0274.jpg data/multihistogram_features.csv 3 multihistogram
 *   ./query data/olympus/pic.0535.jpg data/texture_features.csv 3 texture
 *   ./query data/olympus/pic.0893.jpg data/dnn_features.csv 3 dnn
 *   ./query data/olympus/pic.0164.jpg data/custom_features.csv 5 custom data/dnn_features.csv
 * 
 * What it does:
 *   1. Load target image and extract its features (or load from CSV for DNN/custom)
 *   2. Load all features from CSV database
 *   3. Compare target to every database image using appropriate distance metric
 *   4. Sort results by distance (ascending)
 *   5. Display top N matches
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
    
    // Custom feature type requires an extra argument (DNN CSV)
    bool validArgCount = (argc == 5) || (argc == 6);
    
    if (!validArgCount)
    {
        std::cerr << "Usage: " << argv[0] << " <target_image> <feature_csv> <num_matches> <feature_type> [dnn_csv]" << std::endl;
        std::cerr << "\nFeature types:" << std::endl;
        std::cerr << "  baseline       - uses SSD distance (Task 1)" << std::endl;
        std::cerr << "  histogram      - uses histogram intersection (Task 2)" << std::endl;
        std::cerr << "  multihistogram - uses weighted histogram intersection (Task 3)" << std::endl;
        std::cerr << "  texture        - uses color + texture histograms (Task 4)" << std::endl;
        std::cerr << "  dnn            - uses cosine distance (Task 5)" << std::endl;
        std::cerr << "  custom         - custom blue scene detector with DNN (Task 7)" << std::endl;
        std::cerr << "\nExamples:" << std::endl;
        std::cerr << "  " << argv[0] << " data/olympus/pic.1016.jpg data/baseline_features.csv 3 baseline" << std::endl;
        std::cerr << "  " << argv[0] << " data/olympus/pic.0164.jpg data/histogram_features.csv 3 histogram" << std::endl;
        std::cerr << "  " << argv[0] << " data/olympus/pic.0274.jpg data/multihistogram_features.csv 3 multihistogram" << std::endl;
        std::cerr << "  " << argv[0] << " data/olympus/pic.0535.jpg data/texture_features.csv 3 texture" << std::endl;
        std::cerr << "  " << argv[0] << " data/olympus/pic.0893.jpg data/dnn_features.csv 3 dnn" << std::endl;
        std::cerr << "\nNote: For 'custom' feature type, provide DNN CSV as 5th argument:" << std::endl;
        std::cerr << "  " << argv[0] << " data/olympus/pic.0164.jpg data/custom_features.csv 5 custom data/dnn_features.csv" << std::endl;
        return -1;
    }
    
    std::string targetImagePath = argv[1];  // e.g., "data/olympus/pic.0893.jpg"
    std::string featureCSV = argv[2];       // e.g., "data/custom_features.csv"
    int numMatches = std::stoi(argv[3]);    // e.g., 5
    std::string featureType = argv[4];      // e.g., "custom"
    
    std::string dnnCSV = "";
    if (argc == 6)
    {
        dnnCSV = argv[5];  // e.g., "data/dnn_features.csv"
    }
    
    // Validate feature type
    if (featureType != "baseline" && featureType != "histogram" && 
        featureType != "multihistogram" && featureType != "texture" && 
        featureType != "dnn" && featureType != "custom")
    {
        std::cerr << "Error: Invalid feature type: " << featureType << std::endl;
        std::cerr << "Valid types: baseline, histogram, multihistogram, texture, dnn, custom" << std::endl;
        return -1;
    }
    
    // Custom feature type requires DNN CSV
    if (featureType == "custom" && dnnCSV.empty())
    {
        std::cerr << "Error: Custom feature type requires DNN CSV file as 5th argument" << std::endl;
        std::cerr << "Example: " << argv[0] << " <target> <custom_csv> <num> custom <dnn_csv>" << std::endl;
        return -1;
    }
    
    std::cout << "========================================" << std::endl;
    std::cout << "Image Retrieval Query" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Target image: " << targetImagePath << std::endl;
    std::cout << "Feature database: " << featureCSV << std::endl;
    std::cout << "Number of matches: " << numMatches << std::endl;
    std::cout << "Feature type: " << featureType << std::endl;
    if (!dnnCSV.empty())
    {
        std::cout << "DNN database: " << dnnCSV << std::endl;
    }
    std::cout << "========================================\n" << std::endl;
    
    // Extract just the filename from the full path for comparison
    std::string targetFilename = targetImagePath;
    size_t lastSlash = targetFilename.find_last_of("/\\");
    if (lastSlash != std::string::npos)
    {
        targetFilename = targetFilename.substr(lastSlash + 1);
    }
    
    // === Step 2: Load and extract features from target image ===
    
    cv::Mat targetImage;
    std::vector<float> targetFeature;
    std::vector<float> targetDNNFeature;  // For custom feature type
    
    // For non-DNN and non-custom features, we need to load the image
    if (featureType != "dnn" && featureType != "custom")
    {
        std::cout << "Loading target image..." << std::endl;
        
        targetImage = cv::imread(targetImagePath);
        
        if (targetImage.empty())
        {
            std::cerr << "Error: Failed to load target image: " << targetImagePath << std::endl;
            return -1;
        }
        
        std::cout << "Target image size: " << targetImage.cols << "x" << targetImage.rows << std::endl;
        
        std::cout << "Extracting features from target image..." << std::endl;
        
        int result;
        
        if (featureType == "baseline")
        {
            result = extractBaselineFeature(targetImage, targetFeature);
        }
        else if (featureType == "histogram")
        {
            result = extractRGChromaticityHistogram(targetImage, targetFeature);
        }
        else if (featureType == "multihistogram")
        {
            result = extractMultiHistogram(targetImage, targetFeature);
        }
        else if (featureType == "texture")
        {
            result = extractTextureColorFeature(targetImage, targetFeature);
        }
        else
        {
            std::cerr << "Error: Unknown feature type: " << featureType << std::endl;
            return -1;
        }
        
        if (result != 0)
        {
            std::cerr << "Error: Failed to extract features from target image" << std::endl;
            return -1;
        }
        
        std::cout << "Target feature size: " << targetFeature.size() << " values" << std::endl;
        std::cout << std::endl;
    }
    else if (featureType == "custom")
    {
        // For custom features, load the image and extract custom features
        std::cout << "Loading target image..." << std::endl;
        
        targetImage = cv::imread(targetImagePath);
        
        if (targetImage.empty())
        {
            std::cerr << "Error: Failed to load target image: " << targetImagePath << std::endl;
            return -1;
        }
        
        std::cout << "Target image size: " << targetImage.cols << "x" << targetImage.rows << std::endl;
        
        std::cout << "Extracting custom features from target image..." << std::endl;
        
        if (extractCustomBlueSceneFeature(targetImage, targetFeature) != 0)
        {
            std::cerr << "Error: Failed to extract custom features from target image" << std::endl;
            return -1;
        }
        
        std::cout << "Target custom feature size: " << targetFeature.size() << " values" << std::endl;
        std::cout << std::endl;
        
        // Will load DNN features from CSV later
        std::cout << "Will load DNN features from CSV for target image" << std::endl;
        std::cout << std::endl;
    }
    else
    {
        // For DNN features, we'll load from CSV later
        std::cout << "DNN mode: Will load target features from CSV database" << std::endl;
        std::cout << std::endl;
    }
    
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
    
    // For DNN features, extract target feature from database
    if (featureType == "dnn")
    {
        std::cout << "Searching for target image in database..." << std::endl;
        
        bool found = false;
        
        for (const auto &data : database)
        {
            if (data.filename == targetFilename)
            {
                targetFeature = data.feature;
                found = true;
                std::cout << "Found target image: " << targetFilename << std::endl;
                std::cout << "Target feature size: " << targetFeature.size() << " values" << std::endl;
                std::cout << std::endl;
                break;
            }
        }
        
        if (!found)
        {
            std::cerr << "Error: Target image '" << targetFilename 
                      << "' not found in DNN feature database" << std::endl;
            std::cerr << "Make sure the filename matches exactly (including extension)" << std::endl;
            return -1;
        }
    }
    
    // === Step 4: Load DNN database for custom features ===
    
    std::vector<FeatureData> dnnDatabase;
    
    if (featureType == "custom")
    {
        std::cout << "Loading DNN feature database from CSV..." << std::endl;
        
        if (readFeaturesFromCSV(dnnCSV, dnnDatabase) != 0)
        {
            std::cerr << "Error: Failed to load DNN feature database" << std::endl;
            return -1;
        }
        
        if (dnnDatabase.empty())
        {
            std::cerr << "Error: DNN feature database is empty" << std::endl;
            return -1;
        }
        
        std::cout << "Loaded " << dnnDatabase.size() << " DNN feature vectors" << std::endl;
        std::cout << std::endl;
        
        // Find target DNN features
        std::cout << "Searching for target image in DNN database..." << std::endl;
        
        bool found = false;
        
        for (const auto &data : dnnDatabase)
        {
            if (data.filename == targetFilename)
            {
                targetDNNFeature = data.feature;
                found = true;
                std::cout << "Found target DNN features: " << targetFilename << std::endl;
                std::cout << "Target DNN feature size: " << targetDNNFeature.size() << " values" << std::endl;
                std::cout << std::endl;
                break;
            }
        }
        
        if (!found)
        {
            std::cerr << "Error: Target image '" << targetFilename 
                      << "' not found in DNN feature database" << std::endl;
            return -1;
        }
    }
    
    // === Step 5: Compare target to all database images ===
    
    std::cout << "Computing distances to all database images..." << std::endl;
    
    std::vector<MatchResult> results;
    results.reserve(database.size());  // Reserve space for efficiency
    
    for (size_t i = 0; i < database.size(); i++)
    {
        // Compute distance based on feature type
        float dist;
        
        if (featureType == "baseline")
        {
            // Task 1: Sum of Squared Differences
            dist = distanceSSD(targetFeature, database[i].feature);
        }
        else if (featureType == "histogram")
        {
            // Task 2: Histogram Intersection
            dist = distanceHistogramIntersection(targetFeature, database[i].feature);
        }
        else if (featureType == "multihistogram")
        {
            // Task 3: Weighted Multi-Histogram (2 histograms: top + bottom)
            std::vector<float> weights = {0.5f, 0.5f};
            dist = distanceMultiHistogram(targetFeature, database[i].feature, 2, weights);
        }
        else if (featureType == "texture")
        {
            // Task 4: Color + Texture
            dist = distanceTextureColor(targetFeature, database[i].feature, 256, 16, 0.5f, 0.5f);
        }
        else if (featureType == "dnn")
        {
            // Task 5: Cosine Distance for DNN embeddings
            dist = distanceCosine(targetFeature, database[i].feature);
        }
        else if (featureType == "custom")
        {
            // Task 7: Custom blue scene detector
            // Need to find corresponding DNN features for this database image
            
            std::vector<float> dbDNNFeature;
            bool foundDNN = false;
            
            for (const auto &dnnData : dnnDatabase)
            {
                if (dnnData.filename == database[i].filename)
                {
                    dbDNNFeature = dnnData.feature;
                    foundDNN = true;
                    break;
                }
            }
            
            if (!foundDNN)
            {
                std::cerr << "Warning: DNN features not found for " << database[i].filename << std::endl;
                continue;
            }
            
            // Compute custom distance combining custom features + DNN
            dist = distanceCustomBlueScene(targetFeature, database[i].feature,
                                          targetDNNFeature, dbDNNFeature);
        }
        else
        {
            std::cerr << "Error: Unknown feature type: " << featureType << std::endl;
            return -1;
        }
        
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
    
    // === Step 6: Sort results by distance (ascending) ===
    
    std::cout << "Sorting results by distance..." << std::endl;
    
    // Sort using the comparison operator defined in MatchResult
    // This sorts in ascending order (smallest distance first)
    std::sort(results.begin(), results.end());
    
    std::cout << "Sorting complete" << std::endl;
    
    // === Step 7: Display top N matches ===
    
    printTopMatches(results, numMatches);
    
    // === Step 8: For custom features, also show some least similar (optional but helpful) ===
    
    if (featureType == "custom" && static_cast<int>(results.size()) > numMatches)
    {
        std::cout << "\n======================================" << std::endl;
        std::cout << "Bottom 3 matches (least similar):" << std::endl;
        std::cout << "======================================" << std::endl;
        
        int start = std::max(0, static_cast<int>(results.size()) - 3);
        for (int i = start; i < static_cast<int>(results.size()); i++)
        {
            std::cout << std::setw(2) << (i + 1) << ". " 
                      << std::setw(20) << std::left << results[i].filename 
                      << " (distance: " << std::fixed << std::setprecision(6) 
                      << results[i].distance << ")" << std::endl;
        }
        std::cout << "======================================\n" << std::endl;
    }
    
    // === Step 9: Verify expected results for pic.1016.jpg (baseline only) ===
    
    // Check if this is pic.1016.jpg with baseline features and verify expected matches
    if (targetFilename == "pic.1016.jpg" && featureType == "baseline" && results.size() >= 4)
    {
        std::cout << "Verification for pic.1016.jpg (baseline):" << std::endl;
        std::cout << "Expected top 4 matches: pic.1016.jpg, pic.0986.jpg, pic.0641.jpg, pic.0547.jpg" << std::endl;
        std::cout << "Actual top 4 matches: " 
                  << results[0].filename << ", "
                  << results[1].filename << ", "
                  << results[2].filename << ", "
                  << results[3].filename << std::endl;
        
        // Check if matches are correct
        bool match1 = (results[0].filename == "pic.1016.jpg");
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
    
    // === Step 10: Success message ===
    
    std::cout << "========================================" << std::endl;
    std::cout << "Query completed successfully!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}