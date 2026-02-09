/*
 * Name: Akash Shridhar Shetty, Skandhan Madhusudhana
 * Date: February 2025
 * File: utils.cpp
 *
 * Purpose:
 * Implementation of utility functions for CSV I/O, file operations,
 * and result display for content-based image retrieval system.
 */

#include "utils.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

/**
 * Write features to CSV file
 * Format: filename,feature1,feature2,...,featureN
 * @param filename Output CSV filename
 * @param features Vector of feature data to write
 * @return 0 on success, -1 on error
 * 
 * Implementation details:
 * What it does:
 *  - Opens CSV file for writing
 *  - For each image:
 *      1. Write filename
 *      2. Write comma
 *      3. Write all feature values separated by commas
 *      4. Write newline
 *  - Uses std::fixed for consistent float output
 * 
 * Example CSV output:
 * pic.0001.jpg,120.5,130.2,125.8,...,118.3
 * pic.0002.jpg,115.1,128.9,130.5,...,122.7
 */
int writeFeaturesToCSV(const std::string &filename, 
                       const std::vector<FeatureData> &features)
{
    // Open file for writing
    std::ofstream file(filename);
    
    // Check if file opened successfully
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return -1;
    }
    
    // Set floating point precision (6 decimal places)
    file << std::fixed << std::setprecision(6);
    
    // Write each feature vector
    for (const auto &data : features)
    {
        // Write filename
        file << data.filename;
        
        // Write all feature values separated by commas
        for (size_t i = 0; i < data.feature.size(); i++)
        {
            file << "," << data.feature[i];
        }
        
        // End line
        file << std::endl;
    }
    
    file.close();
    
    std::cout << "Successfully wrote " << features.size() 
              << " feature vectors to " << filename << std::endl;
    
    return 0;
}

/**
 * Read features from CSV file
 * Parses CSV format created by writeFeaturesToCSV
 * @param filename Input CSV filename
 * @param features Output vector of feature data (cleared first)
 * @return 0 on success, -1 on error
 * 
 * Implementation details:
 * What it does:
 *  - Opens CSV file for reading
 *  - For each line:
 *      1. Parse filename (everything before first comma)
 *      2. Parse all feature values (everything after first comma)
 *      3. Store in FeatureData structure
 *  - Uses std::stringstream for parsing
 *  - Uses std::getline with ',' delimiter
 * 
 * Example parsing:
 * Input line: "pic.0001.jpg,120.5,130.2,125.8"
 * Result: filename="pic.0001.jpg", feature=[120.5, 130.2, 125.8]
 */
int readFeaturesFromCSV(const std::string &filename, 
                        std::vector<FeatureData> &features)
{
    // Clear output vector
    features.clear();
    
    // Open file for reading
    std::ifstream file(filename);
    
    // Check if file opened successfully
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file for reading: " << filename << std::endl;
        return -1;
    }
    
    std::string line;
    int lineCount = 0;
    
    // Read file line by line
    while (std::getline(file, line))
    {
        lineCount++;
        
        // Skip empty lines
        if (line.empty())
            continue;
        
        // Create string stream for parsing
        std::stringstream ss(line);
        std::string token;
        
        // Create new FeatureData entry
        FeatureData data;
        
        // First token is filename
        if (!std::getline(ss, token, ','))
        {
            std::cerr << "Warning: Malformed line " << lineCount << std::endl;
            continue;
        }
        data.filename = token;
        
        // Rest are feature values
        while (std::getline(ss, token, ','))
        {
            try {
                float value = std::stof(token);
                data.feature.push_back(value);
            }
            catch (const std::exception &e)
            {
                std::cerr << "Warning: Invalid float value on line " << lineCount 
                          << ": " << token << std::endl;
            }
        }
        
        // Add to features vector if we got some feature values
        if (!data.feature.empty())
        {
            features.push_back(data);
        }
        else
        {
            std::cerr << "Warning: No features found on line " << lineCount << std::endl;
        }
    }
    
    file.close();
    
    std::cout << "Successfully read " << features.size() 
              << " feature vectors from " << filename << std::endl;
    
    return 0;
}

/**
 * Get all image filenames from a directory
 * Filters for common image extensions (.jpg, .jpeg, .png, .bmp)
 * @param dirPath Directory path to search
 * @param filenames Output vector of filenames (basename only, not full path)
 * @return 0 on success, -1 on error
 * 
 * Implementation details:
 * What it does:
 *  - Uses C++17 filesystem to iterate through directory
 *  - Checks if each file has image extension (case-insensitive)
 *  - Extracts filename only (not full path)
 *  - Sorts filenames alphabetically
 * 
 * Example:
 * Input: "/path/to/olympus/"
 * Output: ["pic.0001.jpg", "pic.0002.jpg", "pic.0003.jpg", ...]
 */
int getImageFilenames(const std::string &dirPath, 
                      std::vector<std::string> &filenames)
{
    // Clear output vector
    filenames.clear();
    
    try
    {
        // Check if directory exists
        if (!fs::exists(dirPath) || !fs::is_directory(dirPath))
        {
            std::cerr << "Error: Directory does not exist: " << dirPath << std::endl;
            return -1;
        }
        
        // Valid image extensions
        std::vector<std::string> validExtensions = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"};
        
        // Iterate through directory
        for (const auto &entry : fs::directory_iterator(dirPath))
        {
            // Check if it's a regular file
            if (entry.is_regular_file())
            {
                std::string ext = entry.path().extension().string();
                
                // Check if extension is valid
                if (std::find(validExtensions.begin(), validExtensions.end(), ext) != validExtensions.end())
                {
                    // Add filename (not full path)
                    filenames.push_back(entry.path().filename().string());
                }
            }
        }
        
        // Sort filenames alphabetically for consistency
        std::sort(filenames.begin(), filenames.end());
        
        std::cout << "Found " << filenames.size() << " images in " << dirPath << std::endl;
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error reading directory: " << e.what() << std::endl;
        return -1;
    }
}

/**
 * Print top N matches to console
 * Displays ranked results with distances in a readable format
 * @param results Vector of match results (should already be sorted)
 * @param topN Number of results to display
 * 
 * Implementation details:
 * What it does:
 *  - Prints header
 *  - For each result (up to topN):
 *      1. Print rank number
 *      2. Print filename
 *      3. Print distance with fixed precision
 *  - Uses formatting for alignment
 * 
 * Example output:
 * ======================================
 * Top 3 matches:
 * ======================================
 * 1. pic.1016.jpg        (distance: 0.000000)
 * 2. pic.0986.jpg        (distance: 1234.567890)
 * 3. pic.0641.jpg        (distance: 2345.678901)
 * ======================================
 */
void printTopMatches(const std::vector<MatchResult> &results, int topN)
{
    // Determine how many results to actually print
    int numToPrint = std::min(topN, static_cast<int>(results.size()));
    
    // Print header
    std::cout << "\n======================================" << std::endl;
    std::cout << "Top " << numToPrint << " matches:" << std::endl;
    std::cout << "======================================" << std::endl;
    
    // Print results
    for (int i = 0; i < numToPrint; i++)
    {
        std::cout << std::setw(2) << (i + 1) << ". " 
                  << std::setw(20) << std::left << results[i].filename 
                  << " (distance: " << std::fixed << std::setprecision(6) 
                  << results[i].distance << ")" << std::endl;
    }
    
    std::cout << "======================================\n" << std::endl;
}