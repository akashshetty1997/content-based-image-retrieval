/*
 * Name: Akash Shridhar Shetty
 * Date: February 2025
 * File: utils.h
 *
 * Purpose:
 * Utility functions for CSV I/O, file reading, and result sorting.
 * Provides helper structures and functions for feature database management
 * and query result handling in content-based image retrieval.
 */

#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

/**
 * Structure to hold feature data
 * Contains filename and its corresponding feature vector
 */
struct FeatureData {
    std::string filename;           
    std::vector<float> feature;     
};

/**
 * Structure to hold query results
 * Contains filename and distance from query image
 */
struct MatchResult {
    std::string filename;          
    float distance;                 
    
    bool operator<(const MatchResult &other) const {
        return distance < other.distance;
    }
};

/**
 * Write features to CSV file
 * Format: filename,feature1,feature2,...,featureN
 * @param filename Output CSV filename (e.g., "baseline_features.csv")
 * @param features Vector of feature data to write
 * @return 0 on success, -1 on error
 * 
 * Implementation details:
 * - Opens file in write mode
 * - Writes one line per image: filename followed by comma-separated features
 * - Uses std::fixed and std::setprecision for consistent float formatting
 * 
 * Example output line:
 * pic.1016.jpg,125.3,130.2,142.1,...,118.5
 */
int writeFeaturesToCSV(const std::string &filename, 
                       const std::vector<FeatureData> &features);

/**
 * Read features from CSV file
 * Reads the CSV format created by writeFeaturesToCSV
 * @param filename Input CSV filename
 * @param features Output vector of feature data (cleared and populated)
 * @return 0 on success, -1 on error
 * 
 * Implementation details:
 * - Opens file in read mode
 * - Parses each line: first token is filename, rest are feature values
 * - Uses std::stringstream for parsing comma-separated values
 * - Clears output vector before populating
 * 
 * Example input line:
 * pic.1016.jpg,125.3,130.2,142.1,...,118.5
 */
int readFeaturesFromCSV(const std::string &filename, 
                        std::vector<FeatureData> &features);

/**
 * Get all image filenames from a directory
 * Filters for common image extensions (.jpg, .jpeg, .png, .bmp)
 * @param dirPath Directory path to search
 * @param filenames Output vector of filenames (relative to dirPath)
 * @return 0 on success, -1 on error
 * 
 * Implementation details:
 * - Uses C++17 filesystem or platform-specific directory reading
 * - Filters by extension (case-insensitive)
 * - Returns only filenames, not full paths
 * - Sorts filenames alphabetically for consistency
 * 
 * Example output:
 * {"pic.0001.jpg", "pic.0002.jpg", "pic.0003.jpg", ...}
 */
int getImageFilenames(const std::string &dirPath, 
                      std::vector<std::string> &filenames);

/**
 * Print top N matches to console
 * Displays ranked results with distances
 * @param results Vector of match results (should be sorted)
 * @param topN Number of results to display
 * 
 * Implementation details:
 * - Prints header line
 * - Displays rank, filename, and distance for each result
 * - Formats distance with fixed precision
 * 
 * Example output:
 * Top 3 matches:
 * 1. pic.1016.jpg (distance: 0.000000)
 * 2. pic.0986.jpg (distance: 1234.567)
 * 3. pic.0641.jpg (distance: 2345.678)
 */
void printTopMatches(const std::vector<MatchResult> &results, int topN);

#endif 