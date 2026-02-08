/*
 * Name: Akash Shridhar Shetty
 * Date: February 2025
 * File: gui_query.cpp
 *
 * Purpose:
 * Extension: GUI for content-based image retrieval.
 * Displays target image and top N matches side by side in a window.
 *
 * Usage:
 *   ./gui_query <image_directory> <feature_csv> <num_matches> <feature_type> [dnn_csv]
 *
 * Example:
 *   ./gui_query data/olympus/ data/histogram_features.csv 3 histogram
 *   ./gui_query data/olympus/ data/my_dnn_features.csv 3 dnn
 *   ./gui_query data/olympus/ data/custom_features.csv 5 custom data/my_dnn_features.csv
 *
 * Controls:
 *   Click on any image in the grid to use it as the new target
 *   'q' or ESC - Quit
 *   'n' - Next page of results (if more than displayed)
 *   '1'-'6' - Switch feature type
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include "features.h"
#include "distance.h"
#include "utils.h"

// ========================================
// Constants
// ========================================

const int THUMB_WIDTH = 200;
const int THUMB_HEIGHT = 150;
const int PADDING = 10;
const int LABEL_HEIGHT = 25;
const cv::Scalar BG_COLOR(30, 30, 30);       // Dark background
const cv::Scalar TARGET_BORDER(0, 255, 255); // Yellow border for target
const cv::Scalar MATCH_BORDER(0, 255, 0);    // Green border for matches
const cv::Scalar TEXT_COLOR(255, 255, 255);  // White text
const cv::Scalar DIST_COLOR(180, 180, 180);  // Gray for distance text
const cv::Scalar HEADER_COLOR(0, 200, 255);  // Orange for headers

// ========================================
// Helper Functions
// ========================================

/**
 * Resize image to thumbnail while maintaining aspect ratio
 * Pads with black to fill the thumbnail area
 */
cv::Mat createThumbnail(const cv::Mat &src, int width, int height)
{
    if (src.empty())
    {
        cv::Mat blank(height, width, CV_8UC3, cv::Scalar(50, 50, 50));
        cv::putText(blank, "No Image", cv::Point(30, height / 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1);
        return blank;
    }

    // Calculate scale to fit within thumbnail
    float scaleX = static_cast<float>(width) / src.cols;
    float scaleY = static_cast<float>(height) / src.rows;
    float scale = std::min(scaleX, scaleY);

    int newWidth = static_cast<int>(src.cols * scale);
    int newHeight = static_cast<int>(src.rows * scale);

    // Resize
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(newWidth, newHeight));

    // Create black canvas and center the resized image
    cv::Mat thumb(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    int offsetX = (width - newWidth) / 2;
    int offsetY = (height - newHeight) / 2;
    resized.copyTo(thumb(cv::Rect(offsetX, offsetY, newWidth, newHeight)));

    return thumb;
}

/**
 * Draw a colored border around an image
 */
void drawBorder(cv::Mat &img, const cv::Scalar &color, int thickness = 3)
{
    cv::rectangle(img, cv::Point(0, 0),
                  cv::Point(img.cols - 1, img.rows - 1),
                  color, thickness);
}

/**
 * Extract features for a single image based on feature type
 */
int extractFeatureForType(const cv::Mat &image, const std::string &featureType,
                          std::vector<float> &feature)
{
    if (featureType == "baseline")
        return extractBaselineFeature(image, feature);
    else if (featureType == "histogram")
        return extractRGChromaticityHistogram(image, feature);
    else if (featureType == "multihistogram")
        return extractMultiHistogram(image, feature);
    else if (featureType == "texture")
        return extractTextureColorFeature(image, feature);
    else if (featureType == "custom")
        return extractCustomBlueSceneFeature(image, feature);
    return -1;
}

/**
 * Compute distance based on feature type
 */
float computeDistance(const std::string &featureType,
                      const std::vector<float> &f1, const std::vector<float> &f2,
                      const std::vector<float> &dnn1 = {},
                      const std::vector<float> &dnn2 = {})
{
    if (featureType == "baseline")
        return distanceSSD(f1, f2);
    else if (featureType == "histogram")
        return distanceHistogramIntersection(f1, f2);
    else if (featureType == "multihistogram")
        return distanceMultiHistogram(f1, f2, 2, {0.5f, 0.5f});
    else if (featureType == "texture")
        return distanceTextureColor(f1, f2, 256, 16, 0.5f, 0.5f);
    else if (featureType == "dnn")
        return distanceCosine(f1, f2);
    else if (featureType == "custom")
        return distanceCustomBlueScene(f1, f2, dnn1, dnn2);
    return -1.0f;
}

/**
 * Run a query and return sorted results
 */
std::vector<MatchResult> runQuery(const std::string &targetFilename,
                                  const std::string &featureType,
                                  const std::string &imageDir,
                                  const std::vector<FeatureData> &database,
                                  const std::vector<FeatureData> &dnnDatabase,
                                  const cv::Mat &targetImage)
{
    std::vector<MatchResult> results;
    std::vector<float> targetFeature;
    std::vector<float> targetDNNFeature;

    // Get target features
    if (featureType == "dnn")
    {
        // Load from database
        for (const auto &data : database)
        {
            if (data.filename == targetFilename)
            {
                targetFeature = data.feature;
                break;
            }
        }
    }
    else if (featureType == "custom")
    {
        extractFeatureForType(targetImage, featureType, targetFeature);
        for (const auto &data : dnnDatabase)
        {
            if (data.filename == targetFilename)
            {
                targetDNNFeature = data.feature;
                break;
            }
        }
    }
    else
    {
        extractFeatureForType(targetImage, featureType, targetFeature);
    }

    if (targetFeature.empty())
    {
        std::cerr << "Warning: Could not get features for " << targetFilename << std::endl;
        return results;
    }

    // Compare to all database images
    for (size_t i = 0; i < database.size(); i++)
    {
        float dist;

        if (featureType == "custom")
        {
            // Find DNN feature for this database image
            std::vector<float> dbDNN;
            for (const auto &d : dnnDatabase)
            {
                if (d.filename == database[i].filename)
                {
                    dbDNN = d.feature;
                    break;
                }
            }
            if (dbDNN.empty())
                continue;
            dist = computeDistance(featureType, targetFeature, database[i].feature,
                                   targetDNNFeature, dbDNN);
        }
        else
        {
            dist = computeDistance(featureType, targetFeature, database[i].feature);
        }

        if (dist >= 0)
        {
            MatchResult m;
            m.filename = database[i].filename;
            m.distance = dist;
            results.push_back(m);
        }
    }

    std::sort(results.begin(), results.end());
    return results;
}

/**
 * Build the display canvas showing target + matches
 */
cv::Mat buildDisplay(const std::string &targetFilename,
                     const std::string &featureType,
                     const std::vector<MatchResult> &results,
                     const std::string &imageDir,
                     int numMatches,
                     std::vector<cv::Rect> &clickRegions,
                     std::vector<std::string> &clickFilenames)
{
    clickRegions.clear();
    clickFilenames.clear();

    // Layout: target on left, matches in grid on right
    int cols = std::min(numMatches, 4); // Max 4 per row
    int rows = (numMatches + cols - 1) / cols;

    int cellWidth = THUMB_WIDTH + PADDING;
    int cellHeight = THUMB_HEIGHT + LABEL_HEIGHT + PADDING;

    // Canvas dimensions
    int canvasWidth = PADDING + cellWidth + PADDING + (cols * cellWidth) + PADDING;
    int canvasHeight = 60 + std::max(cellHeight + PADDING, rows * cellHeight + PADDING) + 40;

    cv::Mat canvas(canvasHeight, canvasWidth, CV_8UC3, BG_COLOR);

    // === Header ===
    std::string header = "CBIR - Feature: " + featureType + " | Target: " + targetFilename;
    cv::putText(canvas, header, cv::Point(PADDING, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, HEADER_COLOR, 1);

    std::string controls = "Click match to re-query | 1-6: change feature | q: quit";
    cv::putText(canvas, controls, cv::Point(PADDING, canvasHeight - 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, DIST_COLOR, 1);

    int startY = 50;

    // === Target image (left side) ===
    std::string targetPath = imageDir;
    if (targetPath.back() != '/')
        targetPath += '/';
    targetPath += targetFilename;

    cv::Mat targetImg = cv::imread(targetPath);
    cv::Mat targetThumb = createThumbnail(targetImg, THUMB_WIDTH, THUMB_HEIGHT);
    drawBorder(targetThumb, TARGET_BORDER, 3);

    int targetX = PADDING;
    int targetY = startY;
    targetThumb.copyTo(canvas(cv::Rect(targetX, targetY, THUMB_WIDTH, THUMB_HEIGHT)));

    cv::putText(canvas, "TARGET", cv::Point(targetX + 60, targetY + THUMB_HEIGHT + 18),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, TARGET_BORDER, 1);

    // === Match images (right side grid) ===
    int matchStartX = PADDING + cellWidth + PADDING;

    // Skip index 0 (target itself) and show top matches
    int displayCount = 0;
    for (size_t i = 0; i < results.size() && displayCount < numMatches; i++)
    {
        // Skip self-match
        if (results[i].filename == targetFilename)
            continue;

        int gridRow = displayCount / cols;
        int gridCol = displayCount % cols;

        int x = matchStartX + gridCol * cellWidth;
        int y = startY + gridRow * cellHeight;

        // Load and display match image
        std::string matchPath = imageDir;
        if (matchPath.back() != '/')
            matchPath += '/';
        matchPath += results[i].filename;

        cv::Mat matchImg = cv::imread(matchPath);
        cv::Mat matchThumb = createThumbnail(matchImg, THUMB_WIDTH, THUMB_HEIGHT);
        drawBorder(matchThumb, MATCH_BORDER, 2);

        matchThumb.copyTo(canvas(cv::Rect(x, y, THUMB_WIDTH, THUMB_HEIGHT)));

        // Label: rank and filename
        std::string label = "#" + std::to_string(displayCount + 1) + " " + results[i].filename;
        cv::putText(canvas, label, cv::Point(x + 2, y + THUMB_HEIGHT + 14),
                    cv::FONT_HERSHEY_SIMPLEX, 0.35, TEXT_COLOR, 1);

        // Distance value
        char distStr[32];
        snprintf(distStr, sizeof(distStr), "d=%.4f", results[i].distance);
        cv::putText(canvas, distStr, cv::Point(x + 2, y + THUMB_HEIGHT + 24),
                    cv::FONT_HERSHEY_SIMPLEX, 0.3, DIST_COLOR, 1);

        // Store clickable region
        clickRegions.push_back(cv::Rect(x, y, THUMB_WIDTH, THUMB_HEIGHT));
        clickFilenames.push_back(results[i].filename);

        displayCount++;
    }

    return canvas;
}

// ========================================
// Mouse callback globals
// ========================================

struct MouseData
{
    std::vector<cv::Rect> clickRegions;
    std::vector<std::string> clickFilenames;
    std::string clickedFile;
    bool clicked;
};

void onMouse(int event, int x, int y, int /*flags*/, void *userdata)
{
    if (event != cv::EVENT_LBUTTONDOWN)
        return;

    MouseData *data = static_cast<MouseData *>(userdata);

    for (size_t i = 0; i < data->clickRegions.size(); i++)
    {
        if (data->clickRegions[i].contains(cv::Point(x, y)))
        {
            data->clickedFile = data->clickFilenames[i];
            data->clicked = true;
            return;
        }
    }
}

// ========================================
// Main
// ========================================

int main(int argc, char *argv[])
{
    // === Parse arguments ===

    bool validArgs = (argc == 5) || (argc == 6);

    if (!validArgs)
    {
        std::cerr << "Usage: " << argv[0] << " <image_dir> <feature_csv> <num_matches> <feature_type> [dnn_csv]" << std::endl;
        std::cerr << "\nExamples:" << std::endl;
        std::cerr << "  " << argv[0] << " data/olympus/ data/baseline_features.csv 4 baseline" << std::endl;
        std::cerr << "  " << argv[0] << " data/olympus/ data/histogram_features.csv 4 histogram" << std::endl;
        std::cerr << "  " << argv[0] << " data/olympus/ data/my_dnn_features.csv 4 dnn" << std::endl;
        std::cerr << "  " << argv[0] << " data/olympus/ data/custom_features.csv 6 custom data/my_dnn_features.csv" << std::endl;
        return -1;
    }

    std::string imageDir = argv[1];
    std::string featureCSV = argv[2];
    int numMatches = std::stoi(argv[3]);
    std::string featureType = argv[4];
    std::string dnnCSV = (argc == 6) ? argv[5] : "";

    std::cout << "Loading feature database..." << std::endl;

    // Load feature database
    std::vector<FeatureData> database;
    if (readFeaturesFromCSV(featureCSV, database) != 0 || database.empty())
    {
        std::cerr << "Error: Failed to load feature database" << std::endl;
        return -1;
    }
    std::cout << "Loaded " << database.size() << " features" << std::endl;

    // Load DNN database if needed
    std::vector<FeatureData> dnnDatabase;
    if (featureType == "custom" && !dnnCSV.empty())
    {
        if (readFeaturesFromCSV(dnnCSV, dnnDatabase) != 0)
        {
            std::cerr << "Error: Failed to load DNN database" << std::endl;
            return -1;
        }
        std::cout << "Loaded " << dnnDatabase.size() << " DNN features" << std::endl;
    }

    // Get list of all images for browsing
    std::vector<std::string> allImages;
    getImageFilenames(imageDir, allImages);

    // Start with first image as target
    std::string currentTarget = database[0].filename;

    // Setup window
    std::string windowName = "CBIR - Content-Based Image Retrieval";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    MouseData mouseData;
    mouseData.clicked = false;
    cv::setMouseCallback(windowName, onMouse, &mouseData);

    std::cout << "\n========================================" << std::endl;
    std::cout << "GUI Controls:" << std::endl;
    std::cout << "  Click any match to use as new target" << std::endl;
    std::cout << "  1 = baseline, 2 = histogram, 3 = multihistogram" << std::endl;
    std::cout << "  4 = texture, 5 = dnn, 6 = custom" << std::endl;
    std::cout << "  n/p = next/previous image in database" << std::endl;
    std::cout << "  q/ESC = quit" << std::endl;
    std::cout << "========================================\n"
              << std::endl;

    int currentIndex = 0;
    bool needsUpdate = true;

    while (true)
    {
        if (needsUpdate)
        {
            std::cout << "Querying: " << currentTarget << " with " << featureType << "..." << std::endl;

            // Load target image
            std::string targetPath = imageDir;
            if (targetPath.back() != '/')
                targetPath += '/';
            targetPath += currentTarget;
            cv::Mat targetImage = cv::imread(targetPath);

            // Run query
            std::vector<MatchResult> results = runQuery(
                currentTarget, featureType, imageDir,
                database, dnnDatabase, targetImage);

            // Build display
            std::vector<cv::Rect> regions;
            std::vector<std::string> filenames;
            cv::Mat display = buildDisplay(currentTarget, featureType, results,
                                           imageDir, numMatches, regions, filenames);

            // Update mouse data
            mouseData.clickRegions = regions;
            mouseData.clickFilenames = filenames;
            mouseData.clicked = false;

            cv::imshow(windowName, display);
            needsUpdate = false;
        }

        int key = cv::waitKey(50);

        // Check mouse click
        if (mouseData.clicked)
        {
            currentTarget = mouseData.clickedFile;
            mouseData.clicked = false;
            needsUpdate = true;

            // Update index
            for (size_t i = 0; i < allImages.size(); i++)
            {
                if (allImages[i] == currentTarget)
                {
                    currentIndex = static_cast<int>(i);
                    break;
                }
            }
            continue;
        }

        if (key == 'q' || key == 27) // q or ESC
        {
            break;
        }
        else if (key == 'n') // Next image
        {
            currentIndex = (currentIndex + 1) % allImages.size();
            currentTarget = allImages[currentIndex];
            needsUpdate = true;
        }
        else if (key == 'p') // Previous image
        {
            currentIndex = (currentIndex - 1 + allImages.size()) % allImages.size();
            currentTarget = allImages[currentIndex];
            needsUpdate = true;
        }
        else if (key == '1')
        {
            featureType = "baseline";
            needsUpdate = true;
        }
        else if (key == '2')
        {
            featureType = "histogram";
            needsUpdate = true;
        }
        else if (key == '3')
        {
            featureType = "multihistogram";
            needsUpdate = true;
        }
        else if (key == '4')
        {
            featureType = "texture";
            needsUpdate = true;
        }
        else if (key == '5')
        {
            featureType = "dnn";
            needsUpdate = true;
        }
        else if (key == '6')
        {
            featureType = "custom";
            needsUpdate = true;
        }
    }

    cv::destroyAllWindows();
    std::cout << "GUI closed." << std::endl;

    return 0;
}