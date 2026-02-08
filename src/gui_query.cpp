/*
 * Name: Akash Shridhar Shetty
 * Date: February 2025
 * File: gui_query.cpp
 *
 * Purpose:
 * Extension: All-in-one GUI for content-based image retrieval.
 * Single window with feature type selector, image browser, search bar,
 * and results grid. Loads all feature databases at startup.
 *
 * Usage:
 *   ./gui_query <image_directory> <dnn_csv>
 *
 * Example:
 *   ./gui_query data/olympus/ data/ResNet18_olym.csv
 *
 * Controls:
 *   Click on any image to use it as the new target
 *   Use trackbar to switch feature type
 *   's' - Activate search mode (type filename, Enter to select, Esc to cancel)
 *   'n'/'p' - Next/Previous page of browser images
 *   '1'-'6' - Switch feature type
 *   'q'/ESC - Quit (when not in search mode)
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include "features.h"
#include "distance.h"
#include "utils.h"

// ========================================
// Constants
// ========================================

const int THUMB_W = 160;
const int THUMB_H = 120;
const int SMALL_THUMB_W = 100;
const int SMALL_THUMB_H = 75;
const int PAD = 8;
const int NUM_MATCHES = 6;
const int BROWSER_COLS = 8;

const cv::Scalar BG(30, 30, 30);
const cv::Scalar PANEL_BG(45, 45, 45);
const cv::Scalar TARGET_BORDER(0, 255, 255);
const cv::Scalar MATCH_BORDER(0, 255, 0);
const cv::Scalar BROWSER_BORDER(200, 200, 200);
const cv::Scalar SELECTED_BORDER(0, 200, 255);
const cv::Scalar WHITE(255, 255, 255);
const cv::Scalar GRAY(160, 160, 160);
const cv::Scalar HEADER(0, 200, 255);
const cv::Scalar DIVIDER(80, 80, 80);

// ========================================
// Feature type names
// ========================================

const std::vector<std::string> FEATURE_NAMES = {
    "baseline", "histogram", "multihistogram", "texture", "dnn", "custom"};

const std::vector<std::string> FEATURE_LABELS = {
    "1: Baseline (7x7 SSD)",
    "2: Histogram (rg Chrom)",
    "3: Multi-Hist (Top/Bot)",
    "4: Texture + Color",
    "5: DNN Embeddings",
    "6: Custom (Blue Scene)"};

const std::vector<std::string> CSV_FILES = {
    "data/baseline_features.csv",
    "data/histogram_features.csv",
    "data/multihistogram_features.csv",
    "data/texture_features.csv",
    "", // DNN loaded separately
    "data/custom_features.csv"};

// ========================================
// Helper Functions
// ========================================

cv::Mat makeThumbnail(const cv::Mat &src, int w, int h)
{
    if (src.empty())
    {
        cv::Mat blank(h, w, CV_8UC3, cv::Scalar(50, 50, 50));
        cv::putText(blank, "N/A", cv::Point(w / 3, h / 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1);
        return blank;
    }
    float sx = (float)w / src.cols;
    float sy = (float)h / src.rows;
    float s = std::min(sx, sy);
    int nw = (int)(src.cols * s);
    int nh = (int)(src.rows * s);
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(nw, nh));
    cv::Mat thumb(h, w, CV_8UC3, cv::Scalar(0, 0, 0));
    resized.copyTo(thumb(cv::Rect((w - nw) / 2, (h - nh) / 2, nw, nh)));
    return thumb;
}

void drawBorder(cv::Mat &img, const cv::Scalar &color, int t = 2)
{
    cv::rectangle(img, cv::Point(0, 0), cv::Point(img.cols - 1, img.rows - 1), color, t);
}

int extractFeature(const cv::Mat &img, const std::string &type, std::vector<float> &feat)
{
    if (type == "baseline")
        return extractBaselineFeature(img, feat);
    if (type == "histogram")
        return extractRGChromaticityHistogram(img, feat);
    if (type == "multihistogram")
        return extractMultiHistogram(img, feat);
    if (type == "texture")
        return extractTextureColorFeature(img, feat);
    if (type == "custom")
        return extractCustomBlueSceneFeature(img, feat);
    return -1;
}

float computeDist(const std::string &type,
                  const std::vector<float> &f1, const std::vector<float> &f2,
                  const std::vector<float> &d1 = {}, const std::vector<float> &d2 = {})
{
    if (type == "baseline")
        return distanceSSD(f1, f2);
    if (type == "histogram")
        return distanceHistogramIntersection(f1, f2);
    if (type == "multihistogram")
        return distanceMultiHistogram(f1, f2, 2, {0.5f, 0.5f});
    if (type == "texture")
        return distanceTextureColor(f1, f2, 256, 16, 0.5f, 0.5f);
    if (type == "dnn")
        return distanceCosine(f1, f2);
    if (type == "custom")
        return distanceCustomBlueScene(f1, f2, d1, d2);
    return -1.0f;
}

std::vector<MatchResult> runQuery(const std::string &targetFile,
                                  const std::string &featureType,
                                  const std::string &imageDir,
                                  const std::vector<FeatureData> &db,
                                  const std::vector<FeatureData> &dnnDb,
                                  const cv::Mat &targetImg)
{
    std::vector<MatchResult> results;
    std::vector<float> tFeat, tDNN;

    if (featureType == "dnn")
    {
        for (auto &d : db)
            if (d.filename == targetFile)
            {
                tFeat = d.feature;
                break;
            }
    }
    else if (featureType == "custom")
    {
        extractFeature(targetImg, featureType, tFeat);
        for (auto &d : dnnDb)
            if (d.filename == targetFile)
            {
                tDNN = d.feature;
                break;
            }
    }
    else
    {
        extractFeature(targetImg, featureType, tFeat);
    }

    if (tFeat.empty())
        return results;

    for (size_t i = 0; i < db.size(); i++)
    {
        float dist;
        if (featureType == "custom")
        {
            std::vector<float> dbDNN;
            for (auto &d : dnnDb)
                if (d.filename == db[i].filename)
                {
                    dbDNN = d.feature;
                    break;
                }
            if (dbDNN.empty())
                continue;
            dist = computeDist(featureType, tFeat, db[i].feature, tDNN, dbDNN);
        }
        else
        {
            dist = computeDist(featureType, tFeat, db[i].feature);
        }
        if (dist >= 0)
        {
            MatchResult m;
            m.filename = db[i].filename;
            m.distance = dist;
            results.push_back(m);
        }
    }
    std::sort(results.begin(), results.end());
    return results;
}

// ========================================
// Clickable region tracking
// ========================================

struct ClickRegion
{
    cv::Rect rect;
    std::string filename;
};

struct AppState
{
    std::vector<ClickRegion> regions;
    std::string clickedFile;
    bool clicked;
    int featureIdx;
    std::string searchText;
    bool searchActive;
    std::vector<std::string> searchResults;
};

void onMouse(int event, int x, int y, int, void *userdata)
{
    if (event != cv::EVENT_LBUTTONDOWN)
        return;
    AppState *state = (AppState *)userdata;
    for (auto &r : state->regions)
    {
        if (r.rect.contains(cv::Point(x, y)))
        {
            state->clickedFile = r.filename;
            state->clicked = true;
            return;
        }
    }
}

void onTrackbar(int pos, void *userdata)
{
    AppState *state = (AppState *)userdata;
    state->featureIdx = pos;
}

// ========================================
// Build Display
// ========================================

cv::Mat buildDisplay(const std::string &targetFile,
                     const std::string &featureType,
                     const std::vector<MatchResult> &results,
                     const std::string &imageDir,
                     const std::vector<std::string> &allImages,
                     int browserPage,
                     AppState &state)
{
    state.regions.clear();

    // Calculate layout
    int matchCols = 3;
    int matchRows = 2;
    int matchCellW = THUMB_W + PAD;
    int matchCellH = THUMB_H + 30 + PAD;

    int leftW = THUMB_W + PAD * 3;
    int rightW = matchCols * matchCellW + PAD;
    int browserW = BROWSER_COLS * (SMALL_THUMB_W + PAD);
    int canvasW = std::max(leftW + rightW + PAD, browserW);

    int topH = 45;
    int matchAreaH = matchRows * matchCellH + PAD;
    int browserH = SMALL_THUMB_H + 35 + PAD * 2;
    int searchH = 55;
    int statusH = 30;
    int canvasH = topH + matchAreaH + browserH + searchH + statusH + PAD * 2;

    cv::Mat canvas(canvasH, canvasW, CV_8UC3, BG);

    // === Header bar ===
    cv::rectangle(canvas, cv::Point(0, 0), cv::Point(canvasW, topH), PANEL_BG, -1);
    cv::putText(canvas, "Content-Based Image Retrieval", cv::Point(PAD, 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, HEADER, 1);

    std::string info = "Feature: " + featureType + " | Target: " + targetFile;
    cv::putText(canvas, info, cv::Point(PAD, 38),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, GRAY, 1);

    // === Target image (left) ===
    int tY = topH + PAD;
    std::string tPath = imageDir;
    if (tPath.back() != '/')
        tPath += '/';
    tPath += targetFile;
    cv::Mat tImg = cv::imread(tPath);
    cv::Mat tThumb = makeThumbnail(tImg, THUMB_W, THUMB_H);
    drawBorder(tThumb, TARGET_BORDER, 3);
    tThumb.copyTo(canvas(cv::Rect(PAD, tY, THUMB_W, THUMB_H)));

    cv::putText(canvas, "TARGET", cv::Point(PAD + THUMB_W / 2 - 25, tY + THUMB_H + 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, TARGET_BORDER, 1);

    // Feature type legend (below target)
    int legendY = tY + THUMB_H + 30;
    for (size_t i = 0; i < FEATURE_LABELS.size() && legendY + 15 < topH + (int)matchAreaH; i++)
    {
        cv::Scalar col = (i == (size_t)state.featureIdx) ? HEADER : GRAY;
        cv::putText(canvas, FEATURE_LABELS[i], cv::Point(PAD, legendY),
                    cv::FONT_HERSHEY_SIMPLEX, 0.3, col, 1);
        legendY += 14;
    }

    // === Match results (right) ===
    int mStartX = leftW;
    int mStartY = topH + PAD;
    int displayed = 0;

    for (size_t i = 0; i < results.size() && displayed < NUM_MATCHES; i++)
    {
        if (results[i].filename == targetFile)
            continue;

        int r = displayed / matchCols;
        int c = displayed % matchCols;
        int x = mStartX + c * matchCellW;
        int y = mStartY + r * matchCellH;

        std::string mPath = imageDir;
        if (mPath.back() != '/')
            mPath += '/';
        mPath += results[i].filename;

        cv::Mat mImg = cv::imread(mPath);
        cv::Mat mThumb = makeThumbnail(mImg, THUMB_W, THUMB_H);
        drawBorder(mThumb, MATCH_BORDER, 2);
        mThumb.copyTo(canvas(cv::Rect(x, y, THUMB_W, THUMB_H)));

        // Rank + filename
        char label[64];
        snprintf(label, sizeof(label), "#%d %s", displayed + 1, results[i].filename.c_str());
        cv::putText(canvas, label, cv::Point(x, y + THUMB_H + 12),
                    cv::FONT_HERSHEY_SIMPLEX, 0.3, WHITE, 1);

        // Distance
        char dStr[32];
        snprintf(dStr, sizeof(dStr), "d=%.4f", results[i].distance);
        cv::putText(canvas, dStr, cv::Point(x, y + THUMB_H + 24),
                    cv::FONT_HERSHEY_SIMPLEX, 0.28, GRAY, 1);

        // Clickable region
        ClickRegion cr;
        cr.rect = cv::Rect(x, y, THUMB_W, THUMB_H);
        cr.filename = results[i].filename;
        state.regions.push_back(cr);

        displayed++;
    }

    // === Divider line ===
    int divY = topH + matchAreaH;
    cv::line(canvas, cv::Point(PAD, divY), cv::Point(canvasW - PAD, divY), DIVIDER, 1);

    // === Image browser strip ===
    int bY = divY + PAD;
    cv::putText(canvas, "Image Browser (click to select, n/p to page):",
                cv::Point(PAD, bY + 12), cv::FONT_HERSHEY_SIMPLEX, 0.35, GRAY, 1);

    int bImgY = bY + 18;
    int bStartIdx = browserPage * BROWSER_COLS;

    for (int i = 0; i < BROWSER_COLS; i++)
    {
        int idx = bStartIdx + i;
        if (idx >= (int)allImages.size())
            break;

        int bx = PAD + i * (SMALL_THUMB_W + PAD);

        std::string bPath = imageDir;
        if (bPath.back() != '/')
            bPath += '/';
        bPath += allImages[idx];

        cv::Mat bImg = cv::imread(bPath);
        cv::Mat bThumb = makeThumbnail(bImg, SMALL_THUMB_W, SMALL_THUMB_H);

        if (allImages[idx] == targetFile)
            drawBorder(bThumb, SELECTED_BORDER, 2);
        else
            drawBorder(bThumb, BROWSER_BORDER, 1);

        bThumb.copyTo(canvas(cv::Rect(bx, bImgY, SMALL_THUMB_W, SMALL_THUMB_H)));

        // Filename label
        cv::putText(canvas, allImages[idx], cv::Point(bx, bImgY + SMALL_THUMB_H + 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.22, GRAY, 1);

        // Clickable
        ClickRegion cr;
        cr.rect = cv::Rect(bx, bImgY, SMALL_THUMB_W, SMALL_THUMB_H);
        cr.filename = allImages[idx];
        state.regions.push_back(cr);
    }

    // === Search bar ===
    int searchY = bImgY + SMALL_THUMB_H + 18 + PAD;
    cv::rectangle(canvas, cv::Point(0, searchY), cv::Point(canvasW, searchY + searchH), PANEL_BG, -1);

    cv::putText(canvas, "Search (press 's' to type, Enter to select, Esc to cancel):",
                cv::Point(PAD, searchY + 14), cv::FONT_HERSHEY_SIMPLEX, 0.35, GRAY, 1);

    // Search box
    int boxX = PAD;
    int boxY = searchY + 20;
    int boxW = 250;
    int boxH = 25;
    cv::Scalar boxColor = state.searchActive ? cv::Scalar(80, 80, 80) : cv::Scalar(60, 60, 60);
    cv::rectangle(canvas, cv::Point(boxX, boxY), cv::Point(boxX + boxW, boxY + boxH), boxColor, -1);
    cv::rectangle(canvas, cv::Point(boxX, boxY), cv::Point(boxX + boxW, boxY + boxH),
                  state.searchActive ? HEADER : GRAY, 1);

    // Search text with cursor
    std::string displayText = state.searchText;
    if (state.searchActive)
        displayText += "_";
    if (displayText.empty() && !state.searchActive)
        displayText = "Type filename...";

    cv::Scalar textCol = state.searchActive ? WHITE : GRAY;
    cv::putText(canvas, displayText, cv::Point(boxX + 5, boxY + 17),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, textCol, 1);

    // Search results (show matching filenames)
    if (!state.searchResults.empty() && state.searchActive)
    {
        int srX = boxX + boxW + PAD;
        int maxShow = std::min((int)state.searchResults.size(), 5);
        for (int i = 0; i < maxShow; i++)
        {
            cv::putText(canvas, state.searchResults[i],
                        cv::Point(srX + i * 120, boxY + 17),
                        cv::FONT_HERSHEY_SIMPLEX, 0.33, WHITE, 1);

            // Make search results clickable
            ClickRegion cr;
            cr.rect = cv::Rect(srX + i * 120, boxY, 115, boxH);
            cr.filename = state.searchResults[i];
            state.regions.push_back(cr);
        }
    }

    // === Status bar ===
    int sY = canvasH - statusH;
    cv::rectangle(canvas, cv::Point(0, sY), cv::Point(canvasW, canvasH), PANEL_BG, -1);

    char statusStr[128];
    snprintf(statusStr, sizeof(statusStr),
             "Page %d/%d | Images: %d | Click image to query | 1-6: feature | s: search | n/p: page | q: quit",
             browserPage + 1, (int)(allImages.size() / BROWSER_COLS + 1), (int)allImages.size());
    cv::putText(canvas, statusStr, cv::Point(PAD, sY + 18),
                cv::FONT_HERSHEY_SIMPLEX, 0.32, GRAY, 1);

    return canvas;
}

// ========================================
// Main
// ========================================

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <image_dir> <dnn_csv>" << std::endl;
        std::cerr << "\nExample:" << std::endl;
        std::cerr << "  " << argv[0] << " data/olympus/ data/ResNet18_olym.csv" << std::endl;
        return -1;
    }

    std::string imageDir = argv[1];
    std::string dnnCSV = argv[2];

    std::cout << "========================================" << std::endl;
    std::cout << "CBIR - Interactive GUI" << std::endl;
    std::cout << "========================================" << std::endl;

    // === Load all feature databases ===
    std::map<std::string, std::vector<FeatureData>> databases;

    for (size_t i = 0; i < FEATURE_NAMES.size(); i++)
    {
        std::string name = FEATURE_NAMES[i];
        std::string csv = CSV_FILES[i];

        if (name == "dnn")
        {
            csv = dnnCSV;
        }

        if (csv.empty())
            continue;

        std::cout << "Loading " << name << " features from " << csv << "..." << std::endl;
        std::vector<FeatureData> db;
        if (readFeaturesFromCSV(csv, db) == 0 && !db.empty())
        {
            databases[name] = db;
            std::cout << "  Loaded " << db.size() << " vectors (" << db[0].feature.size() << "D)" << std::endl;
        }
        else
        {
            std::cerr << "  Warning: Could not load " << csv << " (run extract_features first)" << std::endl;
        }
    }

    // Load DNN database separately for custom features
    std::vector<FeatureData> dnnDb;
    if (readFeaturesFromCSV(dnnCSV, dnnDb) == 0)
    {
        std::cout << "DNN database loaded for custom features" << std::endl;
    }

    // Get all image filenames
    std::vector<std::string> allImages;
    getImageFilenames(imageDir, allImages);
    std::cout << "Found " << allImages.size() << " images" << std::endl;

    if (allImages.empty())
    {
        std::cerr << "Error: No images found" << std::endl;
        return -1;
    }

    // === Setup state ===
    AppState state;
    state.clicked = false;
    state.featureIdx = 0;
    state.searchActive = false;

    std::string currentTarget = allImages[0];
    std::string currentFeature = FEATURE_NAMES[0];
    int browserPage = 0;
    int maxPages = (allImages.size() + BROWSER_COLS - 1) / BROWSER_COLS;
    bool needsUpdate = true;

    // === Create window ===
    std::string winName = "CBIR - Content-Based Image Retrieval";
    cv::namedWindow(winName, cv::WINDOW_NORMAL);
    cv::resizeWindow(winName, 900, 580);
    cv::setMouseCallback(winName, onMouse, &state);

    // Trackbar for feature type
    cv::createTrackbar("Feature", winName, NULL, 5, onTrackbar, &state);
    cv::setTrackbarPos("Feature", winName, 0);

    std::cout << "\n========================================" << std::endl;
    std::cout << "GUI Ready! Controls:" << std::endl;
    std::cout << "  Click any image to query it" << std::endl;
    std::cout << "  Trackbar or 1-6: switch feature type" << std::endl;
    std::cout << "  s: search by filename" << std::endl;
    std::cout << "  n/p: next/prev browser page" << std::endl;
    std::cout << "  q/ESC: quit" << std::endl;
    std::cout << "========================================\n"
              << std::endl;

    int lastFeatureIdx = -1;

    while (true)
    {
        // Check if feature type changed via trackbar
        if (state.featureIdx != lastFeatureIdx)
        {
            currentFeature = FEATURE_NAMES[state.featureIdx];
            lastFeatureIdx = state.featureIdx;
            needsUpdate = true;
        }

        if (needsUpdate)
        {
            // Check if database exists for current feature
            if (databases.find(currentFeature) == databases.end())
            {
                std::cout << "Warning: No features loaded for '" << currentFeature
                          << "'. Run: ./extract_features data/olympus/ "
                          << CSV_FILES[state.featureIdx] << " " << currentFeature << std::endl;

                // Show placeholder
                cv::Mat placeholder(400, 700, CV_8UC3, BG);
                std::string msg = "Features not loaded for: " + currentFeature;
                cv::putText(placeholder, msg, cv::Point(50, 180),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1);
                cv::putText(placeholder, "Run extract_features first, then restart GUI",
                            cv::Point(50, 220), cv::FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1);
                cv::imshow(winName, placeholder);
                needsUpdate = false;
                continue;
            }

            std::cout << "Query: " << currentTarget << " [" << currentFeature << "]" << std::endl;

            // Load target image
            std::string tPath = imageDir;
            if (tPath.back() != '/')
                tPath += '/';
            tPath += currentTarget;
            cv::Mat tImg = cv::imread(tPath);

            // Run query
            auto results = runQuery(currentTarget, currentFeature, imageDir,
                                    databases[currentFeature], dnnDb, tImg);

            // Build and show display
            cv::Mat display = buildDisplay(currentTarget, currentFeature, results,
                                           imageDir, allImages, browserPage, state);
            cv::imshow(winName, display);
            needsUpdate = false;
        }

        int key = cv::waitKey(50);

        // Mouse click
        if (state.clicked)
        {
            currentTarget = state.clickedFile;
            state.clicked = false;
            state.searchActive = false;
            state.searchText.clear();
            state.searchResults.clear();
            needsUpdate = true;
            continue;
        }

        // Keyboard handling
        if (state.searchActive)
        {
            // Search mode
            if (key == 27) // ESC - cancel search
            {
                state.searchActive = false;
                state.searchText.clear();
                state.searchResults.clear();
                needsUpdate = true;
            }
            else if (key == 13 || key == 10) // Enter - select first result
            {
                if (!state.searchResults.empty())
                {
                    currentTarget = state.searchResults[0];
                    state.searchActive = false;
                    state.searchText.clear();
                    state.searchResults.clear();
                    needsUpdate = true;
                }
            }
            else if (key == 8 || key == 127) // Backspace
            {
                if (!state.searchText.empty())
                {
                    state.searchText.pop_back();
                    state.searchResults.clear();
                    if (!state.searchText.empty())
                    {
                        for (auto &img : allImages)
                        {
                            if (img.find(state.searchText) != std::string::npos)
                            {
                                state.searchResults.push_back(img);
                                if (state.searchResults.size() >= 5)
                                    break;
                            }
                        }
                    }
                    needsUpdate = true;
                }
            }
            else if (key >= 32 && key <= 126) // Printable character
            {
                state.searchText += (char)key;
                state.searchResults.clear();
                for (auto &img : allImages)
                {
                    if (img.find(state.searchText) != std::string::npos)
                    {
                        state.searchResults.push_back(img);
                        if (state.searchResults.size() >= 5)
                            break;
                    }
                }
                needsUpdate = true;
            }
        }
        else
        {
            // Normal mode
            if (key == 'q' || key == 27)
                break;
            else if (key == 's')
            {
                state.searchActive = true;
                state.searchText.clear();
                state.searchResults.clear();
                needsUpdate = true;
            }
            else if (key == 'n')
            {
                browserPage = (browserPage + 1) % maxPages;
                needsUpdate = true;
            }
            else if (key == 'p')
            {
                browserPage = (browserPage - 1 + maxPages) % maxPages;
                needsUpdate = true;
            }
            else if (key >= '1' && key <= '6')
            {
                state.featureIdx = key - '1';
                cv::setTrackbarPos("Feature", winName, state.featureIdx);
            }
        }
    }

    cv::destroyAllWindows();
    std::cout << "GUI closed." << std::endl;
    return 0;
}