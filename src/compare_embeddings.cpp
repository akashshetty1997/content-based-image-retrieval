/*
 * Name: Akash Shridhar Shetty, Skandhan Madhusudhana
 * Date: February 2025
 * File: compare_embeddings.cpp
 *
 * Purpose:
 * Extension: Generate side-by-side comparison image of provided vs custom DNN embeddings.
 * Saves comparison images to results/ folder for the report.
 *
 * Usage:
 *   ./compare_embeddings <image_dir> <provided_csv> <custom_csv>
 *
 * Example:
 *   ./compare_embeddings data/olympus/ data/ResNet18_olym.csv data/my_dnn_features.csv
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include "distance.h"
#include "utils.h"

const int THUMB_W = 180;
const int THUMB_H = 135;
const int PAD = 10;
const cv::Scalar BG(30, 30, 30);
const cv::Scalar WHITE(255, 255, 255);
const cv::Scalar GRAY(160, 160, 160);
const cv::Scalar HEADER(0, 200, 255);
const cv::Scalar PROVIDED_COLOR(0, 255, 0);
const cv::Scalar CUSTOM_COLOR(255, 165, 0);
const cv::Scalar TARGET_COLOR(0, 255, 255);

cv::Mat makeThumbnail(const cv::Mat &src, int w, int h)
{
    if (src.empty())
    {
        cv::Mat blank(h, w, CV_8UC3, cv::Scalar(50, 50, 50));
        cv::putText(blank, "N/A", cv::Point(w / 3, h / 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1);
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

void drawBorder(cv::Mat &img, const cv::Scalar &color, int t = 3)
{
    cv::rectangle(img, cv::Point(0, 0), cv::Point(img.cols - 1, img.rows - 1), color, t);
}

std::vector<MatchResult> queryDNN(const std::string &targetFile,
                                   const std::vector<FeatureData> &db)
{
    std::vector<MatchResult> results;
    std::vector<float> tFeat;

    for (const auto &d : db)
        if (d.filename == targetFile)
        {
            tFeat = d.feature;
            break;
        }

    if (tFeat.empty())
        return results;

    for (size_t i = 0; i < db.size(); i++)
    {
        if (db[i].filename == targetFile)
            continue;
        float dist = distanceCosine(tFeat, db[i].feature);
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

cv::Mat buildComparisonImage(const std::string &targetFile,
                              const std::string &imageDir,
                              const std::vector<MatchResult> &providedResults,
                              const std::vector<MatchResult> &customResults,
                              int numMatches)
{
    int cols = 1 + numMatches;
    int rows = 2;
    int cellW = THUMB_W + PAD;
    int cellH = THUMB_H + 35;
    int headerH = 50;
    int rowLabelW = 120;

    int canvasW = rowLabelW + cols * cellW + PAD;
    int canvasH = headerH + rows * cellH + PAD * 2;

    cv::Mat canvas(canvasH, canvasW, CV_8UC3, BG);

    // Header
    std::string title = "DNN Embedding Comparison: " + targetFile;
    cv::putText(canvas, title, cv::Point(PAD, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, HEADER, 2);

    // Row labels
    cv::putText(canvas, "Provided", cv::Point(PAD, headerH + cellH / 2 + 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, PROVIDED_COLOR, 1);
    cv::putText(canvas, "CSV", cv::Point(PAD, headerH + cellH / 2 + 22),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, PROVIDED_COLOR, 1);

    cv::putText(canvas, "Custom", cv::Point(PAD, headerH + cellH + cellH / 2 + 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, CUSTOM_COLOR, 1);
    cv::putText(canvas, "ONNX", cv::Point(PAD, headerH + cellH + cellH / 2 + 22),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, CUSTOM_COLOR, 1);

    // Target image path
    std::string tPath = imageDir;
    if (tPath.back() != '/')
        tPath += '/';
    tPath += targetFile;
    cv::Mat tImg = cv::imread(tPath);

    for (int row = 0; row < 2; row++)
    {
        int y = headerH + row * cellH;
        int x = rowLabelW;

        // Target
        cv::Mat tThumb = makeThumbnail(tImg, THUMB_W, THUMB_H);
        drawBorder(tThumb, TARGET_COLOR, 3);
        tThumb.copyTo(canvas(cv::Rect(x, y, THUMB_W, THUMB_H)));
        cv::putText(canvas, "TARGET", cv::Point(x + THUMB_W / 2 - 30, y + THUMB_H + 14),
                    cv::FONT_HERSHEY_SIMPLEX, 0.35, TARGET_COLOR, 1);

        // Matches
        const auto &res = (row == 0) ? providedResults : customResults;
        cv::Scalar borderColor = (row == 0) ? PROVIDED_COLOR : CUSTOM_COLOR;

        for (int m = 0; m < numMatches && m < (int)res.size(); m++)
        {
            int mx = rowLabelW + (m + 1) * cellW;

            std::string mPath = imageDir;
            if (mPath.back() != '/')
                mPath += '/';
            mPath += res[m].filename;

            cv::Mat mImg = cv::imread(mPath);
            cv::Mat mThumb = makeThumbnail(mImg, THUMB_W, THUMB_H);
            drawBorder(mThumb, borderColor, 2);
            mThumb.copyTo(canvas(cv::Rect(mx, y, THUMB_W, THUMB_H)));

            // Label
            char label[64];
            snprintf(label, sizeof(label), "#%d %s", m + 1, res[m].filename.c_str());
            cv::putText(canvas, label, cv::Point(mx, y + THUMB_H + 12),
                        cv::FONT_HERSHEY_SIMPLEX, 0.28, WHITE, 1);

            char dStr[32];
            snprintf(dStr, sizeof(dStr), "d=%.4f", res[m].distance);
            cv::putText(canvas, dStr, cv::Point(mx, y + THUMB_H + 25),
                        cv::FONT_HERSHEY_SIMPLEX, 0.26, GRAY, 1);
        }
    }

    return canvas;
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <image_dir> <provided_csv> <custom_csv>" << std::endl;
        std::cerr << "\nExample:" << std::endl;
        std::cerr << "  " << argv[0] << " data/olympus/ data/ResNet18_olym.csv data/my_dnn_features.csv" << std::endl;
        return -1;
    }

    std::string imageDir = argv[1];
    std::string providedCSV = argv[2];
    std::string customCSV = argv[3];

    std::cout << "========================================" << std::endl;
    std::cout << "DNN Embedding Comparison" << std::endl;
    std::cout << "========================================" << std::endl;

    // Load databases
    std::vector<FeatureData> providedDb, customDb;

    std::cout << "Loading provided embeddings..." << std::endl;
    if (readFeaturesFromCSV(providedCSV, providedDb) != 0 || providedDb.empty())
    {
        std::cerr << "Error: Failed to load provided CSV" << std::endl;
        return -1;
    }
    std::cout << "  Loaded " << providedDb.size() << " vectors (" << providedDb[0].feature.size() << "D)" << std::endl;

    std::cout << "Loading custom embeddings..." << std::endl;
    if (readFeaturesFromCSV(customCSV, customDb) != 0 || customDb.empty())
    {
        std::cerr << "Error: Failed to load custom CSV" << std::endl;
        return -1;
    }
    std::cout << "  Loaded " << customDb.size() << " vectors (" << customDb[0].feature.size() << "D)" << std::endl;

    // Query images to compare
    std::vector<std::string> queryImages = {"pic.0893.jpg", "pic.0164.jpg", "pic.1072.jpg"};
    int numMatches = 3;

    // Create results directory
    system("mkdir -p results");

    for (const auto &query : queryImages)
    {
        std::cout << "\nComparing: " << query << std::endl;

        auto providedResults = queryDNN(query, providedDb);
        auto customResults = queryDNN(query, customDb);

        std::cout << "  Provided top 3: ";
        for (int i = 0; i < 3 && i < (int)providedResults.size(); i++)
            std::cout << providedResults[i].filename << " (" << providedResults[i].distance << ") ";
        std::cout << std::endl;

        std::cout << "  Custom top 3:   ";
        for (int i = 0; i < 3 && i < (int)customResults.size(); i++)
            std::cout << customResults[i].filename << " (" << customResults[i].distance << ") ";
        std::cout << std::endl;

        // Build comparison image
        cv::Mat comparison = buildComparisonImage(query, imageDir,
                                                   providedResults, customResults, numMatches);

        // Save to file
        // Extract number from filename like pic.0893.jpg -> 0893
        std::string num = query.substr(4, 4);
        std::string outFile = "results/comparison_" + num + ".png";

        cv::imwrite(outFile, comparison);
        std::cout << "  Saved: " << outFile << std::endl;

        // Also show in window
        cv::imshow("DNN Comparison: " + query, comparison);
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Comparison complete! Images saved to results/" << std::endl;
    std::cout << "Press any key to close windows." << std::endl;
    std::cout << "========================================" << std::endl;

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}