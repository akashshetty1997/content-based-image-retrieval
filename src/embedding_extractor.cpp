/*
 * Name: Akash Shridhar Shetty, Skandhan Madhusudhana
 * Date: February 2025
 * File: embedding_extractor.cpp
 *
 * Purpose:
 * Extension: Compute ResNet18 embeddings for all images using our own
 * ONNX model, rather than using the pre-computed CSV from the assignment.
 * This lets us compare our own embeddings vs the provided ones.
 *
 * Usage:
 *   ./compute_embeddings <model_path> <image_directory> <output_csv>
 *
 * Example:
 *   ./compute_embeddings data/resnet18-v2-7.onnx data/olympus/ data/my_dnn_features.csv
 */

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include "utils.h"

/**
 * Get embedding from ResNet18 for a single image
 *
 * @param src Source image (BGR)
 * @param embedding Output Mat (1x512 float)
 * @param net The loaded ResNet18 network
 * @return 0 on success, -1 on error
 *
 * What it does:
 *  1. Preprocess image: resize to 224x224, normalize with ImageNet mean/std
 *  2. Forward pass through network
 *  3. Extract output from the flatten layer (512-D embedding)
 */
int getEmbedding(const cv::Mat &src, cv::Mat &embedding, cv::dnn::Net &net)
{
    if (src.empty())
    {
        std::cerr << "Error: Empty image passed to getEmbedding" << std::endl;
        return -1;
    }

    const int NET_SIZE = 224;
    cv::Mat blob;

    // ImageNet preprocessing:
    // - Scale to [0,1] then normalize by std (0.226)
    // - Subtract mean (124, 116, 104) in BGR order
    // - Resize to 224x224
    // - Swap R and B channels (BGR -> RGB)
    cv::dnn::blobFromImage(src,
                           blob,
                           (1.0 / 255.0) * (1.0 / 0.226), // scale factor
                           cv::Size(NET_SIZE, NET_SIZE),  // target size
                           cv::Scalar(124, 116, 104),     // mean subtraction
                           true,                          // swapRB
                           false,                         // no center crop
                           CV_32F);                       // output type

    net.setInput(blob);

    // Forward pass to the flatten layer (512-D embedding)
    embedding = net.forward("onnx_node!resnetv22_flatten0_reshape0");

    return 0;
}

int main(int argc, char *argv[])
{
    // === Step 1: Parse arguments ===

    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_directory> <output_csv>" << std::endl;
        std::cerr << "\nExample:" << std::endl;
        std::cerr << "  " << argv[0] << " data/resnet18-v2-7.onnx data/olympus/ data/my_dnn_features.csv" << std::endl;
        return -1;
    }

    std::string modelPath = argv[1];
    std::string imageDir = argv[2];
    std::string outputCSV = argv[3];

    std::cout << "========================================" << std::endl;
    std::cout << "Custom DNN Embedding Extractor" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Model: " << modelPath << std::endl;
    std::cout << "Image directory: " << imageDir << std::endl;
    std::cout << "Output CSV: " << outputCSV << std::endl;
    std::cout << "========================================\n"
              << std::endl;

    // === Step 2: Load the network ===

    std::cout << "Loading ResNet18 model..." << std::endl;

    cv::dnn::Net net = cv::dnn::readNet(modelPath);

    if (net.empty())
    {
        std::cerr << "Error: Failed to load network from " << modelPath << std::endl;
        return -1;
    }

    std::cout << "Network loaded successfully" << std::endl;

    // Print layer names for verification
    std::vector<cv::String> layerNames = net.getLayerNames();
    std::cout << "Total layers: " << layerNames.size() << std::endl;
    std::cout << std::endl;

    // === Step 3: Get all image filenames ===

    std::vector<std::string> filenames;
    if (getImageFilenames(imageDir, filenames) != 0)
    {
        std::cerr << "Error: Failed to read image filenames" << std::endl;
        return -1;
    }

    std::cout << "Found " << filenames.size() << " images\n"
              << std::endl;

    // === Step 4: Extract embeddings for each image ===

    std::vector<FeatureData> allFeatures;
    allFeatures.reserve(filenames.size());

    int successCount = 0;
    int failCount = 0;

    std::cout << "Extracting embeddings..." << std::endl;

    for (size_t i = 0; i < filenames.size(); i++)
    {
        // Build full path
        std::string fullPath = imageDir;
        if (fullPath.back() != '/')
        {
            fullPath += '/';
        }
        fullPath += filenames[i];

        // Load image
        cv::Mat image = cv::imread(fullPath);
        if (image.empty())
        {
            std::cerr << "\nWarning: Failed to load " << filenames[i] << std::endl;
            failCount++;
            continue;
        }

        // Get embedding
        cv::Mat embedding;
        if (getEmbedding(image, embedding, net) != 0)
        {
            std::cerr << "\nWarning: Failed to get embedding for " << filenames[i] << std::endl;
            failCount++;
            continue;
        }

        // Convert cv::Mat embedding (1x512) to vector<float>
        FeatureData data;
        data.filename = filenames[i];
        data.feature.resize(embedding.cols);

        for (int j = 0; j < embedding.cols; j++)
        {
            data.feature[j] = embedding.at<float>(0, j);
        }

        allFeatures.push_back(data);
        successCount++;

        // Progress
        if ((i + 1) % 50 == 0 || (i + 1) == filenames.size())
        {
            std::cout << "\rProgress: " << (i + 1) << "/" << filenames.size() << std::flush;
        }
    }

    std::cout << "\n"
              << std::endl;

    // === Step 5: Summary ===

    std::cout << "========================================" << std::endl;
    std::cout << "Extraction Summary:" << std::endl;
    std::cout << "  Total images: " << filenames.size() << std::endl;
    std::cout << "  Success: " << successCount << std::endl;
    std::cout << "  Failed: " << failCount << std::endl;
    if (!allFeatures.empty())
    {
        std::cout << "  Embedding size: " << allFeatures[0].feature.size() << " values" << std::endl;
    }
    std::cout << "========================================\n"
              << std::endl;

    // === Step 6: Write to CSV ===

    std::cout << "Writing embeddings to CSV..." << std::endl;

    if (writeFeaturesToCSV(outputCSV, allFeatures) != 0)
    {
        std::cerr << "Error: Failed to write CSV" << std::endl;
        return -1;
    }

    std::cout << "Saved to: " << outputCSV << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Done! You can now query with:" << std::endl;
    std::cout << "  ./query <target> " << outputCSV << " 3 dnn" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}