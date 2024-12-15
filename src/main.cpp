#include "ORBextractor.h"
#include <fstream>
#include <iostream>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iomanip>
using namespace cv;
using namespace std;
using namespace ORB_SLAM2;
void GrabImage(cv::Mat &mImGray, int mbRGB)
{
    // 调整图像大小为 640x480
    resize(mImGray, mImGray, Size(640, 480));

    if(mImGray.channels()==3)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
        }
    }

}

// 评估特征点平均性的函数
double evaluateKeypointUniformity(const vector<KeyPoint>& keypoints, const Size& imageSize, Mat& image, int gridRows = 10, int gridCols = 10) {
    // 创建一个网格来统计每个网格中的特征点数量
    vector<vector<int>> grid(gridRows, vector<int>(gridCols, 0));
    int totalKeypoints = keypoints.size();

    // 计算每个网格的宽度和高度
    double cellWidth = static_cast<double>(imageSize.width) / gridCols;
    double cellHeight = static_cast<double>(imageSize.height) / gridRows;

    // 统计每个网格中的特征点数量
    for (const auto& kp : keypoints) {
        int col = min(static_cast<int>(kp.pt.x / cellWidth), gridCols - 1);
        int row = min(static_cast<int>(kp.pt.y / cellHeight), gridRows - 1);
        grid[row][col]++;
    }

    // 计算每个网格中特征点数量的均方根
    double mean = static_cast<double>(totalKeypoints) / (gridRows * gridCols);
    double sumSquares = 0.0;
    for (const auto& row : grid) {
        for (int count : row) {
            sumSquares += (count - mean) * (count - mean);
        }
    }
    double rms = sqrt(sumSquares / (gridRows * gridCols));

    // 可视化网格和特征点数量
    Mat gridImage = image.clone();
    for (int i = 0; i < gridRows; ++i) {
        for (int j = 0; j < gridCols; ++j) {
            Rect cellRect(j * cellWidth, i * cellHeight, cellWidth, cellHeight);
            // 计算颜色强度
            double intensity = min(1.0, grid[i][j] / mean);
            Scalar color(0, 255 * intensity, 0, 128); 
            Mat roi = gridImage(cellRect);
            Mat colorMat(cellRect.size(), CV_8UC3, color);
            addWeighted(colorMat, 0.5, roi, 0.5, 0, roi);
            rectangle(gridImage, cellRect, Scalar(255, 255, 255), 1);
            putText(gridImage, to_string(grid[i][j]), Point(j * cellWidth + 5, i * cellHeight + 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1.8, LINE_AA);
        }
    }

    // 加上rms
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(3) << rms;
    string uniformityText = "Uniformity (RMS): " + stream.str();
    putText(gridImage, uniformityText, Point(imageSize.width - 300, imageSize.height - 15), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2, LINE_AA);

    // 显示网格图像
    namedWindow("Keypoint Distribution", WINDOW_NORMAL);
    imshow("Keypoint Distribution", gridImage);
    waitKey(0);
    imwrite("orb_features_uniformity.jpg", gridImage);

    // 返回均方根，均方根越小，特征点分布越均匀
    return rms;
}

// 评估特征点平均性的函数
double evaluateKeypointUniformity_(const vector<KeyPoint>& keypoints, const Size& imageSize, Mat& image, int gridRows = 10, int gridCols = 10) {
    // 创建一个网格来统计每个网格中的特征点数量
    vector<vector<int>> grid(gridRows, vector<int>(gridCols, 0));
    int totalKeypoints = keypoints.size();

    // 计算每个网格的宽度和高度
    double cellWidth = static_cast<double>(imageSize.width) / gridCols;
    double cellHeight = static_cast<double>(imageSize.height) / gridRows;

    // 统计每个网格中的特征点数量
    for (const auto& kp : keypoints) {
        int col = min(static_cast<int>(kp.pt.x / cellWidth), gridCols - 1);
        int row = min(static_cast<int>(kp.pt.y / cellHeight), gridRows - 1);
        grid[row][col]++;
    }

    // 计算每个网格中特征点数量的均方根
    double mean = static_cast<double>(totalKeypoints) / (gridRows * gridCols);
    double sumSquares = 0.0;
    for (const auto& row : grid) {
        for (int count : row) {
            sumSquares += (count - mean) * (count - mean);
        }
    }
    double rms = sqrt(sumSquares / (gridRows * gridCols));

    // 可视化网格和特征点数量
    Mat gridImage = image.clone();
    for (int i = 0; i < gridRows; ++i) {
        for (int j = 0; j < gridCols; ++j) {
            Rect cellRect(j * cellWidth, i * cellHeight, cellWidth, cellHeight);
            // 计算颜色强度
            double intensity = min(1.0, grid[i][j] / mean);
            Scalar color(0, 255 * intensity, 0, 128); 
            Mat roi = gridImage(cellRect);
            Mat colorMat(cellRect.size(), CV_8UC3, color);
            addWeighted(colorMat, 0.5, roi, 0.5, 0, roi);
            rectangle(gridImage, cellRect, Scalar(255, 255, 255), 1);
            putText(gridImage, to_string(grid[i][j]), Point(j * cellWidth + 5, i * cellHeight + 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1.8, LINE_AA);
        }
    }

    // 加上rms
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(3) << rms;
    string uniformityText = "Uniformity (RMS): " + stream.str();
    putText(gridImage, uniformityText, Point(imageSize.width - 300, imageSize.height - 15), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2, LINE_AA);

    // 显示网格图像
    // namedWindow("Keypoint Distribution", WINDOW_NORMAL);
    // imshow("Keypoint Distribution", gridImage);
    // waitKey(0);
    // imwrite("orb_features_uniformity.jpg", gridImage);

    // 返回均方根，均方根越小，特征点分布越均匀
    return rms;
}

// 提取特征点并显示
void extractFeatures(const Mat& imGray, const Mat& imLeft, ORBextractor* mpORBextractorLeft, vector<KeyPoint>& mvKeys, Mat &mDescriptors) {
    // orb-slam2中的orb特征提取器
    (*mpORBextractorLeft)(imGray, Mat(), mvKeys, mDescriptors);

    // diaplay
    Mat out1;
    drawKeypoints(imLeft, mvKeys, out1, Scalar::all(-1), DrawMatchesFlags::DEFAULT); 

    // 评估特征点平均性
    double uniformity = evaluateKeypointUniformity(mvKeys, imGray.size(), out1, 5, 5);
    cout << "Keypoint uniformity (RMS): " << uniformity << endl;

    namedWindow("orb_features", WINDOW_NORMAL);
    imshow("orb_features", out1);
    waitKey(0);
    imwrite("orb_features.jpg", out1);
}

// 提取特征点并显示
void extractFeaturesCV(const Mat& imGray,  vector<KeyPoint>& mvKeys, Mat &mDescriptors) {
    // opencv默认的orb特征提取器
    cv::Ptr<cv::ORB> orb = cv::ORB::create(2000);
    orb->detectAndCompute(imGray, noArray(), mvKeys, mDescriptors);

    // diaplay
    Mat out1;
    drawKeypoints(imGray, mvKeys, out1, Scalar::all(-1), DrawMatchesFlags::DEFAULT); 

    // 评估特征点平均性
    double uniformity = evaluateKeypointUniformity(mvKeys, imGray.size(), out1, 5, 5);
    cout << "Keypoint uniformity (RMS): " << uniformity << endl;

    namedWindow("orb_features", WINDOW_NORMAL);
    imshow("orb_features", out1);
    waitKey(0);
    imwrite("orb_features.jpg", out1);
}

// 提取特征点并显示
void extractFeatures_(const Mat& imGray, const Mat& imLeft, ORBextractor* mpORBextractorLeft, vector<KeyPoint>& mvKeys, Mat &mDescriptors) {
    // orb-slam2中的orb特征提取器
    (*mpORBextractorLeft)(imGray, Mat(), mvKeys, mDescriptors);

    // diaplay
    Mat out1;
    drawKeypoints(imLeft, mvKeys, out1, Scalar::all(-1), DrawMatchesFlags::DEFAULT); 

    // 评估特征点平均性
    double uniformity = evaluateKeypointUniformity_(mvKeys, imGray.size(), out1, 5, 5);
    cout << "Keypoint uniformity (RMS): " << uniformity << endl;

    // namedWindow("orb_features", WINDOW_NORMAL);
    // imshow("orb_features", out1);
    // waitKey(0);
    // imwrite("orb_features.jpg", out1);
}

// 提取特征点并显示
void extractFeaturesCV_(const Mat& imGray,  vector<KeyPoint>& mvKeys, Mat &mDescriptors) {
    // opencv默认的orb特征提取器
    cv::Ptr<cv::ORB> orb = cv::ORB::create(2000);
    orb->detectAndCompute(imGray, noArray(), mvKeys, mDescriptors);

    // diaplay
    Mat out1;
    drawKeypoints(imGray, mvKeys, out1, Scalar::all(-1), DrawMatchesFlags::DEFAULT); 

    // 评估特征点平均性
    double uniformity = evaluateKeypointUniformity_(mvKeys, imGray.size(), out1, 5, 5);
    cout << "Keypoint uniformity (RMS): " << uniformity << endl;

    // namedWindow("orb_features", WINDOW_NORMAL);
    // imshow("orb_features", out1);
    // waitKey(0);
    // imwrite("orb_features.jpg", out1);
}

// 匹配特征点的函数
void matchFeatures(const cv::Mat& descriptors1, const cv::Mat& descriptors2, std::vector<cv::DMatch>& matches) {
    // 使用暴力匹配器进行特征点匹配
    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
    matcher.match(descriptors1, descriptors2, matches);
    // 输出匹配结果
    std::cout << "Number of matches: " << matches.size() << std::endl;
}

void computeHomographyWithRANSAC(const std::vector<cv::KeyPoint>& keypoints1, 
                                 const std::vector<cv::KeyPoint>& keypoints2, 
                                 const std::vector<cv::DMatch>& matches, 
                                 cv::Mat& homography, std::vector<cv::DMatch>& inliers) {
    // 提取匹配点
    std::vector<cv::Point2f> points1, points2;
    for (const auto& match : matches) {
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }

    // 使用RANSAC计算单应矩阵，并获取内点掩码
    std::vector<uchar> inlierMask;
    homography = cv::findHomography(points1, points2, cv::RANSAC, 3, inlierMask);

    // 筛选内点对
    inliers.clear();
    for (size_t i = 0; i < matches.size(); ++i) {
        if (inlierMask[i]) {
            inliers.push_back(matches[i]);
        }
    }
    std::cout << "Number of inliers: " << inliers.size() << std::endl;
}

// 评估误差函数
double evaluateHomographyError(const cv::Mat& H_gt, const cv::Mat& H_pred, int width, int height) {
    // 定义图像的4个角点
    std::vector<cv::Point2f> corners = { {0, 0}, {0, static_cast<float>(width - 1)}, {static_cast<float>(height - 1), 0}, {static_cast<float>(height - 1), static_cast<float>(width - 1)} };
    
    // 变换角点
    std::vector<cv::Point2f> real_warped_corners, warped_corners;
    cv::perspectiveTransform(corners, real_warped_corners, H_gt);
    cv::perspectiveTransform(corners, warped_corners, H_pred);
    
    // 计算误差
    double mean_dist = 0.0;
    for (size_t i = 0; i < corners.size(); ++i) {
        mean_dist += cv::norm(real_warped_corners[i] - warped_corners[i]);
    }
    mean_dist /= corners.size();
    
    return mean_dist;
}

int main(int argc, char **argv)
{
    if(argc != 3)
    {
        cerr << endl << "Usage: ./ORBextractor path_to_a_picture path to orb.yaml" << endl;
        return 1;
    }
    string strSettingPath = argv[2];

    // 载入参数文件
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    // 1:RGB 0:BGR
    int nRGB = fSettings["Camera.RGB"];

    if(nRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // 每一帧提取的特征点数 1000
    int nFeatures = fSettings["ORBextractor.nFeatures"];
    // 图像建立金字塔时的变化尺度 1.2
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    // 尺度金字塔的层数 8
    int nLevels = fSettings["ORBextractor.nLevels"];
    // 提取fast特征点的默认阈值 20
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    // 如果默认阈值提取不出足够fast特征点，则使用最小阈值 8
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];
    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;
    
//******************************************************************************************************************************************   
       // 获取文件夹路径
    std::string folderPath = argv[1];

    // 构建文件路径
    std::string img1Path = folderPath + "/img1.ppm";
    std::string img2Path = folderPath + "/img2.ppm";
    std::string H1to2pPath = folderPath + "/H1to2p";

    // 初始化mpORBextractorLeft作为特征点提取器  
    ORBextractor* mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

//*******************************img1***************************************************************************************************************
    // 载入原图
    cv::Mat imLeft1;   
    imLeft1 = cv::imread(img1Path,CV_LOAD_IMAGE_UNCHANGED);
    if( imLeft1.empty() )                      
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    // 转为灰度图
    cv::Mat &mImGray1 = imLeft1;
    GrabImage(mImGray1,nRGB);
 
    // 当前帧图像中提取的特征点集合
    std::vector<cv::KeyPoint> mvKeys1;
    // 特征点对应的描述子
    cv::Mat mDescriptors1;

    // extractFeaturesCV(mImGray1, mvKeys1, mDescriptors1);
    extractFeatures(mImGray1, imLeft1, mpORBextractorLeft, mvKeys1, mDescriptors1);


//*******************************img2***************************************************************************************************************
     cv::Mat imLeft2;   
    imLeft2 = cv::imread(img2Path,CV_LOAD_IMAGE_UNCHANGED);
    // 转为灰度图
    cv::Mat &mImGray2 = imLeft2;
    GrabImage(mImGray2,nRGB);

    // 当前帧图像中提取的特征点集合
    std::vector<cv::KeyPoint> mvKeys2;
    // 特征点对应的描述子
    cv::Mat mDescriptors2;

    // extractFeaturesCV_(mImGray2, mvKeys2, mDescriptors2);
    extractFeatures_(mImGray2, imLeft2, mpORBextractorLeft, mvKeys2, mDescriptors2);

// -----------------------------特征点匹配-------------------------------------
    // 匹配特征点
    std::vector<cv::DMatch> matches;
    matchFeatures(mDescriptors1, mDescriptors2, matches);

    //  // 可视化匹配结果
    // cv::Mat imgMatches;
    // cv::drawMatches(imLeft1, mvKeys1, imLeft2, mvKeys2, matches, imgMatches);
    // // 显示匹配结果
    // cv::imshow("Matches", imgMatches);
    // cv::waitKey(0);

    // // 保存匹配结果
    // cv::imwrite(folderPath + "/matches.png", imgMatches);

// -----------------------------计算单应矩阵并评估-------------------------------------
    // 计算单应矩阵
    cv::Mat H_pred;
    std::vector<cv::DMatch> inliers;
    computeHomographyWithRANSAC(mvKeys1, mvKeys2, matches, H_pred, inliers);
    std::cout << "estimated Homography Matrix: " << H_pred << std::endl;

    // 读取真实单应矩阵     
    cv::Mat H_gt = cv::Mat::eye(3, 3, CV_64F);
    std::ifstream file(H1to2pPath);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            file >> H_gt.at<double>(i, j);
        }
    }
    file.close();
    std::cout << "ground truth Homography Matrix: " << H_gt << std::endl;

    // 评估误差
    double mean_dist = evaluateHomographyError(H_gt, H_pred, 640, 480);
    std::cout << "mean distance between real and estimated corners: " << mean_dist << std::endl;
    
    // 可视化匹配结果
    cv::Mat imgMatchesUsed;
    cv::drawMatches(imLeft1, mvKeys1, imLeft2, mvKeys2, inliers, imgMatchesUsed);
    // 在图像右下角添加内点对数量和单应性误差
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(3) << mean_dist;
    std::string inliersText = "Inliers Number: " + std::to_string(inliers.size());
    std::string errorText = "Homography Estimation Error: " +  stream.str();
    cv::putText(imgMatchesUsed, inliersText, cv::Point(imgMatchesUsed.cols - 273, imgMatchesUsed.rows - 42), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2, LINE_AA);
    cv::putText(imgMatchesUsed, errorText, cv::Point(imgMatchesUsed.cols - 440, imgMatchesUsed.rows - 15), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2, LINE_AA);
    // 显示匹配结果
    cv::imshow("Matches", imgMatchesUsed);
    cv::waitKey(0);
    // 保存匹配结果
    cv::imwrite(folderPath + "/used_matches.png", imgMatchesUsed);

    return 0;
}    