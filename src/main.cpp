#include "ORBextractor.h"
#include <fstream>
#include <iostream>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
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
            Scalar color(0, 255 * intensity, 0, 128); // 红色，半透明
            Mat roi = gridImage(cellRect);
            Mat colorMat(cellRect.size(), CV_8UC3, color);
            addWeighted(colorMat, 0.5, roi, 0.5, 0, roi);
            rectangle(gridImage, cellRect, Scalar(255, 255, 255), 1);
            putText(gridImage, to_string(grid[i][j]), Point(j * cellWidth + 5, i * cellHeight + 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
        }
    }

    // 显示网格图像
    namedWindow("Keypoint Distribution", WINDOW_NORMAL);
    imshow("Keypoint Distribution", gridImage);
    waitKey(0);

    // 返回均方根，均方根越小，特征点分布越均匀
    return rms;
}


int main(int argc, char **argv)
{
    if(argc != 3)
    {
        cerr << endl << "Usage: ./ORBextractor path_to_a_picture named liu.png path to orb.yaml" << endl;
        return 1;
    }
    cout << "liu";
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
    // 初始化mpORBextractorLeft作为特征点提取器  
    ORBextractor* mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
   
    // 载入原图
    cv::Mat imLeft;   
    imLeft = cv::imread(argv[1],CV_LOAD_IMAGE_UNCHANGED);
    if( imLeft.empty() )                      
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    // 转为灰度图
    cv::Mat &mImGray = imLeft;
    GrabImage(mImGray,nRGB);
 
    // 当前帧图像中提取的特征点集合
    std::vector<cv::KeyPoint> mvKeys;
    // 特征点对应的描述子
    cv::Mat mDescriptors;

    // orb-slam2中的orb特征提取器
    (*mpORBextractorLeft)(mImGray,cv::Mat(),mvKeys,mDescriptors);

    // opencv默认的orb特征提取器
    // cv::Ptr<cv::ORB> orb = cv::ORB::create(2000);
    // orb->detectAndCompute(mImGray, noArray(), mvKeys, mDescriptors);

    
    // diaplay
    Mat out1;
    drawKeypoints(imLeft, mvKeys, out1, Scalar::all(-1), DrawMatchesFlags::DEFAULT); 

    // 评估特征点平均性
    double uniformity = evaluateKeypointUniformity(mvKeys, mImGray.size(), out1, 5, 5);
    cout << "Keypoint uniformity (RMS): " << uniformity << endl;

    // drawKeypoints(imLeft, mvKeys, out1, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS); 
    namedWindow("orb_features", WINDOW_NORMAL);
    imshow("orb_features", out1);
    waitKey(0);
    imwrite("orb_features.jpg", out1);
    return 0;

}    