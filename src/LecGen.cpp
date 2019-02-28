#include "LecGen.h"

using namespace cv;
using std::cout;
using std::endl;

LecGen::LecGen(std::string path, std::string videoName, char save_folder)
{
    mPath = path;
    mVideoName = videoName;
    mFolder = save_folder;
}
/*

PSNR from OpenCV tutorial

*/
bool LecGen::PSNR(const Mat& I1, const Mat& I2)
{
    Mat s1;
    absdiff(I1, I2, s1);       // |I1 - I2|
    s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
    s1 = s1.mul(s1);           // |I1 - I2|^2

    Scalar s = sum(s1);         // sum elements per channel

    double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

    if (sse <= 1e-9) // for small values return zero
        return true;
    else
    {
        double  mse = sse / (double)(I1.channels() * I1.total());
        double psnr = 10.0*log10((255 * 255) / mse);
        cout << " psnr: " << psnr << endl;
        return (psnr < 30) ? false : true;
    }
}


/*

MSSIM from OpenCV tutorial

*/

Scalar LecGen::getMSSIM(const Mat& i1, const Mat& i2)
{
    const double C1 = 6.5025, C2 = 58.5225;
    /***************************** INITS **********************************/
    int d = CV_32F;

    Mat I1, I2;
    i1.convertTo(I1, d);           // cannot calculate on one byte large values
    i2.convertTo(I2, d);

    Mat I2_2 = I2.mul(I2);        // I2^2
    Mat I1_2 = I1.mul(I1);        // I1^2
    Mat I1_I2 = I1.mul(I2);        // I1 * I2

    /***********************PRELIMINARY COMPUTING ******************************/

    Mat mu1, mu2;   //
    GaussianBlur(I1, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, Size(11, 11), 1.5);

    Mat mu1_2 = mu1.mul(mu1);
    Mat mu2_2 = mu2.mul(mu2);
    Mat mu1_mu2 = mu1.mul(mu2);

    Mat sigma1_2, sigma2_2, sigma12;

    GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;

    GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;

    GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    ///////////////////////////////// FORMULA ////////////////////////////////
    Mat t1, t2, t3;

    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

    Mat ssim_map;
    divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;

    Scalar mssim = mean(ssim_map); // mssim = average of ssim map
    return mssim;
}

bool LecGen::SSIM(const cv::Mat& mat1, const cv::Mat& mat2) {

    Scalar mssim = getMSSIM(mat1, mat2);
    if (mssim.val[0] > 0.9 && mssim.val[1] > 0.9 && mssim.val[2] > 0.9) {

        cout << "(R, G & B SSIM index)" << endl;
        cout << mssim.val[2] * 100 << "%" << endl;
        cout << mssim.val[1] * 100 << "%" << endl;
        cout << mssim.val[0] * 100 << "%" << endl;
        return true;
    }
    else {
        return false;
    }


}

/*

Primitive Similarity Test

*/
bool LecGen::framesSimilar(const cv::Mat& mat1, const cv::Mat& mat2) {

    cv::Mat gray1, gray2;
    cvtColor(mat1, gray1, CV_BGR2GRAY);
    cvtColor(mat2, gray2, CV_BGR2GRAY);

    // treat two empty mat as identical as well
    if (mat1.empty() && mat2.empty()) {
        return true;
    }
    // if dimensionality of two mat is not identical, these two mat is not identical

    if (mat1.cols != mat2.cols || mat1.rows != mat2.rows || mat1.dims != mat2.dims) {
        return false;
    }
    cv::Mat diff;
    cv::compare(gray1, gray2, diff, cv::CMP_NE);
    int nonzero = cv::countNonZero(diff);
    int total = gray2.rows*gray2.cols;
    double difference = nonzero / (double)total;

    // io for debugging
    if (difference > 0.1) {
        std::cout << "nonzero: " << nonzero << " " << " Total: " << total << " difference: " << difference << std::endl;
        return false;
    }
    else {
        return true;
    }
}

bool LecGen::frameExists(std::vector<cv::Mat>& vec, cv::Mat& frame) {

    bool existance = false;
    for (const auto& i : vec)
    {
        //existance=framesSimilar(i, frame);
        existance = PSNR(i, frame);
        //existance = SSIM(i, frame);
        if (existance)
        {
            std::cout << "page exist" << std::endl;
            break;
        }
    }
    return existance;
}

void LecGen::generate() {

    cv::VideoCapture cap(mPath);
    cap.set(CV_CAP_PROP_FPS, 60);

    if (!cap.isOpened())
    {
        std::cerr << "ERROR: Could not open video " <<mVideoName<< std::endl;
        return;
    }


    int frame_count = 0;
    bool should_stop = false;
    cv::Mat frameM;
    cap >> frameM;

    char cover[128];
    sprintf_s(cover, mFolder+"\\frame_01.jpg");
    cv::imwrite(cover, frameM);
    std::vector<cv::Mat> savedFrame;
    savedFrame.push_back(frameM);

    while (!should_stop)
    {
        cv::Mat frame;
        cap >> frame; //get a new frame from the video

        if (frame.empty())
        {
            should_stop = true; //we arrived to the end of the video
            continue;
        }

        // bool similar=framesSimilar(frameM,frame);
        bool similar = PSNR(frameM, frame);
        // bool similar = SSIM(frameM, frame);

        char filename[256];
        if (!similar) {
            std::cout << "page find" << std::endl;
            bool exist = frameExists(savedFrame, frame);
            if (!exist)
            {
                sprintf_s(filename, mFolder+"\\frame_%06d.jpg", frame_count);
                cv::imwrite(filename, frame);
                savedFrame.push_back(frame);
                std::cout << "page saved" << std::endl;
            }
        }
        frame_count++;
        frameM = frame.clone();
    }

}
