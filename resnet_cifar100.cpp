#include <cstdio>
#include <dlpack/dlpack.h>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <filesystem>
#include <regex>

using namespace std::chrono;
namespace fs = std::filesystem;

void Mat_to_CHW(tvm::runtime::NDArray &data, cv::Mat &frame)
{
    // assert(data && !frame.empty());
    float mean[3] = {0.5, 0.5, 0.5};
    float stddev[3] = {0.5, 0.5, 0.5};
    int channels = frame.channels();
    int height = frame.rows;
    int width = frame.cols;
    for (int c = 0; c < channels; ++c)
    {
        for (int h = 0; h < height; ++h)
        {

            for (int w = 0; w < width; ++w)
            {
                int index = c * height * width + h * width + w;
                // std::cout<<img_data.at<cv::Vec3f>(h, w)[c]<<" ";
                static_cast<float *>(data->data)[index] = (frame.at<cv::Vec3f>(h, w)[c] - mean[c]) / stddev[c];
                // std::cout<<input_image_[index]<<std::endl;
                // std::cin>>a;
            }
            // std::cout<<std::endl;
        }
    }
}

int main()
{
    tvm::runtime::Module mod_dylib = tvm::runtime::Module::LoadFromFile("../tvm_output_lib/resnet_cifar100_tuned.so");
    DLDevice dev{kDLCPU, 0};
    tvm::runtime::NDArray x = tvm::runtime::NDArray::Empty({1, 3, 224, 224}, DLDataType{kDLFloat, 32, 1}, dev);
    tvm::runtime::NDArray y = tvm::runtime::NDArray::Empty({1, 100}, DLDataType{kDLFloat, 32, 1}, dev);
    // get the function from the module(set input data)

    tvm::runtime::Module gmod = mod_dylib.GetFunction("default")(dev);
    tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
    tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
    tvm::runtime::PackedFunc run = gmod.GetFunction("run");
    auto total_duration = microseconds(0);
    int num = 0;
    int correct_num = 0;
    std::string folderPath = "/dataset/cifar-100-python/testdir/";
    for (const auto &entry : fs::directory_iterator(folderPath))
    {
        std::string imgPath = entry.path().string();
        std::regex labelRegex("(\\d+)_(\\d+).png");
        std::smatch match;
        if (std::regex_search(imgPath, match, labelRegex))
        {
            // match[1] 匹配 x，match[2] 匹配 yyyy
            std::string labelStr = match[1];
            std::string imageNumberStr = match[2];
            // std::cout<<match[1]<<std::endl;
            int label = std::stoi(labelStr);
            int imageNumber = std::stoi(imageNumberStr);
            // int label = 0;
            // std::cin>>label;
            /*
            if(imageNumber!=8870){
                continue;
            }
            */
            num++;
            cv::Mat img = cv::imread(imgPath, -1);
            cv::Mat img_f32;
            img.convertTo(img_f32, CV_32FC3, 1.0 / 255.0);
            // img_f64.convertTo(img_f64, CV_64FC3, 1.0 / 255.0);
            cv::Size dsize = cv::Size(224, 224);
            cv::Mat img_resized, img_scaled;
            cv::resize(img_f32, img_resized, dsize, 0, 0, cv::INTER_LINEAR);
            cv::cvtColor(img_resized, img_resized, cv::COLOR_BGR2RGB);
            // cv::imwrite("img2.png",img_resized);
            float data[224 * 224 * 3];
            // std::cout<<"here4"<<std::endl;
            // Mat_to_CHW(x, img_resized);
            float mean[3] = {0.5, 0.5, 0.5};
            float stddev[3] = {0.5, 0.5, 0.5};
            int channels = img_resized.channels();
            int height = img_resized.rows;
            int width = img_resized.cols;
            for (int c = 0; c < channels; ++c)
            {
                for (int h = 0; h < height; ++h)
                {

                    for (int w = 0; w < width; ++w)
                    {
                        int index = c * height * width + h * width + w;
                        // std::cout<<img_data.at<cv::Vec3f>(h, w)[c]<<" ";
                        static_cast<float *>(x->data)[index] = (img_resized.at<cv::Vec3f>(h, w)[c] - mean[c]) / stddev[c];
                        // std::cout<<input_image_[index]<<std::endl;
                        // std::cin>>a;
                    }
                }
            }
            // std::cout<<static_cast<float*>(x->data)[0]<<std::endl;
            auto result = static_cast<float *>(y->data);
            // get the function from the module(run it)
            auto start = system_clock::now();
            set_input("input", x);
            // run the code
            run();
            // get the output
            get_output(0, y);
            auto end = system_clock::now();
            auto duration = duration_cast<microseconds>(end - start);
            total_duration += duration;
            // 将输出的信息打印出来
            result = static_cast<float *>(y->data);
            float max = result[0];
            int predict = 0;
            for (int i = 0; i < 100; i++){
                if (result[i] > max)
                {
                    max = result[i];
                    predict = i;
                }
            }
            if(predict == label){
                correct_num++;
            }
        }
    }
    std::cout<<"total: "<<num<<std::endl;
    std::cout<<"correct: " <<correct_num<<std::endl;
    std::cout<<"acc: "<<(float)correct_num/(float)num<<std::endl;
    std::cout << "time cost: "<<double(total_duration.count()) * microseconds::period::num / microseconds::period::den <<" s "<<std::endl;
}
