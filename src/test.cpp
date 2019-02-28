#include "LecGen.h"

using namespace std;
using namespace cv;

int main() {

    string path = "..//";
    string videoName = "Lecture06.mp4";
    char save_folder =  'L06T';
    LecGen gen(path, videoName, save_folder);
    gen.generate();

}