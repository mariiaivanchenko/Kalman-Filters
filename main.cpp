#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdlib>
#include "EKF.h"

void readValues(const std::string &filename, std::vector<Eigen::Vector3d> &sensorValues) {
    std::ifstream file(filename);
    std::string currentLine;

    while (std::getline(file, currentLine)) {
        Eigen::Vector3d values;
        std::stringstream ss(currentLine);
        std::string token;
        int i = 0;

        while (std::getline(ss, token, ',') && i < 3) {
            values[i++] = std::strtod(token.c_str(), nullptr);
        }

        sensorValues.push_back(std::move(values));
    }
}

void writeValues(const std::string &filename, const std::vector<Eigen::Vector3d> &anglesValues) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open output file: " + filename);
    }

    for (const auto& vec : anglesValues) {
        file << vec[0] << "," << vec[1] << "," << vec[2] << "\n";
    }

    file.close();
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <input_accel.csv> <input_gyro.csv> <input_magn.csv> <output_file.csv>\n";
        return 1;
    }

    std::string inputAccel = argv[1];
    std::string inputGyro = argv[2];
    std::string inputMagn = argv[3];
    std::string outputFilename = argv[4];

    std::vector<Eigen::Vector3d> accelValues;
    std::vector<Eigen::Vector3d> gyroValues;
    std::vector<Eigen::Vector3d> magnValues;
    readValues(inputAccel, accelValues);
    readValues(inputGyro, gyroValues);
    readValues(inputMagn, magnValues);

    EKF ekfFilter;
    std::vector<Eigen::Vector3d> anglesValues;
    initialize(ekfFilter, accelValues[0], magnValues[0]);
    size_t numValues = accelValues.size();
    for (size_t idx = 0; idx < numValues; idx++) {
        predict(ekfFilter, gyroValues[idx]);
        update(ekfFilter, accelValues[idx]);
        anglesValues.push_back(getRPY(ekfFilter));
    }
    writeValues(outputFilename, anglesValues);

    return 0;
}
