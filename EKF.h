#ifndef EKF_DRAFT_EKF_H
#define EKF_DRAFT_EKF_H

#include <iostream>
#include <Eigen/Dense>
#include <math.h>

#define M_PI        3.14159265358979323846264338327950288
#define M_PIf       3.14159265358979323846f
#define RAD    (M_PIf / 180.0f)

// EKF struct
struct EKF {
    double dt = 1.002004008016032;
    double gravity = 9.8;
    Eigen::Matrix<double, 7, 1> stateVector;
    Eigen::Matrix<double, 7, 7> covarianceMatrix;
    Eigen::Matrix<double, 7, 7> processNoiseMatrix;
    Eigen::Matrix3d measureNoiseMatrix;
};

// Initialization
void estimate_first_direction(EKF &ekfObj, Eigen::Vector3d &accelValues, Eigen::Vector3d &magnValues);
void initialize(EKF &ekfObj, Eigen::Vector3d &accelFirst, Eigen::Vector3d &magnFirst);
Eigen::Vector4d getQuaternion(EKF &ekfObj);
Eigen::Vector3d getRPY(EKF &ekfObj);

// Prediction phase
void predict(EKF &ekfObj, const Eigen::Vector3d &gyroValues);
Eigen::Matrix4d getOmegaMatrix(const Eigen::Vector3d &omegaValues);
void calculateDynamicModel(Eigen::Matrix<double, 7, 1> &stateVector, const Eigen::Vector3d &omegaValues, double dt);
Eigen::Matrix<double, 7, 7> calculateDynamicJacobian(Eigen::Matrix<double, 7, 1> &stateVector, const Eigen::Vector3d &omegaValues, double dt);

// Update phase
void update(EKF &ekfObj, const Eigen::Vector3d &accelValues);
Eigen::Matrix3d calculateRotationMatrix(Eigen::Matrix<double, 7, 1> &stateVector);
Eigen::Vector3d calculateMeasurementModel(Eigen::Matrix<double, 7, 1> &stateVector, double gravity);
Eigen::Matrix<double, 3, 7> calculateMeasurementJacobian(Eigen::Matrix<double, 7, 1> &stateVector, double gravity);

#endif //EKF_DRAFT_EKF_H
