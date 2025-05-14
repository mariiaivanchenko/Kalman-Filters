#include "EKF.h"

void estimate_first_direction(EKF &ekfObj, Eigen::Vector3d &accelValues, Eigen::Vector3d &magnValues) {
    // crossproduct
    Eigen::Vector3d H;
    H[0] = magnValues[1]*accelValues[2] - magnValues[2]*accelValues[1];
    H[1] = magnValues[2]*accelValues[0] - magnValues[0]*accelValues[2];
    H[2] = magnValues[0]*accelValues[1] - magnValues[1]*accelValues[0];

    // normalize
    H.normalize();
    accelValues.normalize();

    // crossproduct
    Eigen::Vector3d M;
    M[0] = accelValues[1]*H[2] - accelValues[2]*H[1];
    M[1] = accelValues[2]*H[0] - accelValues[0]*H[2];
    M[2] = accelValues[0]*H[1] - accelValues[1]*H[0];

    Eigen::Vector4d q_est;
    double trace = H[0] + M[1] + accelValues[2];
    q_est[0] = 0.5 * sqrt(1.0 + trace);
    q_est[1] = (accelValues[1] - M[2]) / q_est[0];
    q_est[2] = (H[2] - accelValues[0]) / q_est[0];
    q_est[3] = (M[0] - H[1]) / q_est[0];

    q_est[1] /= 4;
    q_est[2] /= 4;
    q_est[3] /= 4;

    q_est.normalize();

    ekfObj.stateVector[0] = q_est[0];
    ekfObj.stateVector[1] = q_est[1];
    ekfObj.stateVector[2] = q_est[2];
    ekfObj.stateVector[3] = q_est[3];
}


void initialize(EKF &ekfObj, Eigen::Vector3d &accelFirst, Eigen::Vector3d &magnFirst) {
    // initialize state vector
    estimate_first_direction(ekfObj, accelFirst, magnFirst);

//    // quaternion values (currently default)
//    ekfObj.stateVector[0] = 1.0;
//    ekfObj.stateVector[1] = 0.0;
//    ekfObj.stateVector[2] = 0.0;
//    ekfObj.stateVector[3] = 0.0;

    // gyroscope bias values
    ekfObj.stateVector[4] = 0.0;
    ekfObj.stateVector[5] = 0.0;
    ekfObj.stateVector[6] = 0.0;

    // initialize covariance matrix
    ekfObj.covarianceMatrix.setIdentity();

    // initialize process noise matrix
    ekfObj.processNoiseMatrix.setZero();
    ekfObj.processNoiseMatrix(0, 0) = 0.001;
    ekfObj.processNoiseMatrix(1, 1) = 0.001;
    ekfObj.processNoiseMatrix(2, 2) = 0.001;
    ekfObj.processNoiseMatrix(3, 3) = 0.001;

    // initialize measure noise matrix
    ekfObj.measureNoiseMatrix.setZero();
    ekfObj.measureNoiseMatrix(0,0) = 0.03;
    ekfObj.measureNoiseMatrix(1,1) = 0.03;
    ekfObj.measureNoiseMatrix(2,2) = 0.03;
}

Eigen::Vector4d getQuaternion(EKF &ekfObj) {
    Eigen::Vector4d quaternion;
    quaternion <<
    ekfObj.stateVector[0],
    ekfObj.stateVector[1],
    ekfObj.stateVector[2],
    ekfObj.stateVector[3];
    return quaternion;
}

Eigen::Vector3d getRPY(EKF &ekfObj) {
    Eigen::Vector3d result;
    Eigen::Vector4d q = getQuaternion(ekfObj);

    // roll (x-axis rotation)
    double sinr_cosp = 2.0 * (q(0) * q(1) + q(2) * q(3));
    double cosr_cosp = 1.0 - 2.0 * (q(1) * q(1) + q(2) * q(2));
    result(0) = atan2(sinr_cosp, cosr_cosp);
    result(0) /= RAD * 100;

    // pitch (y-axis rotation)
    double sinp = sqrt(1.0 + 2.0 * (q(0) * q(2) - q(1) * q(3)));
    double cosp = sqrt(1.0 - 2.0 * (q(0) * q(2) - q(1) * q(3)));
    result(1) = 2 * atan2(sinp, cosp) - M_PI / 2;
    result(1) /= RAD * 100;


    // yaw (z-axis rotation)
    double siny_cosp = 2.0 * (q(0) * q(3) + q(1) * q(2));
    double cosy_cosp = 1.0 - 2.0 * (q(2) * q(2) + q(3) * q(3));
    result(2) = atan2(siny_cosp, cosy_cosp);
    result(2) /= RAD * 100;

    return result;
}


// Prediction phase
void predict(EKF &ekfObj, const Eigen::Vector3d &gyroValues) {
    calculateDynamicModel(ekfObj.stateVector, gyroValues, ekfObj.dt);
    Eigen::Matrix<double, 7, 7> jacobianMatrix = calculateDynamicJacobian(ekfObj.stateVector, gyroValues, ekfObj.dt);
    ekfObj.covarianceMatrix = ekfObj.covarianceMatrix + ekfObj.dt * (jacobianMatrix * ekfObj.covarianceMatrix * jacobianMatrix.transpose() + ekfObj.processNoiseMatrix);
}

Eigen::Matrix4d getOmegaMatrix(const Eigen::Vector3d &omegaValues) {
    Eigen::Matrix4d omegaMatrix;
    omegaMatrix << 0, -omegaValues(0), -omegaValues(1), -omegaValues(2),
            omegaValues(0), 0, omegaValues(2), omegaValues(1),
            omegaValues(1), -omegaValues(2), 0, omegaValues(0),
            omegaValues(2), omegaValues(1), -omegaValues(0), 0;
    return omegaMatrix;
}

void calculateDynamicModel(Eigen::Matrix<double, 7, 1> &stateVector, const Eigen::Vector3d &omegaValues, double dt) {
    Eigen::Vector3d bias = stateVector.tail<3>();
    Eigen::Vector3d omega = omegaValues - bias;

    Eigen::Vector4d quaternion = stateVector.head<4>();
    Eigen::Matrix4d omegaMatrix = getOmegaMatrix(omega);
    Eigen::Vector4d q_dot = 0.5 * omegaMatrix * quaternion;
    Eigen::Vector4d q_integrated = quaternion + q_dot * dt;

    stateVector.segment<4>(0).normalize();
}

Eigen::Matrix<double, 7, 7> calculateDynamicJacobian(Eigen::Matrix<double, 7, 1> &stateVector, const Eigen::Vector3d &omegaValues, double dt) {
    Eigen::Matrix<double, 7, 7> jacobianMatrix;
    jacobianMatrix.setIdentity();

    Eigen::Vector4d quaternion = stateVector.head<4>();
    Eigen::Vector3d bias = stateVector.tail<3>();

    double p = omegaValues(0) - bias(0);
    double q = omegaValues(1) - bias(1);
    double r = omegaValues(2) - bias(2);

    jacobianMatrix(0, 1) = -p * dt;
    jacobianMatrix(0, 2) = -q * dt;
    jacobianMatrix(0, 3) = -r * dt;
    jacobianMatrix(0, 4) = 0.5 * quaternion(1) * dt;
    jacobianMatrix(0, 5) = 0.5 * quaternion(2) * dt;
    jacobianMatrix(0, 6) = 0.5 * quaternion(3) * dt;

    jacobianMatrix(1, 0) = p * dt;
    jacobianMatrix(1, 2) = r * dt;
    jacobianMatrix(1, 3) = -q * dt;
    jacobianMatrix(1, 4) = -0.5 * quaternion(0) * dt;
    jacobianMatrix(1, 5) = -0.5 * quaternion(3) * dt;
    jacobianMatrix(1, 6) = 0.5 * quaternion(2) * dt;

    jacobianMatrix(2, 0) = q * dt;
    jacobianMatrix(2, 1) = -r * dt;
    jacobianMatrix(2, 3) = p * dt;
    jacobianMatrix(2, 4) = 0.5 * quaternion(3) * dt;
    jacobianMatrix(2, 5) = -0.5 * quaternion(0) * dt;
    jacobianMatrix(2, 6) = -0.5 * quaternion(1) * dt;

    jacobianMatrix(3, 0) = r * dt;
    jacobianMatrix(3, 1) = q * dt;
    jacobianMatrix(3, 2) = -p * dt;
    jacobianMatrix(3, 4) = -0.5 * quaternion(2) * dt;
    jacobianMatrix(3, 5) = 0.5 * quaternion(1) * dt;
    jacobianMatrix(3, 6) = -0.5 * quaternion(0) * dt;

    return jacobianMatrix;
}


// Update phase
void update(EKF &ekfObj, const Eigen::Vector3d &accelValues) {
    // compute measurement model
    Eigen::Vector3d measurementModel = calculateMeasurementModel(ekfObj.stateVector, ekfObj.gravity);

    // compute innovation
    Eigen::Vector3d innovation = accelValues - measurementModel;

    // calculate measurement jacobian
    Eigen::Matrix<double, 3, 7> measurementJacobian = calculateMeasurementJacobian(ekfObj.stateVector, ekfObj.gravity);

    // calculate innovation covariance
    Eigen::Matrix3d innovationCovariance = measurementJacobian * ekfObj.covarianceMatrix * measurementJacobian.transpose() + ekfObj.measureNoiseMatrix;

    // calculate Kalman Gain
    Eigen::Matrix<double, 7, 3> KalmanGain = ekfObj.covarianceMatrix * measurementJacobian.transpose() * innovationCovariance.inverse();

    // update state vector
    ekfObj.stateVector += KalmanGain * innovation;

    // update covariance matrix
    Eigen::Matrix<double, 7, 7> identity;
    identity.setIdentity();
    ekfObj.covarianceMatrix = (identity - KalmanGain * measurementJacobian) * ekfObj.covarianceMatrix;
}

Eigen::Matrix3d calculateRotationMatrix(Eigen::Matrix<double, 7, 1> &stateVector) {
    Eigen::Vector4d q = stateVector.head<4>();
    Eigen::Matrix3d rotationMatrix;
    rotationMatrix << 1.0 - 2.0 * (q(2) * q(2) + q(3) * q(3)),
                     2.0 * (q(1) * q(2) - q(0) * q(3)),
                     2.0 * (q(1) * q(3) + q(0) * q(2)),
                     2.0 * (q(1) * q(2) + q(0) * q(3)),
                     1.0 - 2.0 * (q(1) * q(1) + q(3) * q(3)),
                     2.0 * (q(2) * q(2) - q(0) * q(1)),
                     2.0 * (q(1) * q(3) - q(0) * q(2)),
                     2.0 * (q(2) * q(3) + q(0) * q(1)),
                     1.0 - 2.0 * (q(1) * q(1) + q(2) * q(1));
    return rotationMatrix;
}

Eigen::Vector3d calculateMeasurementModel(Eigen::Matrix<double, 7, 1> &stateVector, double gravity) {
    Eigen::Vector3d gravityVector(0.0, 0.0, gravity);
    Eigen::Matrix3d rotationMatrix = calculateRotationMatrix(stateVector);
    Eigen::Vector3d accelExpected = rotationMatrix.transpose() * gravityVector;
    return accelExpected;
}

Eigen::Matrix<double, 3, 7> calculateMeasurementJacobian(Eigen::Matrix<double, 7, 1> &stateVector, double gravity) {
    Eigen::Vector4d q = stateVector.head<4>();
    Eigen::Matrix<double, 3, 7> measurementJacobian;

    measurementJacobian.setZero();
    measurementJacobian(0,0) = 2.0 * (-gravity * q(2));
    measurementJacobian(0,1) = 2.0 * (gravity * q(3));
    measurementJacobian(0,2) = 2.0 * (-gravity * q(0));
    measurementJacobian(0,3) = 2.0 * (gravity * q(1));

    measurementJacobian(1,0) = 2.0 * (gravity * q(1));
    measurementJacobian(1,1) = 2.0 * (gravity * q(0));
    measurementJacobian(1,2) = 2.0 * (gravity * q(3));
    measurementJacobian(1,3) = 2.0 * (gravity * q(2));

    measurementJacobian(2,0) = 2.0 * (gravity * q(0));
    measurementJacobian(2,1) = 2.0 * (-gravity * q(1));
    measurementJacobian(2,2) = 2.0 * (-gravity * q(2));
    measurementJacobian(2,3) = 2.0 * (-gravity * q(3));

    return measurementJacobian;
}
