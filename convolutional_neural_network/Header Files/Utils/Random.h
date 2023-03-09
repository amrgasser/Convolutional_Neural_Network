//
//  Random.h
//  convolutional_neural_network
//
//  Created by Amr on 3/9/23.
//

#ifndef Random_h
#define Random_h

#include <Eigen/Core>

class Random {
protected:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    
public:
    void set_normal_random(Matrix& m, int size, RNG& rng, Scalar mu, Scalar sigma);
};
#endif /* Random_h */
