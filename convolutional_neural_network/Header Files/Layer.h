//
//  Layer.h
//  CNN
//
//  Created by Amr on 3/9/23.
//

#ifndef Layer_h
#define Layer_h

#include <vector>
#include "Config.h"
#include "RNG.h"
#include "Optimizer.h"
#include <Eigen/Core>

class Layer {

protected:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    
    const int m_input_size;
    const int m_output_size;
    
public:
    Layer(const int int_size, const int out_size):
        m_input_size(int_size),
        m_output_size(out_size)
    {}
    
    virtual ~Layer();
    
    int input_size() const {return m_input_size;}
    int output_size() const {return m_output_size;}
    
    virtual void init(const Scalar & mu, const Scalar& sigma, RNG& rng) = 0;
    virtual void forward(const Matrix& prev_layer_output) = 0;
    virtual const Matrix& output() const = 0;
    virtual void back_prop(const Matrix& previous_layer_output, const Matrix& next_layer_data) = 0;
    virtual Matrix& back_prop_data() const = 0;
    
    virtual std::vector<Scalar> get_param() const = 0;
    virtual void set_param(const std::vector<Scalar>& param){}
    virtual std::vector<Scalar> get_derivatives() const = 0;
};
#endif /* Layer_h */
