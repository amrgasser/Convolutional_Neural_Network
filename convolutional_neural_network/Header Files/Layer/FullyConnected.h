//
//  FullyConnected.h
//  convolutional_neural_network
//
//  Created by Amr on 3/9/23.
//

#ifndef FullyConnected_h
#define FullyConnected_h

#include <Eigen/Core>
#include <vector>
#include <stdexcept>

#include "Config.h"
#include "../Layer.h"
#include "../Utils/Random.h"

template<typename Activation>

class FullyConnected : public Layer
{
private :
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    
    Matrix m_weight;
    Vector m_bias;
    
    Matrix m_dw; // derivatives of weights
    Vector m_db; // derivative of bias
    Matrix m_z; // multiplying weight matrix to input and sum to bias
    Matrix m_a; // result of activation to m_z
    Matrix m_din; // Derivative of inputs
    
//    Matrix
public:
    FullyConnected(const int input_size, const int output_size):
        Layer(input_size, output_size)
    {}
    
    // when initializing weight and bias, we are going to use normal distribution function.
    void init(const Scalar & mu, const Scalar& sigma, RNG& rng)
    {
        m_weight.resize(this->m_in_size, this->m_out_size);
        m_bias.resize(this->m_out_size);
        m_dw.resize(this->m_in_size, this->m_out_size);
        m_bias.resize(this->m_out_size);
        
        internal::set_normal_random(m_weight.data(), m_weight.size(), rng, mu, sigma);
        internal::set_normal_random(m_bias.data(), m_bias.size(), rng, mu, sigma);
        
    }
    void forward(const Matrix& prev_layer_output){
        const int n_cols = (int) prev_layer_output.cols(); //number of cols
        // Z is equal to the transpose of Weights * previous layer output + bias
        m_z.resize(this->m_out_size, n_cols);
        m_z.noalias() = m_weight.transpose()*prev_layer_output;
        m_z.colwise() += m_bias;
        
        //Apply activation
        m_a.resize(this->m_output_size, n_cols);
        Activation::activate(m_z, m_a);
        
    }
    const Matrix& output() const {
        return m_a;
    }
    void back_prop(const Matrix& previous_layer_output, const Matrix& next_layer_data) {
        
    }
    Matrix& back_prop_data() const {
        return m_din;
    }
    
    void update(Optimizer& opt)
    {
        ConstAlignedMapVec dw(m_dw.data(), m_dw.size());
        ConstAlignedMapVec db(m_db.data(), m_db.size());
        AlignedMapVec weight(m_weight.data(), m_weight.size());
        AlignedMapVec bias(m_bias.data(), m_bias.size());
        
        opt.update(dw, weight);
        opt.update(db, bias);
    }
    
    std::vector<Scalar> get_param() const{}
    void set_param(const std::vector<Scalar>& param){}
    std::vector<Scalar> get_derivatives() const { }
    
    
};

#endif /* FullyConnected_h */
