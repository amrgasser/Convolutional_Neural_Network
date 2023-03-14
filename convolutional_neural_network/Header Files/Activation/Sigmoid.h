//
//  Sigmoid.h
//  convolutional_neural_network
//
//  Created by Amr on 3/11/23.
//

#ifndef Sigmoid_h
#define Sigmoid_h

class Sigmoid
{
private:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
public:
	static inline void activate(const Matrix& Z, Matrix& A)
	{
		A.array() = Scalar(1) / (Scalar(1) + (-Z.array()).exp());
	}
	
	static inline void apply_jcobian(const Matrix& Z, const Matrix& A, const Matrix& F, Matrix& G)
	{
		G.array() = A.array() * (1 - A.array()) * F.array();
	}
};

#endif /* Sigmoid_h */
