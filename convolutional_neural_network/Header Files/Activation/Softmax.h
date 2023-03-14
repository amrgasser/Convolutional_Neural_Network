//
//  Softmax.h
//  convolutional_neural_network
//
//  Created by Amr on 3/11/23.
//

#ifndef Softmax_h
#define Softmax_h
#include "../Config.h"

class Softmax
{
private:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
	typedef Eigen::Array<Scalar, 1, Eigen::Dynamic> RowArray;
public:
	static inline void activate(const Matrix& Z, Matrix& A)
	{
		A.array() = (Z.rowwize() - Z.colwise().maxCoeff()).array().exp();
		RowArray colsum = A.colwise().sum();
		
		A.array().rowwise() /= colsum;
	}
	
	static inline void apply_jcobian(const Matrix& Z, const Matrix& A, const Matrix& F, Matrix& G)
	{
		RowArray a_dot_f = A.cwiseProduct(F).colwise().sum();
		G.array() = A.array() * (F.array().rowwise() - a_dot_f);
	}
};

#endif /* Softmax_h */
