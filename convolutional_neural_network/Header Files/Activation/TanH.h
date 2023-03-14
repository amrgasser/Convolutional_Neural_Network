//
//  TanH.h
//  convolutional_neural_network
//
//  Created by Amr on 3/11/23.
//

#ifndef TanH_h
#define TanH_h
#include "../Config.h"

class TanH
{
private:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
public:
	static inline void activate(const Matrix& Z, Matrix& A)
	{
		A.array() = Z.array().tanh();
	}
	
	static inline void apply_jcobian(const Matrix& Z, const Matrix& A, const Matrix& F, Matrix& G)
	{
		G.array() = (Scalar(1) - A.array(0)) * F.array();
	}
};

#endif /* TanH_h */
