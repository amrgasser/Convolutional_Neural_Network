//
//  ReLU.h
//  convolutional_neural_network
//
//  Created by Amr on 3/11/23.
//

#ifndef ReLU_h
#define ReLU_h

#include "../Config.h"

class ReLU
{
private:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
public:
	static inline void activate(const Matrix& Z, Matrix& A)
	{
		
	}
	
	static inline void apply_jcobian(const Matrix& Z, const Matrix& A, const Matrix& F, Matrix& G)
	{
		
	}
};

#endif /* ReLU_h */
