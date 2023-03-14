//
//  Identity.h
//  convolutional_neural_network
//
//  Created by Amr on 3/11/23.
//

#ifndef Identity_h
#define Identity_h

#include <Eigen/Core>
#include "Config.h"

class Identity
{
private:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
	
public:
	static inline void activate(const Matrix& Z, Matrix& A)
	{
		A.noaliaz() = Z;
	}
	
	static inline void apply_jcobian(const Matrix& Z, const Matrix& A, const Matrix& F, Matrix& G)
	{
		G.noalias() = F;
		
	}
};
#endif /* Identity_h */
