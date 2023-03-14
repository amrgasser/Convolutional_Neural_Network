//
//  Optimizer.h
//  convolutional_neural_network
//
//  Created by Amr on 3/9/23.
//

#ifndef Optimizer_h
#define Optimizer_h

class Optimizer{
protected:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
	typedef Vector::ConstAlignedMapType ConstAlignedMapVec;
	typedef Vector::AlignedMapType AlignedMapVec;

public:
	virtual ~Optimizer(){}

	virtual void reset() {};
	virtual void update(ConstAlignedMapVec& dvec, AlignedMapVec& vec) = 0;
};
#endif /* Optimizer_h */
