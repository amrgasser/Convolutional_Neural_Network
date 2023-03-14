//
//  MaxAverage.h
//  convolutional_neural_network
//
//  Created by Amr on 3/11/23.
//

#ifndef MaxAverage_h
#define MaxAverage_h

namespace internal {
    inline Scalar average_block(
                                const Scalar,
                                const int nrow,
                                const int ncol,
                                const int col_stride,
                                int& loc
                                ){
        return 0.0;
    }
    inline Scalar sum_row(const Scalar* x, const int n){
        return 0.0;
    }
    inline Scalar find_max(const Scalar* x, const int n){
        int loc = 0;
        for(int i = 1; i < n; i++)
        {
            loc = (x[i] > x[loc] ? i : loc);
        }
        return loc;
    }

inline Scalar find_block_max(
                             const Scalar,
                             const int nrow,
                             const int ncol,
                             const int col_stride,
                             int& loc
                             ){
    return 0.0;
}
}

#endif /* MaxAverage_h */
