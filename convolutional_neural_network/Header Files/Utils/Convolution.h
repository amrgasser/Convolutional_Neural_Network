//
//  Convolution.h
//  convolutional_neural_network
//
//  Created by Amr on 3/11/23.
//

#ifndef Convolution_h
#define Convolution_h
#include <Eigen/Core>
#include <vector>
#include "../Config.h"

namespace internal {


struct ConvDims
{
	// Input parameters
	const int in_channels;
	const int out_channels;
	const int channel_rows;
	const int channel_cols;
	const int filter_rows;
	const int filter_cols;
	// Image dimension -- one observation with all channels
	const int img_rows;
	const int img_cols;
	// Dimension of the convolution result for each output channel
	const int conv_rows;
	const int conv_cols;

	ConvDims(
		const int in_channels_, const int out_channels_,
		const int channel_rows_, const int channel_cols_,
		const int filter_rows_, const int filter_cols_
	) :
		in_channels(in_channels_), out_channels(out_channels_),
		channel_rows(channel_rows_), channel_cols(channel_cols_),
		filter_rows(filter_rows_), filter_cols(filter_cols_),
		img_rows(channel_rows_), img_cols(in_channels_ * channel_cols_),
		conv_rows(channel_rows_ - filter_rows_ + 1), conv_cols(channel_cols_ - filter_cols_ + 1)
	{}
};
inline void flatten_mat(
	const ConvDims& dim, const Scalar* src, const int stride, const int n_obs,
	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& flat_mat
)
{
	// Number of bytes in the segment that will be copied at one time
	const int& segment_size = dim.filter_rows;
	const std::size_t copy_bytes = sizeof(Scalar) * segment_size;

	Scalar* writer = flat_mat.data();
	const int channel_size = dim.channel_rows * dim.channel_cols;
	for(int i = 0; i < n_obs; i++, src += stride)
	{
		const Scalar* reader_row = src;
		const Scalar* const reader_row_end = src + dim.conv_rows;
		for(; reader_row < reader_row_end; reader_row++)
		{
			const Scalar* reader = reader_row;
			const Scalar* const reader_end = reader + channel_size;
			for(; reader < reader_end; reader += dim.channel_rows, writer += segment_size)
				std::memcpy(writer, reader, copy_bytes);
		}
	}
}
inline void moving_product(
	const int step,
	const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& mat1,
	Eigen::Map< const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> >& mat2,
	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& res
)
{
	const int row1 = mat1.rows();
	const int col1 = mat1.cols();
	const int row2 = mat2.rows();
	const int col2 = mat2.cols();
	const int col_end = col1 - row2;
	int res_start_col = 0;
	for(int left_end = 0; left_end <= col_end; left_end += step, res_start_col += col2)
	{
		res.block(0, res_start_col, row1, col2).noalias() += mat1.block(0, left_end, row1, row2) * mat2;
	}
}
// The main convolution function using the "valid" rule
inline void convolve_valid(
	const ConvDims& dim,
	const Scalar* src, const bool image_outer_loop, const int n_obs, const Scalar* filter_data,
	Scalar* dest)
{
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RMatrix;
	typedef Eigen::Map<const Matrix> ConstMapMat;

	// Flat matrix
	const int flat_rows = dim.conv_rows * n_obs;
	const int flat_cols = dim.filter_rows * dim.channel_cols;
	const int channel_size = dim.channel_rows * dim.channel_cols;
	// Distance between two images
	const int img_stride = image_outer_loop ? (dim.img_rows * dim.img_cols) : channel_size;
	// Distance between two channels
	const int channel_stride = image_outer_loop ? channel_size : (channel_size * n_obs);
	RMatrix flat_mat(flat_rows, flat_cols);

	// Convolution results
	const int& res_rows = flat_rows;
	const int res_cols = dim.conv_cols * dim.out_channels;
	Matrix res = Matrix::Zero(res_rows, res_cols);

	const int& step = dim.filter_rows;
	const int filter_size = dim.filter_rows * dim.filter_cols;
	const int filter_stride = filter_size * dim.out_channels;

	for(int i = 0; i < dim.in_channels; i++, src += channel_stride, filter_data += filter_stride)
	{
		// Flatten source image
		flatten_mat(dim, src, img_stride, n_obs, flat_mat);
		// Compute the convolution result
		ConstMapMat filter(filter_data, filter_size, dim.out_channels);
		moving_product(step, flat_mat, filter, res);
	}

	const int dest_rows = dim.conv_rows;
	const int dest_cols = res_cols * n_obs;
	const Scalar* res_data = res.data();
	const std::size_t copy_bytes = sizeof(Scalar) * dest_rows;
	for(int b = 0; b < dest_cols; b++, dest += dest_rows)
	{
		const int k = b / res_cols;
		const int l = (b % res_cols) / dim.conv_cols;
		const int j = b % dim.conv_cols;
		const int d = j * dim.out_channels + l;
		const int res_col_head = d * res_rows;
		std::memcpy(dest, res_data + res_col_head + k * dim.conv_rows, copy_bytes);
	}
}

inline void moving_product(
	const int padding, const int step,
	const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& mat1,
	const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& mat2,
	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& res
)
{
	const int row1 = mat1.rows();
	const int col1 = mat1.cols();
	const int row2 = mat2.rows();
	const int col2 = mat2.cols();
	int res_start_col = 0;

	// Left padding
	int left_end = -padding;
	int right_end = step;
	for(; left_end < 0 && right_end <= col1; left_end += step, right_end += step, res_start_col += col2)
	{
		res.block(0, res_start_col, row1, col2).noalias() += mat1.leftCols(right_end) *
			mat2.bottomRows(right_end);
	}
	// Main part
	for(; right_end <= col1; left_end += step, right_end += step, res_start_col += col2)
	{
		res.block(0, res_start_col, row1, col2).noalias() += mat1.block(0, left_end, row1, row2) * mat2;
	}
	// Right padding
	for(; left_end < col1; left_end += step, res_start_col += col2)
	{
		if(left_end <= 0)
		{
			res.block(0, res_start_col, row1, col2).noalias() += mat1 * mat2.block(0, -left_end, col1, row2);
		} else {
			const int overlap = col1 - left_end;
			res.block(0, res_start_col, row1, col2).noalias() += mat1.rightCols(overlap) *
				mat2.topRows(overlap);
		}
	}
}
// The main convolution function for the "full" rule
inline void convolve_full(
	const ConvDims& dim,
	const Scalar* src, const int n_obs, const Scalar* filter_data,
	Scalar* dest)
{
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RMatrix;
	typedef Eigen::Map<const Matrix> ConstMapMat;

	// Padding sizes
	const int padding_top = dim.filter_rows - 1;
	const int padding_left = dim.filter_cols - 1;

	// Dimension of convolution result using "full" rule
	const int conv_rows = dim.channel_rows + padding_top;
	const int conv_cols = dim.channel_cols + padding_left;

	// Add (top and bottom) padding to source images
	const int pad_rows = dim.img_rows + padding_top * 2;
	const int pad_cols = dim.img_cols * n_obs;
	Matrix pad_mat(pad_rows, pad_cols);
	ConstMapMat src_mat(src, dim.img_rows, pad_cols);
	pad_mat.topRows(padding_top).setZero();
	pad_mat.bottomRows(padding_top).setZero();
	pad_mat.block(padding_top, 0, dim.img_rows, pad_cols).noalias() = src_mat;
	src = pad_mat.data();
	ConvDims pad_dim(dim.in_channels, dim.out_channels, pad_rows, dim.channel_cols, dim.filter_rows, dim.filter_cols);

	// Flat matrix
	const int flat_rows = conv_rows * n_obs;
	const int flat_cols = dim.filter_rows * dim.channel_cols;
	const int img_stride = pad_rows * dim.img_cols;
	const int channel_stride = pad_rows * dim.channel_cols;
	RMatrix flat_mat(flat_rows, flat_cols);

	// The processing of filters are different from the "valid" rule in two ways:
	// 1. The layout of input channels and output channels are switched
	// 2. The filters need to be rotated, which is equivalent to reversing the vector of each filter
	// We also separate filters that belong to different input channels
	std::vector<Matrix> filters_in(dim.in_channels);
	const int filter_size = dim.filter_rows * dim.filter_cols;
	const int nfilter = dim.in_channels * dim.out_channels;
	for(int i = 0; i < dim.in_channels; i++)
	{
		filters_in[i].resize(filter_size, dim.out_channels);
	}
	const Scalar* reader = filter_data;
	for(int i = 0; i < nfilter; i++, reader += filter_size)
	{
		Scalar* writer = filters_in[i % dim.in_channels].data() + (i / dim.in_channels) * filter_size;
		std::reverse_copy(reader, reader + filter_size, writer);
	}

	// Convolution results
	const int& res_rows = flat_rows;
	const int res_cols = conv_cols * dim.out_channels;
	Matrix res = Matrix::Zero(res_rows, res_cols);

	const int& step = dim.filter_rows;
	const int filter_padding = padding_left * dim.filter_rows;
	for(int i = 0; i < dim.in_channels; i++, src += channel_stride)
	{
		// Flatten source image
		flatten_mat(pad_dim, src, img_stride, n_obs, flat_mat);
		// Compute the convolution result
		moving_product(filter_padding, step, flat_mat, filters_in[i], res);
	}

	// Copy results to destination
	const int& dest_rows = conv_rows;
	const int  dest_cols = res_cols * n_obs;
	const Scalar* res_data = res.data();
	const std::size_t copy_bytes = sizeof(Scalar) * dest_rows;
	for(int b = 0; b < dest_cols; b++, dest += dest_rows)
	{
		const int k = b / res_cols;
		const int l = (b % res_cols) / conv_cols;
		const int j = b % conv_cols;
		const int d = j * dim.out_channels + l;
		const int res_col_head = d * res_rows;
		std::memcpy(dest, res_data + res_col_head + k * conv_rows, copy_bytes);
	}
}

}

#endif /* Convolution_h */
