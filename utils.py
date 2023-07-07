#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image compression util functions.

@author: khe
"""
import numpy as np
from scipy.fft import dct
from scipy.signal import convolve2d


class ChromaSubsampling:
    def __init__(self):
        self.subsampling_ratio = 2

    def downsampling(self, matrix):
        height, width = matrix.shape
        subsampled_height = height // self.subsampling_ratio
        subsampled_width = width // self.subsampling_ratio

        subsampled_matrix = np.zeros((subsampled_height, subsampled_width), dtype=matrix.dtype)

        for h in range(subsampled_height):
            for w in range(subsampled_width):
                subsampled_matrix[h, w] = np.mean(matrix[h*self.subsampling_ratio:(h+1)*self.subsampling_ratio,
                                                        w*self.subsampling_ratio:(w+1)*self.subsampling_ratio])

        return subsampled_matrix

    def upsampling(self, subsampled_matrix):
        subsampled_height, subsampled_width = subsampled_matrix.shape
        height = subsampled_height * self.subsampling_ratio
        width = subsampled_width * self.subsampling_ratio

        upscaled_matrix = np.zeros((height, width), dtype=subsampled_matrix.dtype)

        for h in range(subsampled_height):
            for w in range(subsampled_width):
                upscaled_matrix[h*self.subsampling_ratio:(h+1)*self.subsampling_ratio,
                                w*self.subsampling_ratio:(w+1)*self.subsampling_ratio] = subsampled_matrix[h, w]

        return upscaled_matrix

class ImageBlock():
    def __init__(self, block_height=8, block_width=8):
        self.block_height = block_height
        self.block_width = block_width
        self.left_padding = self.right_padding = self.top_padding = self.bottom_padding = 0

    def forward(self, image):
        self.image_height = image.shape[0]
        self.image_width = image.shape[1]

        # Vertical padding
        if self.image_height % self.block_height != 0:
            vpad =self.block_height - (self.image_height % self.block_height)
            self.top_padding = vpad // 2
            self.bottom_padding = vpad - self.top_padding
            image = np.concatenate((np.repeat(image[:1], self.top_padding, 0), image,
                                    np.repeat(image[-1:], self.bottom_padding, 0)), axis=0)

        # Horizontal padding
        if self.image_width % self.block_width != 0:
            hpad = self.block_width - (self.image_width % self.block_width)
            self.left_padding = hpad // 2
            self.right_padding = hpad - self.left_padding
            image = np.concatenate((np.repeat(image[:, :1], self.left_padding, 1), image,
                                    np.repeat(image[:, -1:], self.right_padding, 1)), axis=1)

        # Update dimension
        self.image_height = image.shape[0]
        self.image_width = image.shape[1]

        # Create blocks
        blocks = []
        indices = []
        for i in range(0, self.image_height, self.block_height):
            for j in range(0, self.image_width, self.block_width):
                blocks.append(image[i:i + self.block_height, j:j + self.block_width])
                indices.append((i, j,))

        blocks = np.array(blocks)
        indices = np.array(indices)
        return blocks, indices

    def backward(self, blocks, indices):
        # Empty image array
        image = np.zeros((self.image_height, self.image_width)).astype(int)
        for block, index in zip(blocks, indices):
            i, j = index
            image[i:i + self.block_height, j:j + self.block_width] = block

        # Remove padding
        if self.top_padding > 0:
            image = image[self.top_padding:, :]
        if self.bottom_padding > 0:
            image = image[:-self.bottom_padding, :]
        if self.left_padding > 0:
            image = image[:, self.left_padding:]
        if self.right_padding > 0:
            image = image[:, :-self.right_padding]
        return image


def dct2d(blocks):
    x = []
    # Normalizing scalar
    def alpha(u):
        return 1 / (2 ** 0.5) if u == 0 else 1
    for block in blocks :
        out = np.zeros((block.shape))
        for u in range(out.shape[0]):
            for v in range(out.shape[1]):
                scalar = 0.25 * alpha(u) * alpha(v)
                val = 0.0
                for i in range(block.shape[0]):
                    for j in range(block.shape[1]):
                        val += block[i, j] * np.cos((2 * i + 1) * u * np.pi / 16) * np.cos((2 * j + 1) * v * np.pi / 16)
                out[u, v] =  scalar * val
        #np.set_printoptions(suppress=True)
        x.append(out)
    x = np.array(x)
    return x
def idct2d(blocks):
    x = []
    # Normalizing scalar
    def alpha(u):
        return 1 / (2 ** 0.5) if u == 0 else 1
    for block in blocks :
        out = np.zeros((block.shape))
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                val = 0.0
                for u in range(block.shape[0]):
                    for v in range(block.shape[1]):
                        scalar = 0.25 * alpha(u) * alpha(v)
                        val += scalar * block[u, v] * np.cos((2 * i + 1) * u * np.pi / 16) * np.cos((2 * j + 1) * v * np.pi / 16)
                out[i, j] = (val)

        #np.set_printoptions(suppress=True)
        x.append(np.round(out))
    x = np.array(x)
    return x
def lum_quantization (blocks) :
    x = []
    Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                  [12, 12, 14, 19, 26, 58, 60, 55],
                  [14, 13, 16, 24, 40, 57, 69, 56],
                  [14, 17, 22, 29, 51, 87, 80, 62],
                  [18, 22, 37, 56, 68, 109, 103, 77],
                  [24, 35, 55, 64, 81, 104, 113, 92],
                  [49, 64, 78, 87, 103, 121, 120, 101],
                  [72, 92, 95, 98, 112, 100, 103, 99]])
    for block in blocks :
        out = np.round(block / Q)
        out[np.where(out == -0)] = 0
        x.append(out.astype(int))
    return np.array(x)
def chr_quantization (blocks) :
    x = []
    Q = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                  [18, 21, 26, 66, 99, 99, 99, 99],
                  [24, 26, 56, 99, 99, 99, 99, 99],
                  [47, 66, 99, 99, 99, 99, 99, 99],
                  [99, 99, 99, 99, 99, 99, 99, 99],
                  [99, 99, 99, 99, 99, 99, 99, 99],
                  [99, 99, 99, 99, 99, 99, 99, 99],
                  [99, 99, 99, 99, 99, 99, 99, 99]])
    for block in blocks:
        out = np.round(block / Q)
        out[np.where(out == -0)] = 0
        x.append(out.astype(int))
    return np.array(x)
def lum_dequantization (blocks) :
    x = []
    Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                  [12, 12, 14, 19, 26, 58, 60, 55],
                  [14, 13, 16, 24, 40, 57, 69, 56],
                  [14, 17, 22, 29, 51, 87, 80, 62],
                  [18, 22, 37, 56, 68, 109, 103, 77],
                  [24, 35, 55, 64, 81, 104, 113, 92],
                  [49, 64, 78, 87, 103, 121, 120, 101],
                  [72, 92, 95, 98, 112, 100, 103, 99]])
    for block in blocks :
        out = block * Q
        x.append(out)
    return np.array(x)
def chr_dequantization (blocks) :
    x = []
    Q = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                  [18, 21, 26, 66, 99, 99, 99, 99],
                  [24, 26, 56, 99, 99, 99, 99, 99],
                  [47, 66, 99, 99, 99, 99, 99, 99],
                  [99, 99, 99, 99, 99, 99, 99, 99],
                  [99, 99, 99, 99, 99, 99, 99, 99],
                  [99, 99, 99, 99, 99, 99, 99, 99],
                  [99, 99, 99, 99, 99, 99, 99, 99]])
    for block in blocks :
        out = block * Q
        x.append(out)
    return np.array(x)


class DCT2D():
    def __init__(self, norm='ortho'):
        if norm is not None:
            assert norm == 'ortho', "norm needs to be in {None, 'ortho'}"
        self.norm = norm

    def forward(self, blocks):
        x = []
        for block in blocks :
            out = dct(dct(block, norm=self.norm, axis=0), norm=self.norm, axis=1)
            x.append(out)
        return np.array(x)

    def backward(self, blocks):
        x = []
        for block in blocks:
            out = dct(dct(block, type=3, norm=self.norm, axis=0), type=3, norm=self.norm, axis=1)
            x.append(np.round(out))
        return np.array(x)