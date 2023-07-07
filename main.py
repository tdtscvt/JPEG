import numpy as np
import cv2
from PIL import Image
from multiprocessing.pool import Pool
import utils
import os
import entrophy
import os

# Instantiation
###############################################################################
byte_size = 0
subsampling = utils.ChromaSubsampling()
Y_block = utils.ImageBlock(block_height=8, block_width=8)
Cr_block = utils.ImageBlock(block_height=8, block_width=8)
Cb_block = utils.ImageBlock(block_height=8, block_width=8)

dct2d = utils.DCT2D(norm='ortho')
# quantization = utils.Quantization()
# read image
imgOriginal = cv2.imread('image.png', cv2.IMREAD_COLOR)
# convert BGR to YCrCb
ycc_img = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2YCR_CB)
# Center
ycc_img = ycc_img.astype(int) - 128

# Downsampling
Y = ycc_img[:, :, 0]
Cr = subsampling.downsampling(ycc_img[:, :, 1])
Cb = subsampling.downsampling(ycc_img[:, :, 2])

Y_blocks, Y_indices = Y_block.forward(Y)
# Y_blocks = utils.dct2d(Y_blocks)
Y_blocks = dct2d.forward(Y_blocks)
Y_blocks = utils.lum_quantization(Y_blocks)
DC_encode, DC_value, DC_code, AC_encode, AC_value, AC_code = entrophy.encode(Y_blocks)
byte_size += DC_encode.nbytes + DC_value.nbytes + DC_code.nbytes + AC_encode.nbytes + AC_value.nbytes + AC_code.nbytes
Y_blocks = entrophy.decode(DC_encode, DC_value, DC_code, AC_encode, AC_value, AC_code)
Y_blocks = utils.lum_dequantization(Y_blocks)
# Y_blocks = utils.idct2d(Y_blocks)
Y_blocks = dct2d.backward(Y_blocks)
Y_image = Y_block.backward(Y_blocks, Y_indices)

Cr_blocks, Cr_indices = Cr_block.forward(Cr)
# Cr_blocks = utils.dct2d(Cr_blocks)
Cr_blocks = dct2d.forward(Cr_blocks)
Cr_blocks = utils.chr_quantization(Cr_blocks)
DC_encode, DC_value, DC_code, AC_encode, AC_value, AC_code = entrophy.encode(Cr_blocks)
byte_size += DC_encode.nbytes + DC_value.nbytes + DC_code.nbytes + AC_encode.nbytes + AC_value.nbytes + AC_code.nbytes
Cr_blocks = entrophy.decode(DC_encode, DC_value, DC_code, AC_encode, AC_value, AC_code)
Cr_blocks = utils.chr_dequantization(Cr_blocks)
# Cr_blocks = utils.idct2d(Cr_blocks)
Cr_blocks = dct2d.backward(Cr_blocks)
Cr_image = Cr_block.backward(Cr_blocks, Cr_indices)
UCr = subsampling.upsampling(Cr_image)

Cb_blocks, Cb_indices = Cb_block.forward(Cb)
# Cb_blocks = utils.dct2d(Cb_blocks)
Cb_blocks = dct2d.forward(Cb_blocks)
Cb_blocks = utils.chr_quantization(Cb_blocks)
DC_encode, DC_value, DC_code, AC_encode, AC_value, AC_code = entrophy.encode(Cb_blocks)
byte_size += DC_encode.nbytes + DC_value.nbytes + DC_code.nbytes + AC_encode.nbytes + AC_value.nbytes + AC_code.nbytes
Cb_blocks = entrophy.decode(DC_encode, DC_value, DC_code, AC_encode, AC_value, AC_code)
Cb_blocks = utils.chr_dequantization(Cb_blocks)
# Cb_blocks = utils.idct2d(Cb_blocks)
Cb_blocks = dct2d.backward(Cb_blocks)
Cb_image = Cb_block.backward(Cb_blocks, Cb_indices)
UCb = subsampling.upsampling(Cb_image)

ycc_img = np.stack((Y_image, UCr, UCb), axis=2)
ycc_img = (ycc_img + 128).astype('uint8')
ycc_img = cv2.cvtColor(ycc_img, cv2.COLOR_YCrCb2BGR)
cv2.imshow("image", ycc_img)
cv2.imwrite("image.jpg", ycc_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Dung luong anh ban dau: ", os.path.getsize('Image.png'))
print("Dung luong du lieu sau khi ma hoa: ", byte_size)
print("Dung luong anh sau khi ma hoa: ", os.path.getsize('image.jpg'))