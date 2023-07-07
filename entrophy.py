import numpy as np
import math


def delta_encode(array):
    encoded = np.zeros(len(array), int)
    for i in range(len(array)):
        if i == 0:
            encoded[i] = array[i]
        else:
            encoded[i] = array[i] - array[i - 1]

    return encoded


def delta_decode(array):
    decoded = np.zeros(len(array), int)
    for i in range(len(array)):
        if i == 0:
            decoded[i] = array[0]
        else:
            decoded[i] = array[i] + decoded[i - 1]

    return decoded


zigzag_indices = np.array([
    (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
    (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5),
    (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2),
    (3, 3), (2, 4), (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4),
    (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4),
    (3, 5), (2, 6), (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3),
    (7, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6),
    (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7)
])


def zigzag_scan(block, order):
    vector = np.zeros((1, 63), int)

    for i in range(len(order)):
        index = np.array(order[i])
        vector[0][i] = block[index[0]][index[1]]

    return vector


def zig_zag_descan(vector, order):
    block = np.zeros((8, 8), int)
    for i in range(len(vector)):
        a = order[i][0]
        b = order[i][1]
        block[a][b] = vector[i]
    return block


"""def run_length_encode(array):
    encoded = np.array([], int)

    for i in range(len(array)):
        j = 0
        while j < len(array[i][0]):
            pre = 0
            while j < len(array[i][0]) and array[i][0][j] == 0:
                pre += 1
                j += 1
            if j >= len(array[i][0]):
                encoded = np.concatenate((encoded, np.array([0])))
            else:
                count_zrl = pre // 16
                for _ in range(count_zrl):
                    encoded = np.concatenate((encoded, np.array([240])))
                pre -= count_zrl * 16
                no_zero_size = get_bin_size(array[i][0][j])
                val = pre * 16 + no_zero_size
                encoded = np.concatenate((encoded, np.array([val])))
                j += 1

    return encoded


def run_length_decode(array):
    decoded = np.empty((1, 63), dtype=int)
    p = 0

    i = 0
    while i < len(array):
        segment = np.zeros((1, 63), int)
        j = 0
        while j < len(segment[0]):
            val = array[i]
            if val == 0:  # EOB
                while j < len(segment[0]):
                    segment[0][j] = 0
                    j += 1
                i += 1
                break
            count_zero = val // 16
            non_zero = val - count_zero * 16
            for _ in range(count_zero):
                segment[0][j] = 0
                j += 1
            segment[0][j] = non_zero
            j += 1
            i += 1
        for m in range(len(segment[0])):
            if segment[0][m] != 0:
                segment[0][m] = round(math.exp(segment[0][m] * math.log(2, math.e)))
        if len(decoded) == 1 and p == 0:
            for m in range(len(decoded[0])):
                decoded[0][m] = segment[0][m]
            p = 1
        else:
            decoded = np.concatenate((decoded, segment))

    return decoded"""


def run_length_encode(array):
    encoded = np.array([], int)

    for i in range(len(array)):
        j = 0
        while j < len(array[i][0]):
            pre = 0
            while j < len(array[i][0]) and array[i][0][j] == 0:
                pre += 1
                j += 1
            if j >= len(array[i][0]):
                encoded = np.concatenate((encoded, np.array([0])))
                encoded = np.concatenate((encoded, np.array([0])))
            else:
                encoded = np.concatenate((encoded, np.array([pre])))
                encoded = np.concatenate((encoded, np.array([array[i][0][j]])))
                j += 1

    return encoded


def run_length_decode(array):
    decoded = np.empty((1, 63), dtype=int)
    p = 0

    i = 0
    while i < len(array):
        segment = np.zeros((1, 63), int)
        j = 0
        while j < len(segment[0]):
            if array[i] == 0 and array[i+1] == 0:
                for m in range(j, len(segment[0])):
                    segment[0][j] = 0
                    j += 1
                i += 2
                break
            count_zero = array[i]
            i += 1
            for m in range(count_zero):
                segment[0][j] = 0
                j += 1
            val = array[i]
            i += 1
            segment[0][j] = val
            j += 1
        if len(decoded) == 1 and p == 0:
            for m in range(len(decoded[0])):
                decoded[0][m] = segment[0][m]
            p = 1
        else:
            decoded = np.concatenate((decoded, segment))

    return decoded


# lay do dai day bit
def get_bin_size(num):
    n = abs(num)
    if n == 0:
        return 1

    count = 0
    while n > 0:
        n = n // 2
        count += 1

    return count


# lay gia tri nhi phan
def get_bin_digit(num):
    size = get_bin_size(num)
    digit = ''
    if num < 0:
        num = round(math.exp(size * math.log(2, math.e))) - 1 - abs(num)
    if size == 1:
        digit += num
    else:
        size -= 1
        q = round(math.exp(size * math.log(2, math.e)))
        i = 0
        while i <= size:
            i += 1
            l = num // q
            num = num - l * q
            q = q // 2
            digit += f'{l}'

    return digit


# lay gia tri thap phan
def get_value(digit):
    for i in digit:
        if i == '1':
            return int(digit, 2)
        break

    replace_digit = ''
    for i in digit:
        if i == '0':
            replace_digit += '1'
        else:
            replace_digit += '0'
    return 0 - int(replace_digit, 2)


# lay he so cua mot gia tri trong mang
def get_index_by_value(array, value):
    for i in range(len(array)):
        if array[i] == value:
            return i

    return -1


# lay tan suat xuat hien trong mang
def calculate_frequency(array):
    value = np.array([], int)
    freq = np.array([], int)

    for i in range(len(array)):
        if array[i] not in value:
            value = np.concatenate((value, np.array([array[i]], int)))
            freq = np.concatenate((freq, np.array([1], int)))
        else:
            freq[get_index_by_value(value, array[i])] += 1

    return value, freq


# lay he so cua 2 tan suat nho nhat > 0
def get_two_least_frequency(value, freq):
    if len(value) == 1:
        return 0, 0

    first = -1
    second = -1
    for i in range(len(value)):
        if freq[i] == 0:
            continue

        if first == -1:
            first = i
            continue

        if freq[i] < freq[first]:
            second = first
            first = i
        elif second == -1:
            second = i
        elif freq[i] < freq[second]:
            second = i

    return first, second


def swap(a, b):
    return b, a


# tim so luong bit can de encode gia tri
def find_huffman_code_sizes(value, freq):
    code_size = np.zeros(value.shape, int)  # do lon code cua gia tri
    cp_freq = np.array(freq)  # tao mot mang freq thay the
    others = np.array([-1 for _ in range(len(value))])  # he so tiep theo trong chuoi gia tri cua nhanh hien tai

    while True:
        first, second = get_two_least_frequency(value, cp_freq)

        # neu chi tra ve mot gia tri thi da tim het cac code size
        if second == -1:
            break

        cp_freq[first] += cp_freq[second]
        cp_freq[second] = 0

        while True:
            code_size[first] += 1
            if others[first] == -1:
                break
            first = others[first]

        others[first] = second

        while True:
            code_size[second] += 1
            if others[second] == -1:
                break
            second = others[second]

    """for i in range(len(code_size)):
        if code_size[i] > 16:
            code_size[i] = 16"""

    # sap xep cac gia tri theo thu tu tang dan code_size
    for i in range(len(value)-1):
        for j in range(len(value)-i-1):
            if code_size[j] > code_size[j+1]:
                value[j], value[j+1] = swap(value[j], value[j+1])
                freq[j], freq[j+1] = swap(freq[j], freq[j+1])
                code_size[j], code_size[j + 1] = swap(code_size[j], code_size[j + 1])

    return code_size


# tim so luong code voi moi do dai
def count_bits(code_size):
    bits = np.zeros(17, int)  # he so cua bit tuong uong voi so luong bit can cho code, xem nhu bits[0] khong co
    m = 15
    for i in range(len(code_size)):
        if bits[code_size[i]] >= 255:
            while bits[code_size[m]] >= 255:
                m -= 1
            code_size[i] = m
            bits[m] += 1
        else:
            bits[code_size[i]] += 1

    return bits


def generate_huffman_code(val, size):
    code_value = np.zeros(val.shape, int)  # chua gia tri theo thap phan cua code
    code = np.empty(val.shape, dtype=object)  # chua code thuc su

    c = 0
    i = size[0]
    k = 0
    while k < len(val):
        code_value[k] = c
        c += 1
        k += 1
        while k < len(val) and size[k] != i:
            c *= 2
            i += 1

    for i in range(len(val)):
        cd = np.array([i for i in '{0:b}'.format(code_value[i])], "b")
        while len(cd) < size[i]:
            cd = np.concatenate((np.array([0], "b"), cd))
        code[i] = cd

    return code_value, code


def compare_array(arr1, arr2):
    if len(arr1) != len(arr2):
        return False
    for i in range(len(arr1)):
        if arr1[i] != arr2[i]:
            return False
    return True


def huffman_decode(encode, huff_code, huff_value):
    decoded = np.array([], int)

    i = 0
    while i < len(encode):
        code = np.array([], int)
        in_table = 0
        index = -1
        while in_table == 0:
            code = np.concatenate((code, np.array([encode[i]])))
            i += 1
            for m in range(len(huff_code)):
                if compare_array(code, huff_code[m]):
                    in_table = 1
                    index = m
                    break
        decoded = np.concatenate((decoded, np.array([huff_value[index]], int)))

    return decoded


def encode(blocks):
    # lay cac phan tu DC tu cac block
    DCs = np.zeros(len(blocks), int)
    for i in range(len(blocks)):
        DCs[i] = blocks[i][0][0]

    DCs_process = delta_encode(DCs)  # cac gia tri can phai encode

    # lay tan suat xuat hien cua tung gia tri trong mang
    DC_value, DC_freq = calculate_frequency(DCs_process)
    DC_code_size = find_huffman_code_sizes(DC_value, DC_freq)

    DC_code_value, DC_code = generate_huffman_code(DC_value, DC_code_size)

    DC_encode = np.array([], "B")
    for i in range(len(DCs_process)):
        index = get_index_by_value(DC_value, DCs_process[i])
        for j in DC_code[index]:
            DC_encode = np.concatenate((DC_encode, np.array([j], "B")))

    ACs = np.zeros((len(blocks), 1, 63), int)
    for i in range(len(ACs)):
        ACs[i] = zigzag_scan(blocks[i], zigzag_indices)
    ACs_process = run_length_encode(ACs)  # cac gia tri can phai encode

    # lay tan suat xuat hien cua tung gia tri trong mang
    AC_value, AC_freq = calculate_frequency(ACs_process)
    AC_code_size = find_huffman_code_sizes(AC_value, AC_freq)

    AC_code_value, AC_code = generate_huffman_code(AC_value, AC_code_size)

    AC_encode = np.array([], "B")
    for i in range(len(ACs_process)):
        index = get_index_by_value(AC_value, ACs_process[i])
        for j in AC_code[index]:
            AC_encode = np.concatenate((AC_encode, np.array([j], "B")))

    return DC_encode, DC_value, DC_code, AC_encode, AC_value, AC_code


def decode(DC_encode, DC_value, DC_code, AC_encode, AC_value, AC_code):
    DC_decode = huffman_decode(DC_encode, DC_code, DC_value)
    AC_decode = huffman_decode(AC_encode, AC_code, AC_value)
    DC_de_delta = delta_decode(DC_decode)
    AC_de_run = run_length_decode(AC_decode)
    blocks = np.zeros((len(DC_decode), 8, 8), int)

    for i in range(len(blocks)):
        block = zig_zag_descan(AC_de_run[i], zigzag_indices)
        block[0][0] = DC_de_delta[i]
        blocks[i] = block

    return blocks





