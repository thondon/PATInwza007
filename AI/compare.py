import numpy as np
from sklearn.metrics import mean_squared_error

# Load the numpy arrays
file1 = np.load('./data/4/t1.npy')
file2 = np.load('./data/4/e1.npy')

# Replace NaNs with 0
file1 = np.nan_to_num(file1, nan=0)
file2 = np.nan_to_num(file2, nan=0)

# Check shapes match
assert file1.shape == file2.shape, "Shapes do not match"

# --- คำนวณ MSE ---
mse = mean_squared_error(file1, file2)
mse_per_len = mse / len(file1)

print(f"Mean Squared Error (MSE): {mse}")
print(f"MSE / len: {mse_per_len}")

# --- เช็กทิศทางการเปลี่ยนแปลง ---
errors = []

for i in range(len(file1) - 1):
    diff1 = file1[i + 1] - file1[i]
    diff2 = file2[i + 1] - file2[i]

    if (diff1 > 0 and diff2 <= 0) or (diff1 < 0 and diff2 >= 0):
        errors.append(i)

# --- ฟังก์ชันจัดรูปแบบ index ---
def format_indices(indices):
    if not indices:
        return ''
    result = []
    start = indices[0]
    end = start

    for idx in indices[1:]:
        if idx == end + 1:
            end = idx
        else:
            if start == end:
                result.append(f"({start})" if start < 0 else f"{start}")
            else:
                result.append(f"({start})-({end})" if start < 0 or end < 0 else f"{start}-{end}")
            start = end = idx

    if start == end:
        result.append(f"({start})" if start < 0 else f"{start}")
    else:
        result.append(f"({start})-({end})" if start < 0 or end < 0 else f"{start}-{end}")

    return result

# --- แสดงผลลัพธ์ ---
formatted_errors = format_indices(errors)
print(f"Mismatch indices ({len(formatted_errors)}): {",".join(formatted_errors)}")

# --- หา Longest Increasing Adjacent Subsequence ---
def find_longest_increasing_subseq(arr):
    max_len = 1
    current_len = 1
    start_idx = 0
    max_start_idx = 0

    for i in range(1, len(arr)):
        if arr[i] > arr[i - 1]:
            current_len += 1
        else:
            if current_len > max_len:
                max_len = current_len
                max_start_idx = start_idx
            current_len = 1
            start_idx = i

    # เช็กครั้งสุดท้าย
    if current_len > max_len:
        max_len = current_len
        max_start_idx = start_idx

    # คืนค่า start index, end index, และ length
    return max_start_idx, max_start_idx + max_len - 1, max_len

# --- เรียกใช้กับ file1 หรือ file2 ---
start_idx, end_idx, length = find_longest_increasing_subseq(file1)

print(f"Longest Increasing Adjacent Subsequence in file1:")
print(f"Start index: {start_idx}")
print(f"End index: {end_idx}")
print(f"Length: {length}")