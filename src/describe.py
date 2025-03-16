import os
import sys
import math
import pandas as pd


def get_min_max(arr):
    if not arr:
        return None, None

    min_value = arr[0]
    max_value = arr[0]

    for num in arr:
        if num < min_value:
            min_value = num
        if num > max_value:
            max_value = num

    return min_value, max_value


def calculate_percentile(percentile: int, initial_arr: list[float]):
	arr = sorted(initial_arr)
	index = (percentile / 100) * (len(arr) - 1)
	lower_index = int(index)
	upper_index = lower_index + 1
	if upper_index >= len(arr):
		return arr[lower_index]

	return arr[lower_index] + (arr[upper_index] - arr[lower_index]) * (index - lower_index)


def calculate_params(initial_arr: list[float]):
	arr = [x for x in initial_arr if not math.isnan(x)]

	count = len(arr)
	mean = sum(arr) / count
	variance = sum(math.pow(x - mean, 2) for x in arr) / count
	std = math.sqrt(variance)
	percentile_25 = calculate_percentile(25, arr)
	percentile_50 = calculate_percentile(50, arr)
	percentile_75 = calculate_percentile(75, arr)
	min, max = get_min_max(arr)

	return [count, mean, std, percentile_25, percentile_50, percentile_75, min, max]


def main():
	try:
		if (len(sys.argv) != 2):
			raise FileNotFoundError

		file_path = sys.argv[1]
		if not os.path.isfile(file_path):
			raise FileNotFoundError

		df = pd.read_csv(file_path)
		df = df.dropna(axis=1, how='all')
		numeric_df = df.select_dtypes(include=['number'])
		hashmap = {col: numeric_df[col].tolist() for col in numeric_df.columns}
		prepared_map = {key: calculate_params(value) for key, value in hashmap.items()}

		res = pd.DataFrame(prepared_map)
		res.index = ['Count', 'Mean', 'Std', '25%', '50%', '75%', 'Min', 'Max']
		pd.set_option('display.max_columns', None)
		print(res)
	except FileNotFoundError:
		print("Invalid file path.")
	except PermissionError:
		print("Permission denied.")
	except:
		print("Unexpected error occurred.")


if __name__ == '__main__':
	main()
