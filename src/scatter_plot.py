import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
	try:
		if (len(sys.argv) != 2):
			raise FileNotFoundError

		file_path = sys.argv[1]
		if not os.path.isfile(file_path):
			raise FileNotFoundError

		df = pd.read_csv(file_path)
		numeric_columns = df.select_dtypes(include=['float64']).columns.tolist()
		if len(numeric_columns) < 2:
			print("Not enough float columns for comparison.")
			return

		correlation_matrix = df[numeric_columns].corr()
		correlation_pairs = correlation_matrix.unstack()
		correlation_pairs = correlation_pairs[correlation_pairs < 1]  # Exclude self-correlation
		most_similar_pair = correlation_pairs.idxmax()  # Get the pair with the highest correlation

		feature1, feature2 = most_similar_pair
		print(f"The two most similar features are: {feature1} and {feature2}")

		plt.figure(figsize=(10, 6))
		sns.scatterplot(data=df, x=feature1, y=feature2)
		plt.title(f'Scatter Plot of {feature1} vs {feature2}')
		plt.xlabel(feature1)
		plt.ylabel(feature2)
		plt.grid()
		plt.show()
	except FileNotFoundError:
		print("Invalid file path.")
	except PermissionError:
		print("Permission denied.")
	except:
		print("Unexpected error occurred.")


if __name__ == '__main__':
	main()
