import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_course_homogeneity(df, course):
	house_means = df.groupby('Hogwarts House')[course].mean()
	means_std = house_means.std()

	overall_mean = df[course].mean()
	overall_std = df[course].std()
	cv_means = means_std / abs(house_means.mean()) * 100

	return {
		'house_means': house_means,
		'means_std': means_std,
		'cv_means': cv_means,
		'overall_mean': overall_mean,
		'overall_std': overall_std
	}


def plot_distributions(df, course):
	plt.figure(figsize=(10, 6))
	sns.boxplot(data=df, x='Hogwarts House', y=course)
	plt.title(f'Distribution of {course} Scores by House')
	plt.xticks(rotation=45)
	
	# Add house means as points
	means = df.groupby('Hogwarts House')[course].mean()
	plt.plot(range(len(means)), means, 'ro', label='House Mean')
	
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.show()


def main():
	try:
		if len(sys.argv) != 2:
			raise FileNotFoundError
		
		file_path = sys.argv[1]
		if not os.path.isfile(file_path):
			raise FileNotFoundError

		df = pd.read_csv(file_path)
		df.dropna(subset=['Hogwarts House'], inplace=True)
		courses = ['Arithmancy', 'Astronomy', 'Herbology', 
				'Defense Against the Dark Arts', 'Divination', 
				'Muggle Studies', 'Ancient Runes', 
				'History of Magic', 'Transfiguration', 
				'Potions', 'Care of Magical Creatures', 
				'Charms', 'Flying']

		results = {}
		for course in courses:
			results[course] = analyze_course_homogeneity(df, course)

		# Sort courses by homogeneity (using coefficient of variation of means)
		sorted_courses = sorted(results.items(), key=lambda x: x[1]['cv_means'])

		print("\nCourse Homogeneity Analysis (sorted from most to least homogeneous):")
		print("\nCourse               CV of Means  Std of Means")
		print("-" * 50)
		for course, stats in sorted_courses:
			print(f"{course:<20} {stats['cv_means']:8.2f}%    {stats['means_std']:8.2f}")

		# Show house means for most homogeneous course
		most_homogeneous = sorted_courses[0][0]
		print(f"\nMost homogeneous course: {most_homogeneous}")
		print("\nHouse means for this course:")
		print(results[most_homogeneous]['house_means'])
		
		# Plot top 3 most homogeneous courses
		for course, _ in sorted_courses[:3]:
			plot_distributions(df, course)
	except FileNotFoundError:
		print("Invalid file path.")
	except PermissionError:
		print("Permission denied.")
	except:
		print("Unexpected error occurred.")


if __name__ == '__main__':
	main()
