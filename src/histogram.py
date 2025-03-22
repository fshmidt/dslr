import os
import sys
import pandas as pd
import matplotlib.pyplot as plt


def analyze_course_homogeneity(df, course):
    valid_df = df.dropna(subset=[course])
    
    if len(valid_df) < 4:
        return {
            'house_means': pd.Series(),
            'means_std': float('nan'),
            'cv_means': float('nan'),
            'overall_mean': float('nan'),
            'overall_std': float('nan')
        }
    
    house_means = valid_df.groupby('Hogwarts House')[course].mean()
    
    if len(house_means) != 4:
        return {
            'house_means': house_means,
            'means_std': float('nan'),
            'cv_means': float('nan'),
            'overall_mean': valid_df[course].mean(),
            'overall_std': valid_df[course].std()
        }

    means_std = house_means.std()
    overall_mean = valid_df[course].mean()
    overall_std = valid_df[course].std()

    mean_of_means = house_means.mean()
    cv_means = (means_std / abs(mean_of_means) * 100) if mean_of_means != 0 else float('inf')

    return {
        'house_means': house_means,
        'means_std': means_std,
        'cv_means': cv_means,
        'overall_mean': overall_mean,
        'overall_std': overall_std
    }


def plot_distributions(df, course):
    valid_df = df.dropna(subset=[course])
    
    if len(valid_df) < 4:
        print(f"Skipping plot for {course}: insufficient data")
        return
    
    plt.figure(figsize=(10, 6))

    houses = valid_df['Hogwarts House'].unique()
    for house in houses:
        house_data = valid_df[valid_df['Hogwarts House'] == house][course]
        if len(house_data) > 0:
            plt.hist(house_data, bins=10, alpha=0.5, label=house, density=True)
    
    plt.title(f'Distribution of {course} Scores by House')
    plt.xlabel(f'{course} Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    try:
        if len(sys.argv) != 2:
            raise FileNotFoundError("Please provide a CSV file path as argument")

        file_path = sys.argv[1]
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError("Empty CSV file")

        df.dropna(subset=['Hogwarts House'], inplace=True)
        
        courses = ['Arithmancy', 'Astronomy', 'Herbology', 
                  'Defense Against the Dark Arts', 'Divination', 
                  'Muggle Studies', 'Ancient Runes', 
                  'History of Magic', 'Transfiguration', 
                  'Potions', 'Care of Magical Creatures', 
                  'Charms', 'Flying']

        results = {}
        for course in courses:
            if course in df.columns:
                results[course] = analyze_course_homogeneity(df, course)
            else:
                print(f"Warning: {course} not found in dataset")

        valid_results = [(course, stats) for course, stats in results.items() 
                        if not pd.isna(stats['cv_means'])]
        sorted_courses = sorted(valid_results, 
                              key=lambda x: x[1]['cv_means'] if x[1]['cv_means'] != float('inf') else float('inf'))

        print("\nCourse Homogeneity Analysis (sorted from most to least homogeneous):")
        print("\nCourse               CV of Means  Std of Means")
        print("-" * 50)
        for course, stats in sorted_courses:
            cv_display = f"{stats['cv_means']:8.2f}%" if stats['cv_means'] != float('inf') else "inf    "
            std_display = f"{stats['means_std']:8.2f}" if not pd.isna(stats['means_std']) else "nan"
            print(f"{course:<20} {cv_display}    {std_display}")

        if sorted_courses:
            most_homogeneous = sorted_courses[0][0]
            print(f"\nMost homogeneous course: {most_homogeneous}")
            print("\nHouse means for this course:")
            print(results[most_homogeneous]['house_means'])

            for course, _ in sorted_courses[:3]:
                plot_distributions(df, course)
        else:
            print("No valid courses found for analysis")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except PermissionError:
        print("Error: Permission denied when accessing file")
    except Exception as e:
        print(f"Unexpected error occurred: {str(e)}")


if __name__ == '__main__':
    main()