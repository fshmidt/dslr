import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

houses = {
    'Gryffindor',
    'Hufflepuff',
    'Ravenclaw',
    'Slytherin'
}


def get_house_color(house: str):
    colors = {
        'Gryffindor': 'red',
        'Hufflepuff': 'yellow',
        'Ravenclaw': 'blue',
        'Slytherin': 'green'
    }

    if house not in colors.keys():
        raise ValueError('Invalid house')

    return colors[house]


def main():
    try:
        if len(sys.argv) != 2:
            raise FileNotFoundError
        
        file_path = sys.argv[1]
        if not os.path.isfile(file_path):
            raise FileNotFoundError

        df = pd.read_csv(file_path, index_col=False)
        features = df.select_dtypes('number').columns.tolist()
        features_len = len(features)

        _, axes = plt.subplots(
            nrows=features_len,
            ncols=features_len,
        )

        for i, ax_row in enumerate(axes):
            for j, cell in enumerate(ax_row):
                cell.set_xticks([])
                cell.set_yticks([])
                cell.set_xlabel(
                    '\n'.join(features[j].split()),
                    fontsize=7
                )
                cell.set_ylabel(
                    '\n'.join(features[i].split()),
                    fontsize=7
                )

                if i != j:
                    cell.scatter(
                        x=df[features[i]],
                        y=df[features[j]],
                        c=[
                            get_house_color(house)
                            for house in df['Hogwarts House']
                        ],
                        s=1
                    )
                else:
                    for house in houses:
                        cell.hist(
                            df.loc[df['Hogwarts House'] == house, features[i]],
                            alpha=.5,
                            label=house,
                            color=get_house_color(house)
                        )

        for ax in axes.flat:  # Remove the labels in the middle of the plot
            ax.label_outer()

        plt.show()
    except FileNotFoundError:
        print("Invalid file path.")
    except PermissionError:
        print("Permission denied.")
    except:
        print("Unexpected error occurred.")


if __name__ == '__main__':
    main()
