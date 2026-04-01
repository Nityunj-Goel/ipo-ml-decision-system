import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(df, column, bins=10, step_size = 10, density=False):
    plt.figure(figsize=(30, 4))
    plt.hist(df[column].dropna(), bins=bins, edgecolor='black', density=density)
    xmin, xmax = df[column].min(), df[column].max()
    plt.xticks(np.arange(xmin, xmax + step_size, step_size))
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()

def plot_boxplot_range(df, column, lower=0, upper=100, whis: float=1.5):

    data = df[column].dropna()
    l = np.percentile(data, lower)
    u = np.percentile(data, upper)

    filtered = data[(data >= l) & (data <= u)]
    q1 = np.percentile(filtered, 25)
    mean = np.mean(filtered)
    median = np.percentile(filtered, 50)
    q3 = np.percentile(filtered, 75)
    iqr = q3 - q1
    lower_whisker = filtered[filtered >= q1 - whis * iqr].min()
    upper_whisker = filtered[filtered <= q3 + whis * iqr].max()

    plt.figure(figsize=(6, 4))
    plt.boxplot(filtered, whis=whis)
    plt.title(f"{column} ({lower}-{upper} percentile range)")
    plt.show()

    print(f"Lower whisker: {lower_whisker:g}")
    print(f"25% (Q1): {q1:g}")
    print(f"50% (Median): {median:g}")
    print(f"Mean: {mean:g}")
    print(f"75% (Q3): {q3:g}")
    print(f"Upper whisker: {upper_whisker:g}")
