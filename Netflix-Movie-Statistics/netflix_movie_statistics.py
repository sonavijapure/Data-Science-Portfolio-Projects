#MOVIE STATISTICS PROJECT
#ANALYSING MOVIE DATASET, CLEANING MISSING AND DUPLICATE DATA, PLOTTING GRAPHS FOR BETTER UNDERSTANDING
#INTERPRETING SUMMARY STATISTICS IN CONTEXT

#IMPORTING LIBRARIES   
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.max_rows', None)



#LOADING CSV FILE TO DATAFRAME
df = pd.read_csv('NetflixOriginals.csv', encoding='latin-1')

#CHECKING DUPLICATES AND MISSING VALUES
print(df.isna().sum())
print(df.duplicated().sum())

#CHECKING PARIMARY LANGAUGE COUNT OF MOVIES
print(df.groupby('Language')['Language'].count().sort_values(ascending=False))

#CREATING ANOTHER COLUMN FOR MULTIPLE LANGUAGES
df['language_type'] = df['Language'].apply(lambda x: 'Multi Language' if '/' in str(x) else x)

movie_langauge  = df['language_type'].value_counts()

#PLOTTING BAR GRAPH TO COMPARE COUNTS OF DIFFERNET MOVIE LANGUAGES
movie_langauge.sort_values().plot.barh( figsize=(8,5), edgecolor='black')
plt.title("Which Languages Dominate Netflix Films?")
plt.ylabel("Languages")
plt.xlabel("Frequency")

plt.tight_layout()
plt.show()

#CALCULATING PROPORTION OF EACH LANGUAGE
for language, count in movie_langauge.items():
    proportion = count / len(df) * 100
    print(f"Proportion of {language}: {round(proportion,2)}%")


#PLOTTING HISTOGRAM OF IMDB SCORES
plt.figure(figsize=(8,5))
plt.hist(df['IMDB Score'], bins=20, edgecolor='black')

#CALCULATING STATISTICS FOR IMDB SCORES
imdb_mean = df['IMDB Score'].mean()
imdb_median = df['IMDB Score'].median()
imdb_std = df['IMDB Score'].std()

#PLOTTING STATISTICAL VALUES OF IMDB SCORES
plt.axvline(imdb_mean, color='red', linestyle='dashed', linewidth=2, label='Mean')
plt.axvline(imdb_median, color='green', linestyle='dashed', linewidth=2, label='Median')
plt.axvline(imdb_mean - imdb_std, color='purple', linestyle='dashed', linewidth=1, label='±1 Std Dev')
plt.axvline(imdb_mean + imdb_std, color='purple', linestyle='dashed', linewidth=1)

plt.title("Distribution of IMDB Scores")
plt.xlabel("IMDB Score")
plt.ylabel("Number of Movies")

plt.legend()
plt.tight_layout()
plt.show()

#BAR PLOTTING FOR MOVIE GENRE
genre_count = df['Genre'].value_counts()

plt.figure(figsize=(8,5))
plt.bar(genre_count.index, genre_count.values,  edgecolor='black')
plt.xticks(rotation=90)
plt.title('Distribution of Movie Genre')
plt.xlabel('Movie Genre')
plt.ylabel('Number of Movies')

plt.tight_layout()
plt.show()


genre_df = df.copy()
genre_df['Genre'] = genre_df['Genre'].str.split('/')
genre_df = genre_df.explode('Genre')
genre_df['Genre'] = genre_df['Genre'].str.strip()

#IMDB SCORES BY GENRE
genre_rating = genre_df.groupby('Genre')['IMDB Score'].mean().sort_values(ascending=False)

plt.figure(figsize=(8,5))
plt.bar(genre_rating.index, genre_rating.values,  edgecolor='black')
plt.xticks(rotation=90)
plt.title('Average Mean Distribution of Movie Genre by IMDB ratings')
plt.xlabel('Movie Genre')
plt.ylabel('IMDB Raitng')

plt.tight_layout()
plt.show()


#RUNTIME IN MINUTES
plt.figure(figsize=(8,5))
plt.hist(df['Runtime'], bins=20, edgecolor='black')
plt.title('Runtime Distribution')
plt.xlabel('Runtime in Minutes')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# Runtime distribution is skewed and contains outliers,
# so median and IQR are used as robust measures.
# IQR method is also used to identify acceptable lower and upper bounds for runtime values.
runtime_q1 = df['Runtime'].quantile(0.25)
runtime_q3 = df['Runtime'].quantile(0.75)
runtime_iqr = runtime_q3 - runtime_q1
runtime_lower = runtime_q1 - (1.5 * runtime_iqr)
runtime_upper = runtime_q3 + (1.5 * runtime_iqr)

runtime_df = df[(df['Runtime'] > runtime_lower) & (df['Runtime'] < runtime_upper)]
runtime_median = runtime_df['Runtime'].median()

plt.figure(figsize=(8,5))
plt.hist(runtime_df['Runtime'], bins=20, edgecolor='black', label='Runtime')
plt.axvline(runtime_median, color='green', linestyle='dashed', linewidth=2, label='Median')
plt.axvline(runtime_q1, color='purple', linestyle='dashed', linewidth=1, label='Rutime Q1')
plt.axvline(runtime_q3, color='purple', linestyle='dashed', linewidth=1, label='Rutime Q3')
plt.axvline(runtime_lower, color='red', linestyle='dashed', linewidth=1, label='Runtime Lower Bound')
plt.axvline(runtime_upper, color='red', linestyle='dashed', linewidth=1, label='Runtime Upper Bound')
plt.title('Runtime Distribution After Outlier Removal')
plt.xlabel('Runtime in Minutes')
plt.ylabel('Frequency')

plt.legend()
plt.tight_layout()
plt.show()


#REALTIONSHIP BETWEEN IMDB SCORE BY RUNTIME
plt.figure(figsize=(8,5))

plt.scatter(df['Language'],df['Runtime'],alpha=0.5, edgecolors='black')
plt.xticks(rotation=90)

plt.title("Runtime vs Language")
plt.xlabel("Language")
plt.ylabel("IMDB Score")

plt.tight_layout()
plt.show()

#OBSERVATIONS -
# Runtime varies across languages but no clear pattern is visible.
# Points are spread out across languages.
# Most languages contain a mix of short and long runtime movies.


#PLOT AND SUMMARY OF IMDB SCORE BY RUNTIME
#FINDING CORRELATION COEFFICIENT BETWEEN IMDB SCORE AND RUNTIME
r = df['Runtime'].corr(df['IMDB Score'])
plt.figure(figsize=(8,5))

plt.scatter(df['Runtime'], df['IMDB Score'], alpha=0.5, edgecolors='black')

plt.title("Runtime vs IMDB Score")
plt.xlabel("Runtime")
plt.ylabel("IMDB Score")

# SHOW r ON GRAPH OF "Runtime vs IMDB Score"
plt.text(
    0.05, 0.95,
    f"r = {r:.2f}",
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment="top"
)
plt.legend()
plt.tight_layout()
plt.show()

#OBSERVATIONS -
# Very weak or no linear relationship between runtime and IMDB score.
# Points are spread out with no clear upward or downward trend.