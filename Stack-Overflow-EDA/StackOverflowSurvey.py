# Project Goal
# This project explores the Stack Overflow Developer Survey dataset to understand:
# 	Developer roles
# 	Salary distribution
# 	Remote work preferences
# 	Education levels
# 	Popular programming languages
# 	Relationship between experience and salary
# The project also demonstrates:
# 	Missing value handling
# 	Data cleaning
# 	Feature engineering
# 	Exploratory Data Analysis (EDA)
# 	Visualization best practices
# ________________________________________

# 1. IMPORT LIBRARIES

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# Display settings
pd.set_option('display.max_rows', None)
plt.ion()

# 2. LOAD DATASET

# Load Stack Overflow survey dataset

df = pd.read_csv('results.csv')

# Total rows in dataset
max_rows = len(df)

print(f"Dataset Shape: {df.shape}")
# Observation
# Dataset contains thousands of developer survey responses.
# Many columns contain large amounts of missing data.
# Some columns are not useful for analysis.

# 3. INITIAL DATA EXPLORATION

print(df.head())
print(df.info())
print(df.describe())
# Observation
# 	Dataset contains both numerical and categorical data.
# 	Some columns contain multi-value responses separated by semicolons.
# 	Salary and experience columns contain outliers.

# 4. MISSING VALUE ANALYSIS

missing_data = round((1 - df.count() / max_rows) * 100, 2)

print(missing_data.sort_values(ascending=False).head(20))
# Visualize Missing Data
msno.bar(df)
plt.title('Missing Data Overview')
plt.tight_layout()
plt.show()
# Observation
# 	Several AI-related survey fields contain more than 90% missing values.
# 	These columns provide very little usable information.
# 	Removing highly incomplete columns improves analysis quality.

# 5. DROP HIGHLY MISSING COLUMNS

columns_to_drop = [
    'AIAgentObsWrite',
    'SOTagsWant Entry',
    'SOTagsHaveEntry',
    'AIModelsWantEntry',
    'AIAgentOrchWrite',
    'JobSatPoints_15_TEXT',
    'AIAgentKnowWrite',
    'AIModelsHaveEntry',
    'SO_Actions_15_TEXT',
    'AIAgentExtWrite',
    'CommPlatformWantEntr',
    'CommPlatformHaveEntr',
    'DatabaseWantEntry',
    'OfficeStackWantEntry',
    'TechOppose_15_TEXT',
    'TechEndorse_13_TEXT',
    'DevEnvWantEntry',
    'DatabaseHaveEntry',
    'OfficeStackHaveEntry',
    'WebframeWantEntry',
    'AIAgentObserveSecure',
    'DevEnvHaveEntry',
    'PlatformWantEntry',
    'LanguagesWantEntry',
    'WebframeHaveEntry',
    'AIAgentKnowledge',
    'AIAgentOrchestration',
    'AIAgentImpactStrongly disagree',
    'PlatformHaveEntry',
    'LanguagesHaveEntry',
    'AIAgentImpactSomewhat disagree',
    'AgentUsesGeneral',
    'AIAgentImpactStrongly agree',
    'AIAgentExternal',
    'AIAgentChallengesStrongly disagree',
    'AIAgentImpactNeutral'
]

# Drop unnecessary columns

df.drop(columns=columns_to_drop, inplace=True)

print(df.shape)
# Observation
# 	Dataset becomes easier to manage after removing noisy columns.
# 	Reduced dimensionality improves readability and performance.

# 6. CLEAN IMPORTANT COLUMNS

# Remove Rows Missing Core Information
important_columns = ['Country', 'Employment', 'DevType']

# Remove rows missing critical information

df.dropna(subset=important_columns, inplace=True)

print(df.shape)
# Observation
# 	Rows missing country, employment, or developer type are not useful for workforce analysis.
# 	Missingness appears related across these columns.


# 7. EMPLOYMENT ANALYSIS

employment_plot = sns.catplot(
    x='Country',
    col='Employment',
    data=df,
    kind='count',
    height=6,
    aspect=1.5
)
plt.tight_layout()
plt.show()
# Observation
# 	Employment type varies significantly by country.
# 	Full-time employment dominates the survey responses.

# 8. DEVELOPER ROLE ANALYSIS

# Create smaller dataframe

dev_df = df[['Country', 'DevType']].copy()

# Create role flags

dev_df.loc[dev_df['DevType'].str.contains('back-end', case=False, na=False), 'BackEnd'] = True
dev_df.loc[dev_df['DevType'].str.contains('front-end', case=False, na=False), 'FrontEnd'] = True
dev_df.loc[dev_df['DevType'].str.contains('full-stack', case=False, na=False), 'FullStack'] = True
dev_df.loc[dev_df['DevType'].str.contains('mobile', case=False, na=False), 'Mobile'] = True
dev_df.loc[dev_df['DevType'].str.contains('administrator', case=False, na=False), 'Admin'] = True
# Reshape Data
# Convert wide format to long format

dev_df = dev_df.melt(
    id_vars=['Country'],
    value_vars=['BackEnd', 'FrontEnd', 'FullStack', 'Mobile', 'Admin'],
    var_name='DeveloperCategory',
    value_name='DeveloperFlag'
)

# Remove null rows

dev_df.dropna(inplace=True)

# Top Countries Analysis
# Select top 5 countries

top_countries = dev_df['Country'].value_counts().head(5).index

filtered_dev_df = dev_df[
    dev_df['Country'].isin(top_countries)
]

# Plot developer categories

dev_plot = sns.catplot(
    x='Country',
    col='DeveloperCategory',
    data=filtered_dev_df,
    kind='count',
    height=6,
    aspect=1.5
)
plt.tight_layout()
plt.show()
# Observation
# •	Full-stack and back-end development roles appear most common.
# •	Developer specialization differs across countries.

# 9. EDUCATION ANALYSIS

# Remove missing education rows

df.dropna(subset=['EdLevel'], inplace=True)

# Count education levels

education_counts = (
    df['EdLevel']
    .value_counts()
    .reset_index()
)

education_counts.columns = ['Education', 'Count']
# Simplify Education Labels
education_counts['Education'] = education_counts['Education'].replace({
    'Bachelor’s degree (B.A., B.S., B.Eng., etc.)': 'Bachelor',
    'Master’s degree (M.A., M.S., M.Eng., MBA, etc.)': 'Master',
    'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)': 'High School',
    'Some college/university study without earning a degree': 'Some College',
    'Professional degree (JD, MD, Ph.D, Ed.D, etc.)': 'Professional Degree',
    'Associate degree (A.A., A.S., etc.)': 'Associate Degree',
    'Primary/elementary school': 'Primary School',
    'Other (please specify)': 'Other'
})
# Visualization
plt.figure(figsize=(10, 5))

sns.barplot(
    x='Education',
    y='Count',
    data=education_counts
)

plt.xticks(rotation=45)
plt.title('Education Level Distribution')
plt.xlabel('Education Level')
plt.ylabel('Number of Developers')
plt.tight_layout()
plt.show()
# Observation
# 	Bachelor’s degrees are the most common education level.
# 	Many developers also enter the industry without formal degrees.

# 10. SALARY ANALYSIS

# Remove Missing Salary Data
salary_df = df.dropna(subset=['ConvertedCompYearly']).copy()
# Salary Statistics
print(f"Mean Salary: {salary_df['ConvertedCompYearly'].mean():,.0f}")
print(f"Median Salary: {salary_df['ConvertedCompYearly'].median():,.0f}")
print(f"Max Salary: {salary_df['ConvertedCompYearly'].max():,.0f}")
print(f"Min Salary: {salary_df['ConvertedCompYearly'].min():,.0f}")
# Observation
# 	Mean salary is much higher than median salary.
# 	This suggests a right-skewed distribution.
# 	A small number of very high salaries increase the average.
# ________________________________________
# Remove Extreme Salary Outliers
salary_filtered = salary_df[
    (salary_df['ConvertedCompYearly'] > 10000) &
    (salary_df['ConvertedCompYearly'] < 1000000)
]
# Salary Histogram
plt.figure(figsize=(10, 5))

plt.hist(
    salary_filtered['ConvertedCompYearly'],
    bins=100
)

plt.title('Salary Distribution')
plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.show()
# Salary Boxplot
plt.figure(figsize=(10, 5))

plt.boxplot(
    salary_filtered['ConvertedCompYearly']
)

plt.title('Salary Boxplot')
plt.ylabel('Salary')
plt.show()
# Observation
# 	Most salaries cluster within a smaller lower range.
# 	The dataset still contains some high-income outliers.
# 	Salary distribution is not normally distributed.

# 11. REMOTE WORK ANALYSIS

remote_df = df.dropna(subset=['RemoteWork']).copy()
# Count Work Preferences
remote_counts = (
    remote_df['RemoteWork']
    .value_counts()
    .reset_index()
)

remote_counts.columns = ['RemoteWork', 'Count']
# Simplify Labels
remote_counts['RemoteWork'] = remote_counts['RemoteWork'].replace({
    'Remote': 'WFH',
    'Hybrid (some remote, leans heavy to in-person)': 'Hybrid - Office Heavy',
    'Hybrid (some in-person, leans heavy to flexibility)': 'Hybrid - Flexible',
    'In-person': 'Office',
    'Your choice (very flexible, you can come in when you want or just as needed)': 'Flexible Choice'
})
# Visualization
remote_counts.plot(
    kind='bar',
    x='RemoteWork',
    y='Count',
    figsize=(10, 5)
)

plt.title('Remote Work Preference')
plt.xlabel('Work Type')
plt.ylabel('Number of Developers')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# Observation
# 	Hybrid and remote work models are highly preferred.
# 	Fully in-office work appears less common.

# 12. PROGRAMMING LANGUAGE ANALYSIS

languages = df['LanguageHaveWorkedWith'].dropna()
# Split Multi-Value Responses
languages = languages.str.split(';')
languages = languages.explode()
# Top Languages
top_languages = languages.value_counts().head(10)

print(top_languages)
# Visualization
top_languages.plot(
    kind='bar',
    figsize=(10, 5)
)

plt.title('Top 10 Programming Languages')
plt.xlabel('Programming Language')
plt.ylabel('Number of Developers')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# Observation
# 	JavaScript, Python, and SQL remain among the most commonly used languages.
# 	Multi-language usage is extremely common among developers.


# 13. EXPERIENCE VS SALARY ANALYSIS

# Clean Experience Column
experience_df = df[['YearsCode', 'ConvertedCompYearly']].copy()

# Remove Missing Values
experience_df.dropna(inplace=True)
# Remove Outliers
experience_df = experience_df[
    (experience_df['YearsCode'] < 50) &
    (experience_df['ConvertedCompYearly'] < 1000000)
]
# Scatter Plot
plt.figure(figsize=(12, 6))

plt.scatter(
    experience_df['YearsCode'],
    experience_df['ConvertedCompYearly'],
    alpha=0.3,
    s=10
)

plt.title('Experience vs Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
# Observation
# 	Salary generally increases with experience.
# 	Salary spread becomes wider at higher experience levels.
# 	Some experienced developers earn significantly more than others.
# 	The relationship is positive but not perfectly linear.

input("press enter to close")

# 14. FINAL INSIGHTS
# =========================
# Key Findings
# 1. Salary Distribution
# •	Salary data is heavily right-skewed.
# •	Median salary better represents typical developer income.
# •	Outlier removal improves visualization quality.
# 2. Remote Work
# •	Remote and hybrid work models dominate developer preferences.
# •	Flexible work arrangements remain important in the industry.
# 3. Developer Roles
# •	Full-stack and back-end development are highly common.
# •	Role popularity varies by country.
# 4. Education
# •	Formal degrees remain common.
# •	Many developers enter the industry through non-traditional paths.
# 5. Experience vs Salary
# •	Experience positively impacts salary.
# •	Salary variation increases for senior developers.
# ________________________________________
# =========================
# 15. PROJECT LEARNINGS
# =========================
# This project demonstrates:
# •	Data cleaning
# •	Missing value handling
# •	Outlier treatment
# •	Feature engineering
# •	Exploratory data analysis
# •	Data visualization
# •	Real-world dataset challenges
# It also highlights the importance of:
# •	Cleaning messy data
# •	Understanding distributions
# •	Handling categorical variables
# •	Interpreting visual patterns
# ________________________________________
# End of Project
