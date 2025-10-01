import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set a style for the plots for better aesthetics
sns.set(style="whitegrid")

def generate_synthetic_data(num_students=50):
    """
    Generates a synthetic dataset for student performance analysis.
    This allows the script to be runnable without an external CSV file.
    """
    genders = ['male', 'female']
    races = ['group A', 'group B', 'group C', 'group D', 'group E']
    parental_education = ["some high school", "high school", "some college", 
                          "associate's degree", "bachelor's degree", "master's degree"]
    lunch_types = ['standard', 'free/reduced']
    test_prep_courses = ['none', 'completed']

    data = {
        'gender': np.random.choice(genders, num_students),
        'race/ethnicity': np.random.choice(races, num_students),
        'parental_education': np.random.choice(parental_education, num_students),
        'lunch': np.random.choice(lunch_types, num_students),
        'test_preparation_course': np.random.choice(test_prep_courses, num_students),
        'math_score': np.random.randint(0, 101, num_students),
        'reading_score': np.random.randint(0, 101, num_students),
        'writing_score': np.random.randint(0, 101, num_students)
    }

    df = pd.DataFrame(data)
    
    # Introduce some correlations to make the data more realistic
    df['reading_score'] = np.clip(df['reading_score'] + np.random.randint(0, 10, num_students) + df['test_preparation_course'].map({'completed': 10, 'none': 0}), 0, 100)
    df['writing_score'] = np.clip(df['writing_score'] + np.random.randint(0, 10, num_students) + df['test_preparation_course'].map({'completed': 10, 'none': 0}), 0, 100)
    df['math_score'] = np.clip(df['math_score'] + np.random.randint(0, 10, num_students) + df['test_preparation_course'].map({'completed': 10, 'none': 0}), 0, 100)
    
    return df

def analyze_and_visualize(df):
    """
    Performs data analysis and generates visualizations.
    """
    print("--- Dataset Information ---")
    df.info()
    print("\n--- Descriptive Statistics ---")
    print(df.describe())
    
    # Add a column for the average score
    df['average_score'] = df[['math_score', 'reading_score', 'writing_score']].mean(axis=1)

    # Visualization 1: Distribution of Student Scores
    plt.figure(figsize=(15, 5))
    plt.suptitle('Distribution of Student Scores', fontsize=16)
    
    plt.subplot(1, 3, 1)
    sns.histplot(df['math_score'], kde=True, bins=20, color='skyblue')
    plt.title('Math Scores')
    
    plt.subplot(1, 3, 2)
    sns.histplot(df['reading_score'], kde=True, bins=20, color='salmon')
    plt.title('Reading Scores')
    
    plt.subplot(1, 3, 3)
    sns.histplot(df['writing_score'], kde=True, bins=20, color='lightgreen')
    plt.title('Writing Scores')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Visualization 2: Box plots to compare scores by gender
    plt.figure(figsize=(12, 6))
    plt.suptitle('Score Distribution by Gender', fontsize=16)
    
    plt.subplot(1, 3, 1)
    sns.boxplot(x='gender', y='math_score', data=df)
    plt.title('Math Scores')
    
    plt.subplot(1, 3, 2)
    sns.boxplot(x='gender', y='reading_score', data=df)
    plt.title('Reading Scores')
    
    plt.subplot(1, 3, 3)
    sns.boxplot(x='gender', y='writing_score', data=df)
    plt.title('Writing Scores')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Visualization 3: Average scores by parental education level
    parental_edu_order = ["some high school", "high school", "some college", 
                          "associate's degree", "bachelor's degree", "master's degree"]
    
    plt.figure(figsize=(14, 7))
    sns.barplot(
        x='parental_education',
        y='average_score',
        data=df,
        order=parental_edu_order,
        palette='viridis',
        ci=None
    )
    plt.title('Average Score by Parental Education Level', fontsize=16)
    plt.xlabel('Parental Education', fontsize=12)
    plt.ylabel('Average Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Visualization 4: Impact of test preparation course on scores
    plt.figure(figsize=(12, 6))
    plt.suptitle('Impact of Test Preparation Course on Average Scores', fontsize=16)
    sns.barplot(x='test_preparation_course', y='average_score', data=df, palette='pastel')
    plt.xlabel('Test Preparation Course', fontsize=12)
    plt.ylabel('Average Score', fontsize=12)
    plt.show()

    # Visualization 5: Correlation matrix of the scores
    score_columns = ['math_score', 'reading_score', 'writing_score']
    correlation_matrix = df[score_columns].corr()
    
    plt.figure(figsize=(8, 7))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Student Scores', fontsize=16)
    plt.show()

if __name__ == "__main__":
    # Generate the dataset
    student_data = generate_synthetic_data()
    
    # Run the analysis and visualization
    analyze_and_visualize(student_data)

