import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, max as pyspark_max
from pyspark.ml.feature import StandardScaler
from pyspark.ml import Pipeline

# Step 1: Load and preprocess data with Pandas
olympics_data = pd.read_csv('olympics_data.csv')

# Step 2: Perform EDA and visualization with Pandas, Numpy, Seaborn, and Matplotlib
# Example: Visualization to find dominant countries and sports categories
top_countries = olympics_data[(olympics_data['Year'] >= 2000) & (olympics_data['Year'] <= 2012)]
top_countries = top_countries.groupby('Country')['Medal'].count().nlargest(3)
top_countries.plot(kind='bar', xlabel='Country', ylabel='Number of Medals', title='Top 3 Countries with Most Medals (2000-2012)')
plt.show()

# Step 3: Transfer cleaned dataset to MySQL using SQL Alchemy
engine = create_engine('mysql+mysqlconnector://username:password@localhost:3306/olympics')
olympics_data.to_sql(name='olympics_data', con=engine, if_exists='replace', index=False)

# Step 4: Analyze data with Pyspark
spark = SparkSession.builder.appName('OlympicsAnalysis').getOrCreate()
spark_df = spark.read.format('jdbc').options(
    url='jdbc:mysql://localhost:3306/olympics',
    driver='com.mysql.jdbc.Driver',
    dbtable='olympics_data',
    user='username',
    password='password'
).load()

# Example: Finding oldest player and their details
oldest_player = spark_df.select('Name', 'Country', 'Age').orderBy(col('Age').desc()).limit(1).collect()

# Step 5: Apply log transformation for data normalization
# Example: Log transformation on numeric columns
numeric_cols = ['Height', 'Weight']
for col_name in numeric_cols:
    spark_df = spark_df.withColumn(col_name, col(col_name).cast('double'))
    spark_df = spark_df.withColumn(col_name + '_log', F.log(spark_df[col_name] + 1))

# Step 6: Visualization to identify top countries with most medals
# Example: Using Seaborn and Matplotlib for visualization
top_countries = spark_df.filter((col('Year') >= 2000) & (col('Year') <= 2012))
top_countries = top_countries.groupBy('Country').agg(F.count('Medal').alias('Total Medals')).orderBy(col('Total Medals').desc()).limit(3)
top_countries_df = top_countries.toPandas()
sns.barplot(x='Country', y='Total Medals', data=top_countries_df)
plt.title('Top 3 Countries with Most Medals (2000-2012)')
plt.xlabel('Country')
plt.ylabel('Number of Medals')
plt.show()

# Step 7: Further analysis and insights with Pyspark
# Example: Additional insights or machine learning models with Pyspark

# Step 8: Close Spark session
spark.stop()
