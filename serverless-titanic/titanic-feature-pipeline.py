import os
import modal
#import great_expectations as ge
import hopsworks
import pandas as pd
import numpy as np

project = hopsworks.login()
fs = project.get_feature_store()

titanic_df = pd.read_csv("https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv")
df = titanic_df[["Pclass", "Sex", "Age", "SibSp", "Survived"]]

# Drop rows containing NaN/None
df_final = df.dropna()

# Conver strings in Sex feature to int for classifier
df_final.Sex[df_final.Sex == "male"] = "1"
df_final.Sex[df_final.Sex == "female"] = "2"

titanic_fg = fs.get_or_create_feature_group(
    name="titanic",
    version=4,
    primary_key=["Pclass","Sex","Age","SibSp"], 
    description="Titanic dataset")
titanic_fg.insert(df_final, write_options={"wait_for_job" : False})

#expectation_suite = ge.core.ExpectationSuite(expectation_suite_name="iris_dimensions")    
#value_between(expectation_suite, "sepal_length", 4.5, 8.0)
#value_between(expectation_suite, "sepal_width", 2.1, 4.5)
#value_between(expectation_suite, "petal_length", 1.2, 7)
#value_between(expectation_suite, "petal_width", 0.2, 2.5)
#iris_fg.save_expectation_suite(expectation_suite=expectation_suite, validation_ingestion_policy="STRICT")    
    

