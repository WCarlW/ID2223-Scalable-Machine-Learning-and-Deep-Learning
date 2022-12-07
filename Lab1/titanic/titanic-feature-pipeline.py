import os
import modal
#import great_expectations as ge
import hopsworks
import pandas as pd
import numpy as np


project = hopsworks.login()
fs = project.get_feature_store()

titanic_df = pd.read_csv("https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv")
titanic_df = titanic_df[["Pclass", "Sex", "Age","SibSp","Survived"]]

# Drop rows containing NaN/None
titanic_df = titanic_df.dropna()

# Conver strings in Sex feature to int for classifier
titanic_df.Sex[titanic_df.Sex == "male"] = "1"
titanic_df.Sex[titanic_df.Sex == "female"] = "2"
titanic_fg = fs.get_or_create_feature_group(
    name="titanic_modal",
    version=3,
    primary_key=["Pclass", "Sex", "Age", "SibSp"],
    description="Titanic dataset")
titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})

#expectation_suite = ge.core.ExpectationSuite(expectation_suite_name="iris_dimensions")    
#value_between(expectation_suite, "sepal_length", 4.5, 8.0)
#value_between(expectation_suite, "sepal_width", 2.1, 4.5)
#value_between(expectation_suite, "petal_length", 1.2, 7)
#value_between(expectation_suite, "petal_width", 0.2, 2.5)
#iris_fg.save_expectation_suite(expectation_suite=expectation_suite, validation_ingestion_policy="STRICT")    
    

