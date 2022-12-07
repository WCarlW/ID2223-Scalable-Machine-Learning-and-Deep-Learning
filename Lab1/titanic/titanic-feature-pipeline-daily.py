import os
import modal

LOCAL=False

if LOCAL == False:
   stub = modal.Stub("titanic_daily")
   image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4"])

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("Titanic"))
   def f():
       g()


def generate_passenger(Pclass_max,Pclass_min, Sex, Age_max, Age_min,SibSp_max,SibSp_min,Survived):
    """
    Returns a single iris flower as a single row in a DataFrame
    """
    import pandas as pd
    import random
    import numpy as np

    df = pd.DataFrame({ "Pclass": [int((np.round(random.uniform(Pclass_max, Pclass_min))))],
                       "Sex": [Sex],
                       "Age": [random.uniform(Age_max, Age_min)],
                       "SibSp": [int(random.uniform(SibSp_max, SibSp_min))]
                      })
    df['Survived'] = Survived
    return df


def get_random_passenger():
    """
    Returns a DataFrame containing one random iris flower
    """
    import pandas as pd
    import random

    survived_df = generate_passenger(
        3, 0, "1", 80, 0, 8, 0, 1)
    died_df = generate_passenger(
        3, 0, "2", 80, 0, 8, 0, 0)

    # randomly pick one of these 3 and write it to the featurestore
    pick_random = random.uniform(0,2)
    if pick_random >= 1:
        passenger_df = survived_df
        print("survived added")
    else:
        passenger_df = died_df
        print("died added")

    return passenger_df


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login(api_key_value= "U1TTeOPaUDFWhd6N.H604QVLj5yOFVPGeSrXsoFY2IKrGkdqR0iTzWMr22rZxXQrn5VoYKdb4fghqxTna")
    fs = project.get_feature_store()

    titanic_df = get_random_passenger()

    titanic_fg = fs.get_feature_group(name="titanic_modal",version=3)
    titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        stub.deploy("titanic_daily")
        with stub.run():
            f()
