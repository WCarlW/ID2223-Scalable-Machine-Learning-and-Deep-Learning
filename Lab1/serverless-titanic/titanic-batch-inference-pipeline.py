import os
import modal
    
LOCAL=True

if LOCAL == False:
   stub = modal.Stub()
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4","joblib","seaborn","sklearn","dataframe-image"])
   @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()

def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests

    project = hopsworks.login()
    fs = project.get_feature_store()
    
    mr = project.get_model_registry()
    model = mr.get_model("titanic", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/titanic_model.pkl")
    
    feature_view = fs.get_feature_view(name="titanic", version=4)
    batch_data = feature_view.get_batch_data()
    
    y_pred = model.predict(batch_data)

    survived_pic_path = "https://cdn.pixabay.com/photo/2016/11/21/13/58/ball-1845546_960_720.jpg"
    died_pic_path = "https://cdn.pixabay.com/photo/2016/07/22/01/22/sad-1533965__340.jpg"

    # print("Prediction is: ", y_pred)
    offset = 10
    survived = y_pred[y_pred.size-offset]
    # flower_url = "https://raw.githubusercontent.com/featurestoreorg/serverless-ml-course/main/src/01-module/assets/" + flower + ".png"
    # print("Flower predicted: " + flower)
    if int(survived) == 1:
        url = survived_pic_path
    else:
        url = died_pic_path
    
    print("Survived predicted: " + str(survived))
    img = Image.open(requests.get(url, stream=True).raw)            
    img.save("./latest_titanic.png")
    dataset_api = project.get_dataset_api()    
    dataset_api.upload("./latest_titanic.png", "Resources/images", overwrite=True)
   
    titanic_fg = fs.get_feature_group(name="titanic", version=4)
    df = titanic_fg.read() 
    #print(df)
    label = df.iloc[-offset]["survived"]
    # label_url = "https://raw.githubusercontent.com/featurestoreorg/serverless-ml-course/main/src/01-module/assets/" + label + ".png"
    if int(label) == 1:
        label_url = survived_pic_path
    else:
        label_url = died_pic_path
    print("Survived actual: " + str(label))
    img = Image.open(requests.get(label_url, stream=True).raw)            
    img.save("./actual_titanic.png")
    dataset_api.upload("./actual_titanic.png", "Resources/images", overwrite=True)
    
    monitor_fg = fs.get_or_create_feature_group(name="titanic_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="Titanic Prediction/Outcome Monitoring"
                                                )
    
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [survived],
        'label': [label],
        'datetime': [now],
       }
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})
    
    history_df = monitor_fg.read()
    # Add our prediction to the history, as the history_df won't have it - 
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])


    df_recent = history_df.tail(4)
    dfi.export(df_recent, './df_recent.png', table_conversion = 'matplotlib')
    dataset_api.upload("./df_recent.png", "Resources/images", overwrite=True)
    
    predictions = history_df[['prediction']]
    labels = history_df[['label']]

    # Only create the confusion matrix when our titanic_predictions feature group has examples of all 2 titanic situations
    print("Number of different titanic predictions to date: " + str(predictions.value_counts().count()))
    if predictions.value_counts().count() == 2:
        results = confusion_matrix(labels, predictions)
    
        df_cm = pd.DataFrame(results, ['True Survived', 'True Dead'],
                             ['Pred Survived', 'Pred Dead'])
    
        cm = sns.heatmap(df_cm, annot=True)
        fig = cm.get_figure()
        fig.savefig("./confusion_matrix.png")
        dataset_api.upload("./confusion_matrix.png", "Resources/images", overwrite=True)
    else:
        print("You need 2 different titanic predictions to create the confusion matrix.")
        print("Run the batch inference pipeline more times until you get 2 different titanic predictions") 


if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()

