import os
import json
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    Model, IntegerField, FloatField,
    TextField, IntegrityError,BooleanField
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect
import warnings
warnings.filterwarnings('ignore')



########################################
# Begin database stuff

# The connect function checks if there is a DATABASE_URL env var.
# If it exists, it uses it to connect to a remote postgres db.
# Otherwise, it connects to a local sqlite db stored in predictions.db.
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class Prediction(Model):
    observation_id = TextField(unique=True)
    observation = TextField()
    label = BooleanField()
    
    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)

# End database stuff
########################################

########################################
# Unpickle the previously-trained model


with open('columns.json') as fh:
    columns = json.load(fh)

pipeline = joblib.load('pipeline.pickle')

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)


# End model un-pickling
########################################


########################################
# Begin webserver stuff

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # Flask provides a deserialization convenience function called
    # get_json that will work if the mimetype is application/json.
    observation = request.get_json()
    
    ## a single observation into a dataframe that will work with a pipeline.
    obs = pd.DataFrame([observation])#.astype(dtypes)
    _id = obs['observation_id'].values[0]
    ## Now get ourselves an actual prediction of the positive class.
    label = pipeline.predict(obs)[0]
    response = {'observation_id':_id,'label': str(label)}
    p = Prediction(
        observation_id=_id,
        label=label,
        observation=request.data
    )
    try:
        p.save()
    except IntegrityError:
        error_msg = 'Observation ID: "{}" already exists'.format(_id)
        response['error'] = error_msg
        print(error_msg)
        DB.rollback()
    return jsonify(response)


@app.route('/update', methods=['POST'])
def update():
    observation = request.get_json()
    obs = pd.DataFrame([observation])#, columns=columns).astype(dtypes)
    try:
        p = Prediction.get(Prediction.observation_id == obs['observation_id'].values[0])
        p.label = str(obs['label'].values[0])
        p.save()
        response = {'observation_id':obs['observation_id'].values[0],'label': str(obs['label'].values[0])}
        return jsonify(response)
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(obs['observation_id'].values[0])
        return jsonify({'error': error_msg})


@app.route('/list-db-contents')
def list_db_contents():
    return jsonify([
        model_to_dict(obs) for obs in Prediction.select()
    ])


# End webserver stuff
########################################

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)


"""
    
curl -X POST http://localhost:5000/predict -d '{"observation_id": "8b2de40d-d98b-4cb5-aa49-f471gbja8b","Type": "Entity inspection","Date": "3919-08-16 14:37:00+00:00","Part of a standard enforcement protocol": true,"Galactic X": 3434.23,"Galactic Y": 2321.12,"Reproduction": "Sexual","Age range": "Young","Self-defined species category": "Terran - Northern","Officer-defined species category": "Terran","Governing law": "Intergalactic Substance Regulation 3919","Object of inspection": "Controlled substances","Inspection involving more than just outerwear": false,"Enforcement station": "Dyson Sphere F76-JK"}' -H "Content-Type:application/json"

curl -X POST http://localhost:5000/update -d '{"observation_id": "8b2de40d-d98b-4cb5-aa49-f471gbja8d","label": true}' -H "Content-Type:application/json"


"""


