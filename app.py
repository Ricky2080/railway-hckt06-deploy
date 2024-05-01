import os
import json
import pickle
import joblib
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from peewee import (
    SqliteDatabase, PostgresqlDatabase, Model, IntegerField,
    FloatField, TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect


########################################
# Begin database stuff

# The connect function checks if there is a DATABASE_URL env var.
# If it exists, it uses it to connect to a remote postgres db.
# Otherwise, it connects to a local sqlite db stored in predictions.db.
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    proba = FloatField()
    true_class = IntegerField(null=True)

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

# Input validation functions
def check_request(request):
    """
        Validates that our request is well formatted
        
        Returns:
        - assertion value: True if request is ok, False otherwise
        - error message: empty if request is ok, False otherwise
    """
    
    if "observation_id" not in request:
        error = "Field `observation_id` missing from request: {}".format(request)
        return False, error
    
    if "data" not in request:
        error = "Field `data` missing from request: {}".format(request)
        return False, error
    
    return True, ""

def check_valid_column(observation):
    """
        Validates that our observation only has valid columns
        
        Returns:
        - assertion value: True if all provided columns are valid, False otherwise
        - error message: empty if all provided columns are valid, False otherwise
    """
    
    valid_columns = {
                "age",
                "sex",
                "race",
                "workclass",
                "education",
                "marital-status",
                "capital-gain",
                "capital-loss",
                "hours-per-week"
    }
    
    keys = set(observation.keys())
    #print('len(valid_columns - keys):',len(valid_columns - keys))
    if len(valid_columns - keys) > 0: 
        missing = valid_columns - keys
        error = "Missing columns: {}".format(missing)
        return False, error
    
    if len(keys - valid_columns) > 0: 
        extra = keys - valid_columns
        error = "Unrecognized columns provided: {}".format(extra)
        return False, error    

    if len(keys) != 9:
        extra = keys - valid_columns
        error = "Number of columns provided is not correct: {}".format(extra)
        return False, error  
    
    return True, ""

def check_categorical_values(observation):
    """
        Validates that all categorical fields are in the observation and values are valid
        
        Returns:
        - assertion value: True if all provided categorical columns contain valid values, 
                           False otherwise
        - error message: empty if all provided columns are valid, False otherwise
    """
    
    valid_category_map = {
        "sex": ['Male', 'Female'],
        "race": ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo','Other'],
        "workclass": ['State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov','Local-gov',
                      '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'],
        "education": ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
                      'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
                      '5th-6th', '10th', '1st-4th', 'Preschool', '12th'],
        "marital-status": ['Never-married', 'Married-civ-spouse', 'Divorced',
                           'Married-spouse-absent', 'Separated', 'Married-AF-spouse','Widowed']
    }
    
    for key, valid_categories in valid_category_map.items():
        if key in observation:
            value = observation[key]
            if value not in valid_categories:
                error = "Invalid value provided for {}: {}. Allowed values are: {}".format(
                    key, value, ",".join(["'{}'".format(v) for v in valid_categories]))
                return False, error
        else:
            error = "Categorical field {} missing"
            return False, error

    return True, ""

def check_age(observation):
    """
        Validates that observation contains valid hour value 
        
        Returns:
        - assertion value: True if age is valid, False otherwise
        - error message: empty if age is valid, False otherwise
    """
    
    age = observation.get("age")
        
    if age is None: 
        error = "Field `age` missing"
        return False, error

    if (not isinstance(age, int)) or np.isnan(age) or np.isinf(age):
    #if np.isnan(age) or np.isinf(age):
        error = "Field `age` is not an integer"
        return False, error
    
    if age < 10 or age > 100:
        error = "Field `age` is not between 10 and 100. `age` is {}".format(age)
        return False, error

    return True, ""

def check_capital_gain(observation):
    """
        Validates that observation contains valid hour value 
        
        Returns:
        - assertion value: True if capital_gain is valid, False otherwise
        - error message: empty if capital_gain is valid, False otherwise
    """
    
    capital_gain = observation.get("capital-gain")
    #print('capital_gain:',capital_gain)    
    if capital_gain is None:
        error = "Field `capital-gain` missing"
        return False, error

    if not isinstance(capital_gain, int):
        error = "Field `capital-gain` is not an integer"
        return False, error
    
    if capital_gain < 0:
        error = "Field `capital-gain` is not a positive integer. `capital-gain` is {}".format(capital_gain)
        return False, error

    return True, ""

def check_capital_loss(observation):
    """
        Validates that observation contains valid hour value 
        
        Returns:
        - assertion value: True if capital_loss is valid, False otherwise
        - error message: empty if capital_loss is valid, False otherwise
    """
    
    capital_loss = observation.get("capital-loss")
    #print('capital_loss:',capital_loss)    
    if capital_loss is None:
        error = "Field `capital-loss` missing"
        return False, error

    if not isinstance(capital_loss, int):
        error = "Field `capital-loss` is not an integer"
        return False, error
    
    if capital_loss < 0:
        error = "Field `capital-loss` is not a positive integer.  `capital-loss` is {}".format(capital_loss)
        return False, error

    return True, ""

def check_hours_per_week(observation):
    """
        Validates that observation contains valid hour value 
        
        Returns:
        - assertion value: True if hours_per_week is valid, False otherwise
        - error message: empty if hours_per_week is valid, False otherwise
    """
    
    hours_per_week = observation.get("hours-per-week")
        
    if hours_per_week is None:
        error = "Field `hours-per-week` missing"
        return False, error

    if not isinstance(hours_per_week, int):
        error = "Field `hours-per-week` is not an integer"
        return False, error
    
    if hours_per_week < 0 or hours_per_week > 168:
        error = "Field `hours-per-week` is not between 0 and 168. `hours-per-week` is {}".format(hours_per_week)
        return False, error

    return True, ""


########################################
# Begin webserver stuff

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    obs_dict = request.get_json()
  
    request_ok, error = check_request(obs_dict)
    if not request_ok:
        response = {'error': error}
        return jsonify(response)

    _id = obs_dict['observation_id']
    observation = obs_dict['data']

    columns_ok, error = check_valid_column(observation)
    if not columns_ok:
        response = {'error': error}
        return jsonify(response)

    categories_ok, error = check_categorical_values(observation)
    if not categories_ok:
        response = {'error': error}
        return jsonify(response)

    age_ok, error = check_age(observation)
    if not age_ok:
        response = {'error': error}
        return jsonify(response)

    capital_gain_ok, error = check_capital_gain(observation)
    if not capital_gain_ok:
        response = {'error': error}
        return jsonify(response)
    
    capital_loss_ok, error = check_capital_loss(observation)
    if not capital_loss_ok:
        response = {'error': error}
        return jsonify(response)
    
    hours_per_week_ok, error = check_hours_per_week(observation)
    if not hours_per_week_ok:
        response = {'error': error}
        return jsonify(response)


    obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
    proba = pipeline.predict_proba(obs)[0, 1]
    prediction = pipeline.predict(obs)[0]
    response = {'prediction': bool(prediction), 'probability': proba}
    p = Prediction(
        observation_id=_id,
        proba=proba,
        observation=request.data,
    )
    try:
        p.save()
    except IntegrityError:
        error_msg = "ERROR: Observation ID: '{}' already exists".format(_id)
        response["error"] = error_msg
        print(error_msg)
        DB.rollback()
    return jsonify(response)


@app.route('/update', methods=['POST'])
def update():
    obs = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == obs['observation_id'])
        p.true_class = obs['true_class']
        p.save()
        return jsonify(model_to_dict(p))
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(obs['observation_id'])
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
