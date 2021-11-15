from flask import Flask, render_template,request
import numpy as np
import pandas as pd
from joblib import load
import plotly.express as px
import plotly.graph_objects as go
import uuid



app = Flask(__name__)

# @app.route("/") # GET method by default
@app.route("/",methods = ['GET','POST'])
def hello_world():
    request_type_str = request.method
    if request_type_str == 'GET':
        # href_i = "static/base_pic.svg"
        return render_template('index.html',href_i = "static/base_pic.svg")
    else:
        # print(request.form['text'],"-------------") # request.form is dictionary with 'text' key
        text = request.form['text']
        # href_i = "static/base_pic.svg"
        # path = "static/smile.png"
        random_string = uuid.uuid4().hex
        model = load('linearmodel.joblib')
        out_path = 'static/'+random_string+'out_pic.svg'
        make_picture("AgesAndHeights.pkl", model, text,out_path)
        # out_path = 'static/out_pic.svg'
        return render_template('index.html',href_i = out_path)


## ------------------------------------
# make picture 
def make_picture(training_data_file_name, model,input_text,output_file):
    data = pd.read_pickle(training_data_file_name)
    data = data[data.Age>0]
#     lm = LinearRegression()
#     lm.fit(data['Age'].to_numpy().reshape(-1,1), data['Height'])
    x_new = np.arange(19).reshape(-1,1)
    preds = model.predict(x_new)

    fig = px.scatter(x=data['Age'],y=data['Height'],title = "Height vs Age of people",
                     labels={'x':'Age (years)','y':'Height (inches)'}) 
    
    fig.add_trace(go.Scatter(x = x_new.reshape(19),y= preds,mode='lines',name='model'))
#     fig.write_image('base_pic.svg',width = 800) # for creation of base image
    
    new_inp_np_arr = floats_string_to_np_arr(input_text)
    new_preds = model.predict(new_inp_np_arr)
    fig.add_trace(go.Scatter(x=new_inp_np_arr.reshape(len(new_inp_np_arr)),y=new_preds,
                                                     name = "new Outputs",mode = 'markers',
                                                     marker = dict(color = 'purple',
                                                                  size = 20),
                                                     line = dict(color = 'purple',
                                                                width = 2)
                                                     ))
    fig.write_image( output_file,width = 800)
    # fig.show()


## ------------------------------------
# is float function


def is_float(s):
    try:
        float(s)
        return True
    except:
        return False
    
def floats_string_to_np_arr(float_str):
    floats  = np.array([float(x) for x in float_str.split(',') if is_float(x)])
    print(floats)
    
    return floats.reshape(len(floats),1)