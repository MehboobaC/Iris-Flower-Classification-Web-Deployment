from flask import Flask,render_template,request
import pickle
import numpy as np
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    sl=float(request.values['SL'])
    sw=float(request.values['SW'])
    pl=float(request.values['PL'])
    pw=float(request.values['PW'])
    inp=[sl,sw,pl,pw]
    inp=np.reshape(inp,(1,-1))
    output=model.predict(inp)
    output=output.item()
    return render_template('result.html',prediction_text="Your flower is {}".format(output))
if __name__ == "__main__":
    app.run()