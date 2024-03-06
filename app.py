from flask import Flask , render_template,request,redirect
import pandas as pd
import pickle
app = Flask(__name__)

model2=pickle.load(open("trained_model.pkl","rb"))
bike=pd.read_csv("Used_Bikes.csv")
@app.route("/")
def index():
    brand = sorted(bike["brand"].unique())
    bike_name = sorted(bike["bike_name"].unique())
    owner = sorted(bike["owner"].unique())
    return render_template('index.html',brand=brand,bike_name=bike_name,owner=owner)

@app.route("/predict",methods=['POST'])
def predict():
    brand=request.form.get('brand')
    bike_name=request.form.get('bike_name')
    kms_driven=int(request.form.get('kms_driven'))
    owner=request.form.get('owner')
    age=int(request.form.get('age'))
    power=int(request.form.get('power'))
    print(brand,bike_name,kms_driven,owner,age,power)
    prediction=model2.predict(pd.DataFrame([[brand,bike_name,kms_driven,owner,age,power]] ,
    columns=["brand","bike_name","kms_driven","owner","age","power"] ) )
    return str(prediction[0])

if __name__=="__main__":
    app.run(debug=True)