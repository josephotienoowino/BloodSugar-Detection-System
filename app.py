#===============================================imports===================================
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
import csv
import io
import matplotlib.pyplot as plt
import base64
from io import StringIO
import pickle
from sklearn.linear_model import LogisticRegression
from flask_login import login_required, current_user, LoginManager, UserMixin, login_user, logout_user
from datetime import datetime, timezone
from flask import make_response
from reportlab.pdfgen import canvas
from flask import jsonify


clf = LogisticRegression()

#===================================================initialisation of the app==============================
app = Flask(__name__, template_folder='/home/jose/Desktop/Diabetes System/templates')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'mysecretkey'
model = pickle.load(open("model.pkl", "rb"))
db = SQLAlchemy(app)

# initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)

#=================================================database===============================================
now = datetime.utcnow()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class PredictionResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow().replace(tzinfo=timezone.utc))
    pregnancies = db.Column(db.Integer, nullable=False)
    glucose = db.Column(db.Integer, nullable=False)
    blood_pressure = db.Column(db.Integer, nullable=False)
    skin_thickness = db.Column(db.Integer, nullable=False)
    insulin = db.Column(db.Integer, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    dpf = db.Column(db.Float, nullable=False)
    age = db.Column(db.Integer, nullable=False)
    result = db.Column(db.String(50), nullable=False)

# ----------------=============================route for the app=====================
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/about')
def about():
    return render_template('about.html')

# ===============================================signup===========================session
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email address already exists')
            return redirect(url_for('signup'))
        else:
            new_user = User(username=username, email=email,
                            password=generate_password_hash(password))
            db.session.add(new_user)
            db.session.commit()
            flash('Account created successfully')
            return redirect(url_for('login'))
    return render_template('signup.html')
# -==================================================login session-------=========================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password')
            return redirect(url_for('login'))
    return render_template('login.html')
# ==================================================logout session======================================
@app.route('/logout')
def logout():
    session.clear()
    flash('You have logged out')
    return redirect(url_for('login'))

# ----------------------------------Dashboard-+++++++++++++---------------------------------#
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    user_id = session.get('user_id')
    if user_id:
        user = User.query.get(user_id)
        prediction_text = ''
        if request.method == 'POST':
            text1 = request.form['1']
            text2 = request.form['2']
            text3 = request.form['3']
            text4 = request.form['4']
            text5 = request.form['5']
            text6 = request.form['6']
            text7 = request.form['7']
            text8 = request.form['8']

            row_df = pd.DataFrame([pd.Series([text1,text2,text3, text4, text5, text6,text7,text8])])

            prediction = model.predict_proba(row_df)
            output = '{0:.{1}f}'.format(prediction[0][1], 2)
            output = float(output) * 100

            if output > 40.0:
                prediction_text = f'You have a positive diabetes detection.\nProbability of having Diabetes is {output}%'
                result = 'positive'
            else:
                prediction_text = f'You are safe.\n Probability of having diabetes is {output}%'
                result = 'negative'
            
            # create a new instance of PredictionResult and add it to the database
            new_result = PredictionResult(user_id=user.id, pregnancies=text1, glucose=text2, blood_pressure=text3, 
                                           skin_thickness=text4, insulin=text5, bmi=text6, dpf=text7, age=text8, 
                                           result=result)
            db.session.add(new_result)
            db.session.commit()

        return render_template('dashboard.html', user=user, prediction=prediction_text)
    else:
        return redirect(url_for('login'))

#+++++++++++++++++++++++++++++++++++++++results+++++++++++++++++++++++++++++++++
@app.route('/generate-report')
def generate_report():
    # query the database to get all the prediction results for the current user
    user_id = session.get('user_id')
    results = PredictionResult.query.filter_by(user_id=user_id).all()

    # create a CSV file from the results data
    csv_data = io.StringIO()
    csv_writer = csv.writer(csv_data)
    csv_writer.writerow(['Timestamp', 'Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI', 'DPF', 'Age', 'Result'])
    for result in results:
        csv_writer.writerow([result.timestamp, result.pregnancies, result.glucose, result.blood_pressure, result.skin_thickness, result.insulin, result.bmi, result.dpf, result.age, result.result])


    # generate a response with the CSV file as an attachment
    response = make_response(csv_data.getvalue())
    response.headers['Content-Disposition'] = 'attachment; filename=prediction-results.csv'
    response.headers['Content-Type'] = 'text/csv'

    return response 
#-------------------------------PDF Report---------------------------------
    # # generate a PDF file from the results data
    # pdf_data = io.BytesIO()
    # pdf_canvas = canvas.Canvas(pdf_data)
    # y = 700
    # for result in results:
    #     pdf_canvas.drawString(50, y, f"Timestamp: {result.timestamp}")
    #     pdf_canvas.drawString(50, y - 20, f"Pregnancies: {result.pregnancies}")
    #     pdf_canvas.drawString(50, y - 40, f"Glucose: {result.glucose}")
    #     pdf_canvas.drawString(50, y - 60, f"Blood Pressure: {result.blood_pressure}")
    #     pdf_canvas.drawString(50, y - 80, f"Skin Thickness: {result.skin_thickness}")
    #     pdf_canvas.drawString(50, y - 100, f"Insulin: {result.insulin}")
    #     pdf_canvas.drawString(50, y - 120, f"BMI: {result.bmi}")
    #     pdf_canvas.drawString(50, y - 140, f"DPF: {result.dpf}")
    #     pdf_canvas.drawString(50, y - 160, f"Age: {result.age}")
    #     pdf_canvas.drawString(50, y - 180, f"Result: {result.result}")
    #     y -= 200
    # pdf_canvas.showPage()
    # pdf_canvas.save()

    # # generate a response with the PDF file as an attachment
    # pdf_response = make_response(pdf_data.getvalue())
    # pdf_response.headers['Content-Disposition'] = 'attachment; filename=prediction-results.pdf'
    # pdf_response.headers['Content-Type'] = 'application/pdf'

    # return pdf_response

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
