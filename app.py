from flask import Flask, render_template, request, redirect, session, flash
import sqlite3
import joblib
import pandas as pd
import os
from werkzeug.security import generate_password_hash, check_password_hash
import json

app = Flask(__name__)
app.secret_key = 'your_secret_key'

model = joblib.load("model/calibrated_model.pkl")
label_maps = joblib.load("model/label_maps.pkl")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "heart_app.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def home():
    return redirect('/login')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        sex = request.form['sex']

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        existing_user = cursor.fetchone()

        if existing_user:
            flash("Username already exists!", "error")
            return redirect('/register')

        cursor.execute("INSERT INTO users (username, password, sex, role) VALUES (?, ?, ?, ?)",
                       (username, password, sex, 'user'))
        conn.commit()
        conn.close()
        flash("Registration successful. Please log in.", "success")
        return redirect('/login')

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        session.clear()  
        username = request.form['username']
        password = request.form['password']
    
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['role'] = user['role']
            session['username'] = user['username']
            session['sex'] = user['sex']
            return redirect('/admin' if user['role'] == 'admin' else '/predict')
        else:
            flash("Invalid credentials!", "error")
            return redirect('/login')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully.", "success")
    return redirect('/login')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user_id' not in session:
        return redirect('/login')

    if request.method == 'POST':
        form_data = {key: request.form[key] for key in request.form}
        df = pd.DataFrame([form_data])

        df["Sex"] = df["Sex"].map({"M": 1, "F": 0})
        df["ExerciseAngina"] = df["ExerciseAngina"].map({"Y": 1, "N": 0})
        for col, mapping in label_maps.items():
            df[col] = df[col].map(mapping)
        numeric_cols = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

        if df[numeric_cols].isnull().any().any():
            flash("Invalid input detected. Please check numeric fields.", "error")
            return render_template('form.html', data=form_data, submit_label='🧮 Predict Risk')

        risk = model.predict_proba(df)[0][1]
        risk_percent = round(risk * 100, 2)
        level = "Low" if risk_percent < 30 else "Moderate" if risk_percent < 70 else "High"
        age_value = int(df["Age"].iloc[0])
        input_data_json = json.dumps(form_data)

        conn = get_db()
        conn.execute("""
            INSERT INTO predictions (user_id, age, risk_percent, risk_level, input_data)
            VALUES (?, ?, ?, ?, ?)
        """, (session['user_id'], age_value, risk_percent, level, input_data_json))
        conn.commit()
        conn.close()

        message = f"Your risk level is {level}. Please consider medical advice if needed."
        return render_template('result.html', risk_percent=risk_percent, risk_level=level, message=message)

    return render_template('form.html', data={}, submit_label='🧮 Predict Risk')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect('/login')

    conn = get_db()
    raw_records = conn.execute("""
        SELECT id, user_id, age, risk_percent, risk_level, timestamp, input_data
        FROM predictions WHERE user_id = ?
    """, (session['user_id'],)).fetchall()
    conn.close()

    records = []
    for r in raw_records:
        record = dict(r)
        try:
            record['input_data'] = json.loads(record['input_data'])
        except Exception:
            record['input_data'] = {}
        records.append(record)

    return render_template('dashboard.html', predictions=records)

@app.route('/admin')
def admin_panel():
    if 'user_id' not in session or session.get('role') != 'admin':
        return "Access denied", 403

    username_filter = request.args.get('username', '').strip()
    filter_date = request.args.get('filter_date')
    risk_level = request.args.get('risk_level')

    conn = get_db()
    cursor = conn.cursor()

    query = """
        SELECT p.id, u.username, p.user_id, p.age, p.risk_percent, p.risk_level, p.timestamp, p.input_data
        FROM predictions p
        JOIN users u ON p.user_id = u.id
        WHERE 1=1
    """
    params = []

    if username_filter:
        query += " AND u.username LIKE ?"
        params.append(f"%{username_filter}%")
    if filter_date:
        query += " AND DATE(p.timestamp) = DATE(?)"
        params.append(filter_date)
    if risk_level:
        query += " AND p.risk_level = ?"
        params.append(risk_level)

    query += " ORDER BY p.timestamp DESC"

    raw_records = cursor.execute(query, tuple(params)).fetchall()
    conn.close()

    records = []
    for r in raw_records:
        record = dict(r)
        try:
            record['input_data'] = json.loads(record['input_data'])
        except Exception:
            record['input_data'] = {}
        records.append(record)

    return render_template('admin.html', records=records)



@app.route('/delete/<int:id>', methods=['POST'])
def delete(id):
    if 'user_id' not in session:
        return redirect('/login')
    conn = get_db()
    conn.execute("DELETE FROM predictions WHERE id = ? AND user_id = ?", (id, session['user_id']))
    conn.commit()
    conn.close()
    flash("Prediction deleted.", "success")
    return redirect('/dashboard')

@app.route('/edit/<int:id>', methods=['GET', 'POST'])
def edit_prediction(id):
    if 'user_id' not in session:
        return redirect('/login')

    conn = get_db()
    cursor = conn.cursor()
    record = cursor.execute("SELECT * FROM predictions WHERE id = ? AND user_id = ?", 
                            (id, session['user_id'])).fetchone()

    if not record:
        flash("Prediction not found or access denied.", "error")
        return redirect('/dashboard')

    if request.method == 'POST':
        form_data = {key: request.form[key] for key in request.form}
        df = pd.DataFrame([form_data])

        df["Sex"] = df["Sex"].map({"M": 1, "F": 0})
        df["ExerciseAngina"] = df["ExerciseAngina"].map({"Y": 1, "N": 0})
        for col, mapping in label_maps.items():
            df[col] = df[col].map(mapping)
        numeric_cols = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        if df[numeric_cols].isnull().any().any():
            flash("Invalid input in numeric fields.", "error")
            return render_template('form.html', data=form_data, submit_label='Update')

        risk = model.predict_proba(df)[0][1]
        risk_percent = round(risk * 100, 2)
        level = "Low" if risk_percent < 30 else "Moderate" if risk_percent < 70 else "High"
        input_data_json = json.dumps(form_data)

        cursor.execute("""
            UPDATE predictions
            SET age = ?, risk_percent = ?, risk_level = ?, input_data = ?
            WHERE id = ? AND user_id = ?
        """, (int(df["Age"].iloc[0]), risk_percent, level, input_data_json, id, session['user_id']))
        conn.commit()
        conn.close()

        flash("Prediction updated successfully.", "success")
        return redirect('/dashboard')

    input_data = json.loads(record['input_data'])
    return render_template('form.html', data=input_data, submit_label='Update')


@app.route('/admin/delete/<int:id>', methods=['POST'])
def admin_delete(id):
    if 'user_id' not in session or session.get('role') != 'admin':
        return "Access denied", 403

    conn = get_db()
    conn.execute("DELETE FROM predictions WHERE id = ?", (id,))
    conn.commit()
    conn.close()
    flash("Prediction deleted by admin.", "success")
    return redirect('/admin')


@app.route('/admin/edit/<int:id>', methods=['GET', 'POST'])
def admin_edit(id):
    if 'user_id' not in session or session.get('role') != 'admin':
        return "Access denied", 403

    conn = get_db()
    cursor = conn.cursor()
    record = cursor.execute("SELECT * FROM predictions WHERE id = ?", (id,)).fetchone()

    if not record:
        flash("Prediction not found.", "error")
        return redirect('/admin')

    if request.method == 'POST':
        form_data = {key: request.form[key] for key in request.form}
        df = pd.DataFrame([form_data])

        df["Sex"] = df["Sex"].map({"M": 1, "F": 0})
        df["ExerciseAngina"] = df["ExerciseAngina"].map({"Y": 1, "N": 0})
        for col, mapping in label_maps.items():
            df[col] = df[col].map(mapping)
        numeric_cols = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        if df[numeric_cols].isnull().any().any():
            flash("Invalid input in numeric fields.", "error")
            return render_template('form.html', data=form_data, submit_label='Update')

        risk = model.predict_proba(df)[0][1]
        risk_percent = round(risk * 100, 2)
        level = "Low" if risk_percent < 30 else "Moderate" if risk_percent < 70 else "High"
        input_data_json = json.dumps(form_data)

        cursor.execute("""
            UPDATE predictions
            SET age = ?, risk_percent = ?, risk_level = ?, input_data = ?
            WHERE id = ?
        """, (int(df["Age"].iloc[0]), risk_percent, level, input_data_json, id))
        conn.commit()
        conn.close()

        flash("Prediction updated by admin.", "success")
        return redirect('/admin')

    input_data = json.loads(record['input_data'])
    return render_template('form.html', data=input_data, submit_label='Update')


@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect('/login')

    conn = get_db()
    user_id = session['user_id']
    user_data = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()

    predictions = conn.execute("""
        SELECT risk_level, risk_percent, timestamp FROM predictions
        WHERE user_id = ?
    """, (user_id,)).fetchall()
    conn.close()


    total = len(predictions)
    avg_risk = round(sum([p['risk_percent'] for p in predictions])/total, 2) if total > 0 else 0
    last_time = predictions[0]['timestamp'] if total > 0 else "N/A"
    levels = {'Low': 0, 'Moderate': 0, 'High': 0}
    for p in predictions:
        levels[p['risk_level']] += 1

    return render_template("profile.html",
                           user=user_data,
                           total=total,
                           avg_risk=avg_risk,
                           last_time=last_time,
                           levels=levels)


if __name__ == '__main__':
    app.run(debug=True)
