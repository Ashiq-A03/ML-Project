import pandas as pd
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# 1. Load your CSV data
df = pd.read_csv('student_data.csv')

# 2. Encode categorical columns to match HTML form field names
df['participation'] = LabelEncoder().fit_transform(df['participation'])
df['parental_involvement'] = LabelEncoder().fit_transform(df['parental_involvement'])

# X matches the HTML field ordering (except for gender/student_id which are unused in original logic)
input_cols = [
    'attendance', 'avg_assignment_score', 'participation', 'study_hours',
    'assignments_submitted', 'quiz_score', 'project_score', 'previous_gpa',
    'extra_activities', 'parental_involvement'
]
X = df[input_cols]
y = df['performance']  # Your label column must be named 'performance' (or adjust here)

# 3. Fit your model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

participation_encoder = LabelEncoder().fit(['Low', 'Medium', 'High'])
parental_encoder = LabelEncoder().fit(['Low', 'Medium', 'High'])

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            data = request.form
            input_data = [
                int(data.get('attendance')),
                int(data.get('avg_assignment_score')),
                participation_encoder.transform([data.get('participation')])[0],
                int(data.get('study_hours')),
                int(data.get('assignments_submitted')),
                int(data.get('quiz_score')),
                int(data.get('project_score')),
                float(data.get('previous_gpa')),
                int(data.get('extra_activities')),
                parental_encoder.transform([data.get('parental_involvement')])[0]
            ]
            pred = model.predict([input_data])
            prediction = pred[0]
        except Exception as e:
            prediction = f"Error: {str(e)}"
    return render_template("home.html", prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
