import pandas as pd
import numpy as np

# Set a seed for reproducibility
np.random.seed(42)

# Generate 5000 student records for a larger dataset
num_students = 5000

# Generate features
student_ids = [f'2326-{str(i+1).zfill(4)}' for i in range(num_students)]
attendance = np.random.uniform(low=0, high=100, size=num_students)
marks = np.random.uniform(low=0, high=100, size=num_students)
attempts = np.random.randint(1, 25, size=num_students)
fees_due = np.random.uniform(low=0, high=150000, size=num_students)

# Create a more structured relationship with dropout status
# Students with multiple 'bad' metrics will have a high probability of dropping out
dropout_prob = np.zeros(num_students)

# Strong risk factors: low attendance, very low marks, high attempts, high fees
is_low_attendance = attendance < 65
is_low_marks = marks < 24
is_high_attempts = attempts > 15
is_high_fees = fees_due > 80000

# High-risk students: Meet multiple strong risk factors
high_risk_condition = (is_low_attendance) | (is_low_marks & is_high_attempts) | (is_high_fees)
dropout_prob[high_risk_condition] = 0.9 + np.random.rand(np.sum(high_risk_condition)) * 0.1

# Medium-risk students: Meet one or two medium risk factors
is_med_attendance = (attendance >= 65) & (attendance <= 75)
is_med_marks = (marks >= 24) & (marks <= 60)
is_med_attempts = (attempts >= 7) & (attempts <= 15)
is_med_fees = (fees_due >= 50000) & (fees_due <= 80000)

medium_risk_condition = (is_med_attendance) | (is_med_marks & is_med_attempts) | (is_med_fees)
# Ensure medium risk students aren't also in the high-risk group
medium_risk_condition = medium_risk_condition & ~high_risk_condition
dropout_prob[medium_risk_condition] = 0.4 + np.random.rand(np.sum(medium_risk_condition)) * 0.2

# Low-risk students: All other students
low_risk_condition = ~(high_risk_condition | medium_risk_condition)
dropout_prob[low_risk_condition] = 0.05 + np.random.rand(np.sum(low_risk_condition)) * 0.15

# Introduce noise to the probabilities to make it less deterministic
dropout_prob = np.clip(dropout_prob + np.random.normal(0, 0.05, num_students), 0, 1)

did_dropout = (np.random.rand(num_students) < dropout_prob).astype(int)

# Create the DataFrame
data = pd.DataFrame({
    'StudentID': student_ids,
    'attendance': attendance,
    'marks': marks,
    'attempts': attempts,
    'fees_due': fees_due,
    'did_dropout': did_dropout
})

# Save to CSV
data.to_csv('student_data_enhanced.csv', index=False)

print("Enhanced student_data_enhanced.csv file created successfully.")
print(data.head())
print(f"Number of dropout cases: {data['did_dropout'].sum()}")