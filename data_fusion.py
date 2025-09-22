import pandas as pd

def fuse_data():
    """
    Reads data from multiple CSV files and merges them into a single DataFrame.
    """
    try:
        # Read the individual data sources
        attendance_df = pd.read_csv('attendance.csv')
        assessments_df = pd.read_csv('assessments.csv')
        fees_df = pd.read_csv('fees.csv')

        print("Successfully read individual data sources.")

        # Merge the DataFrames based on StudentID
        # Start with attendance and merge assessments
        merged_df = pd.merge(attendance_df, assessments_df, on='StudentID', how='left')
        
        # Merge the fees data into the combined DataFrame
        merged_df = pd.merge(merged_df, fees_df, on='StudentID', how='left')
        
        print("Data fusion successful. Merged DataFrame:")
        print(merged_df)
        
        return merged_df

    except FileNotFoundError as e:
        print(f"Error: One of the data files was not found. Please ensure all three CSVs are in the same directory.")
        print(e)
        return None

if __name__ == '__main__':
    fused_data = fuse_data()
    if fused_data is not None:
        # We can now use this fused_data DataFrame for prediction
        # (This is a placeholder, we'll integrate this into the dashboard next)
        fused_data.to_csv('fused_student_data.csv', index=False)
        print("\nFused data saved to 'fused_student_data.csv'.")