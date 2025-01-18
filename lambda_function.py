import os
import json
import io
import base64
import boto3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Set MPLCONFIGDIR to /tmp/matplotlib
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

# Create the directory if it doesn't exist
os.makedirs('/tmp/matplotlib', exist_ok=True)

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


# Initialize the S3 client
s3 = boto3.client('s3')

# S3 bucket and CSV file details
BUCKET_NAME = 'marbleisotopes'
CSV_KEY = 'dataset.csv'
# Key for saving the top 3 classes
TOP_CLASSES_KEY = 'top_classes.json'

# Initialize the dataframe as None; it will be loaded on the first invocation
db = None

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The Axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    facecolor : str
        The face color of the ellipse.
    **kwargs
        Additional keyword arguments passed to Ellipse.

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    # Eigenvalues and eigenvectors for the covariance matrix
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Scale the ellipse to the desired number of standard deviations
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # Calculate the transformation
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def load_csv_from_s3():
    """
    Loads the CSV file from S3 and returns a pandas DataFrame.
    """
    global db
    global items
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=CSV_KEY)
        data = response['Body'].read().decode('utf-8')
        db = pd.read_csv(io.StringIO(data))
        db = db.dropna(subset=['d18O', 'd13C', 'MARBLE GROUP basic'])
        print(f"Successfully loaded data from {BUCKET_NAME}/{CSV_KEY}")
    except Exception as e:
        print(f"Error loading CSV from S3: {e}")
        db = None


def handler(event, context):
    global db
    # Extract query string parameters from the event
    query_params = event.get('queryStringParameters', {}) or {}

    # Common headers for CORS and content type
    headers = {
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'OPTIONS,POST,GET',
        'Content-Type': 'application/json'
    }

    # Check if 'getImage' is in the query parameters
    if 'getImage' in query_params:
        # Check if both 'i1' and 'i2' are present in the query parameters
        if 'i1' in query_params and 'i2' in query_params:
            try:
                i1 = float(query_params['i1'])  # Convert i1 to float for plotting
                i2 = float(query_params['i2'])  # Convert i2 to float for plotting
            except ValueError:
                response_body = {
                    'message': '"i1" and "i2" must be valid numbers.'
                }
                return {
                    'statusCode': 400,
                    'headers': headers,
                    'body': json.dumps(response_body)
                }
            items_str = query_params.get('items', '')
            items = items_str.split(',') if items_str else []
            # Reload the CSV if not already loaded
            if db is None:
                load_csv_from_s3()
                if db is None:
                    response_body = {
                        'message': 'Failed to load data from S3.'
                    }
                    return {
                        'statusCode': 500,
                        'headers': headers,
                        'body': json.dumps(response_body)
                    }
            if items:
                filtered_db = db[db["MARBLE GROUP basic"].isin(items)]
            else:
                filtered_db = db.copy()  # If no items specified, use the entire dataset
            # Begin plotting
            fig, ax = plt.subplots(figsize=(8, 6))

            # Plot data from CSV
            try:
                test_values = [[i1, i2]]
                classes = filtered_db["MARBLE GROUP basic"].unique()
                X = np.column_stack((filtered_db["d18O"], filtered_db["d13C"]))
                y = filtered_db["MARBLE GROUP basic"].values
                clf = LinearDiscriminantAnalysis()
                clf.fit(X, y)
                predicted_class = clf.predict(test_values)[0]
                probabilities = clf.predict_proba(test_values)[0]
                # Identify the top three probabilities
                top_n = 3
                top_indices = np.argsort(probabilities)[::-1][:top_n]
                top_classes = clf.classes_[top_indices]
                top_probabilities = probabilities[top_indices]

                db_top3 = filtered_db[filtered_db["MARBLE GROUP basic"].isin(top_classes)]

                # 7. Start Plotting
                fig, ax = plt.subplots(figsize=(10, 8))

                # Assign colors to top three classes using 'tab10' colormap
                colors = plt.get_cmap('tab10', len(top_classes))

                # Scaling factor for the confidence ellipse (e.g., 95% confidence interval)
                # For a 95% confidence interval in 2D, n_std ≈ sqrt(5.991)
                n_std = np.sqrt(5.991)  # Adjust based on desired confidence level


                for idx, cls in enumerate(top_classes):
                    subset = db_top3[db_top3["MARBLE GROUP basic"] == cls]
                    x = subset['d18O']
                    y = subset['d13C']

                    # Scatter plot for the current class
                    ax.scatter(x, y, alpha=0.6, label=cls, color=colors(idx))

                    # Add confidence ellipse for the current class
                    confidence_ellipse(x, y, ax, n_std=n_std, edgecolor=colors(idx), linewidth=2)

            except KeyError as e:
                response_body = {
                    'message': f'Missing expected column in CSV: {e}'
                }
                return {
                    'statusCode': 500,
                    'headers': headers,
                    'body': json.dumps(response_body)
                }

            # Plot the API inputs i1 and i2
            ax.scatter([i1], [i2], color='red', edgecolors='k')

            # Set labels and title
            ax.set_xlabel('i1')
            ax.set_ylabel('i2')
            ax.legend(title='MARBLE GROUP', fontsize=10, title_fontsize=12)
            prob_text = "\n".join([f"{cls}: {prob * 100:.2f}%" for cls, prob in zip(top_classes, top_probabilities)])
            annotation_text = f"Predicted Class: {predicted_class}\nTop 3 Probabilities:\n{prob_text}"
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax.text(0.05, 0.95, annotation_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)

            # Enhance grid for better readability
            ax.grid(True, linestyle='--', alpha=0.5)
            # Watermark
            ax.text(0.5, 0.5, 'Team 41', transform=ax.transAxes,
                    fontsize=40, color='gray', alpha=0.5,
                    ha='center', va='center')
            # Save the plot to a BytesIO object
            img_io = io.BytesIO()
            fig.savefig(img_io, format='png', bbox_inches='tight')
            plt.close(fig)  # Close the figure to free memory
            img_io.seek(0)  # Rewind the buffer

            # Convert the image to a base64 string
            img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

            # Prepare the response body with the base64-encoded image, top classes and their probabilities
            response_body = {
                'image': img_base64,
                'classes': top_classes.tolist(), # Use tolist() to make it more compatible with json
                'probabilities': top_probabilities.tolist() # Used in frontend to calculate if top probability is >= 60%
            }

            # Return the response with proxy integration support
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps(response_body)
            }
        else:
            # If i1 or i2 is missing
            response_body = {
                'message': 'Both "i1" and "i2" parameters are required when requesting an image.'
            }
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps(response_body)
            }
    
    # If neither 'getImage' nor 'i1' and 'i2' are present, return an error message
    #---------------- Maybe needs a new guard? only if i1 and i2 are not present? ----------------#
    response_body = {
        'message': 'Missing required parameters. Provide "getImage" or "i1" and "i2" parameters.'
    }
    return {
        'statusCode': 400,
        'headers': headers,
        'body': json.dumps(response_body)
    }