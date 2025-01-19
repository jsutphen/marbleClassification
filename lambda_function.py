import os
import json
from matplotlib.patches import Patch
import io
import base64
import boto3
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import chi2
from scipy.spatial import distance

# Use a non-interactive backend for matplotlib
matplotlib.use('Agg')

# Set MPLCONFIGDIR to /tmp/matplotlib
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

# Create the directory if it doesn't exist
os.makedirs('/tmp/matplotlib', exist_ok=True)

# Initialize the S3 client
s3 = boto3.client('s3')

# S3 bucket and CSV file details
BUCKET_NAME = 'marbleisotopes'
CSV_KEY = 'dataset.csv'

# Initialize the dataframe as None; it will be loaded on the first invocation
db = None

def absolute_probability(x, y, sample):
    '''
    x: values of samples in x dimension
    y: values of samples in y dimension
    sample: two-dimensional data point, for which the abs. prob. will be computed

    returns: the absolute probability, ranging from 1 exactly in the mean to close to 0 far away from the data
    '''
    mean = (np.mean(x), np.mean(y))
    cov = np.cov(x, y)
    inv_cov = np.linalg.inv(cov)
    mahalanobis = distance.mahalanobis(sample, mean, inv_cov)
    survival_prob = 1 - chi2.cdf(mahalanobis**2, df=2)
    return survival_prob

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    # Eigenvalues and eigenvectors for the covariance matrix
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      facecolor=facecolor,
                      **kwargs)

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
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=CSV_KEY)
        data = response['Body'].read().decode('utf-8')
        db = pd.read_csv(io.StringIO(data))
        db = db.dropna(subset=['d18O', 'd13C', 'MARBLE GROUP basic'])
    except Exception as e:
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

            # Filter only the requested classes if specified, else use all data
            if items:
                filtered_db = db[db["MARBLE GROUP basic"].isin(items)]
            else:
                filtered_db = db.copy()

            # Fit LDA to these classes
            X = np.column_stack((filtered_db["d18O"], filtered_db["d13C"]))
            y = filtered_db["MARBLE GROUP basic"].values
            clf = LinearDiscriminantAnalysis()
            clf.fit(X, y)

            # Perform prediction
            test_values = [[i1, i2]]
            predicted_class = clf.predict(test_values)[0]
            probabilities = clf.predict_proba(test_values)[0]

            # Prepare to plot
            fig, ax = plt.subplots(figsize=(10, 8))
            colors = plt.get_cmap('tab10', len(clf.classes_))

            # Scaling factor for the confidence ellipse
            # For ~90% CI in 2D, n_std ~ sqrt(4.605)
            # Adjust as needed
            n_std = np.sqrt(4.605)

            # Plot each class in the filtered set
            for idx, cls in enumerate(clf.classes_):
                subset = filtered_db[filtered_db["MARBLE GROUP basic"] == cls]
                x = subset['d18O']
                y = subset['d13C']

                # Scatter
                ax.scatter(x, y, alpha=0.6, label=cls, color=colors(idx))

                # Confidence ellipse
                confidence_ellipse(x, y, ax, n_std=n_std,
                                   edgecolor=colors(idx), linewidth=2)

            # Plot the input point
            ax.scatter([i1], [i2], color='red', edgecolors='k')
            ax.set_xlim([-12, 2])
            ax.set_ylim([-4, 6])
            ax.axhline(y=0, color='black', linewidth=1, linestyle='solid')  # Horizontal line at y=0
            ax.axvline(x=0, color='black', linewidth=1, linestyle='solid')  # Vertical line at x=0
            legend_labels = [f"{cls} ({prob * 100:.2f}%)" for cls, prob in zip(clf.classes_, probabilities)]

            # Create legend handles
            legend_handles = [Patch(facecolor=colors(idx), edgecolor='ghostwhite', label=label) for idx, label in enumerate(legend_labels)]

            # Add the legend to the plot
            ax.legend(handles=legend_handles, title="Relative Probability", loc="best", framealpha=0.6, facecolor="ghostwhite")

            # Enhance grid for better readability
            ax.grid(True, linestyle='--', alpha=0.5)

            # Watermark
            ax.text(0.5, 0.5, 'Team 41', transform=ax.transAxes,
                    fontsize=40, color='gray', alpha=0.5,
                    ha='center', va='center')

            # -------------------------------------------------------------------- #
            # Compute absolute probability for the predicted class
            # -------------------------------------------------------------------- #
            class_mask = (filtered_db["MARBLE GROUP basic"] == predicted_class)
            X_class = np.column_stack((
                filtered_db.loc[class_mask, "d18O"],
                filtered_db.loc[class_mask, "d13C"]
            ))
            sample_point = [i1, i2]
            abs_prob = absolute_probability(X_class[:, 0], X_class[:, 1], sample=sample_point)
            # -------------------------------------------------------------------- #

            # Save the plot to a BytesIO object
            img_io = io.BytesIO()
            fig.savefig(img_io, format='png', bbox_inches='tight')
            plt.close(fig)
            img_io.seek(0)

            # Convert the image to a base64 string
            img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

            # Prepare response
            response_body = {
                'image': img_base64,
                'classes': list(clf.classes_),
                'probabilities': probabilities.tolist(),
                'absolute': np.round(abs_prob, 4)
            }

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
    response_body = {
        'message': 'Missing required parameters. Provide "getImage" or "i1" and "i2" parameters.'
    }
    return {
        'statusCode': 400,
        'headers': headers,
        'body': json.dumps(response_body)
    }