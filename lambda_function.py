import os
import json
import io
import base64
import boto3
import pandas as pd
import numpy as np
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
BUCKET_NAME = 'testousama'
CSV_KEY = '1_DB_Basic_dec_24.csv'

# Initialize the dataframe as None; it will be loaded on the first invocation
db = None

def load_csv_from_s3():
    """
    Loads the CSV file from S3 and returns a pandas DataFrame.
    """
    global db
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=CSV_KEY)
        data = response['Body'].read().decode('utf-8')
        db = pd.read_csv(io.StringIO(data))
        print(f"Successfully loaded data from {BUCKET_NAME}/{CSV_KEY}")
    except Exception as e:
        print(f"Error loading CSV from S3: {e}")
        db = None

# Load the CSV data once when the Lambda container is initialized
load_csv_from_s3()

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

            # Begin plotting
            fig, ax = plt.subplots(figsize=(8, 6))

            # Plot data from CSV
            try:
                classes = db["MARBLE GROUP basic"].unique()
                colors = plt.get_cmap('tab10', len(classes))

                for idx, cls in enumerate(classes):
                    subset = db[db["MARBLE GROUP basic"] == cls]
                    ax.scatter(subset['d18O'], subset['d13C'], 
                               color=colors(idx), 
                               label=cls, 
                               alpha=0.6, 
                               edgecolors='w', 
                               linewidth=0.5)
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
            ax.scatter([i1], [i2], color='red', label='API Inputs', edgecolors='k', s=100)
            ax.annotate(f'i1 = {i1}', (i1, i1), textcoords="offset points", xytext=(0,10), ha='center', color='red')
            ax.annotate(f'i2 = {i2}', (i2, i2), textcoords="offset points", xytext=(0,10), ha='center', color='red')

            # Set labels and title
            ax.set_xlabel('i1')
            ax.set_ylabel('i2')
            ax.set_title('Scatterplot of CSV Data with API Inputs')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside the plot
            plt.tight_layout()

            # Save the plot to a BytesIO object
            img_io = io.BytesIO()
            fig.savefig(img_io, format='png', bbox_inches='tight')
            plt.close(fig)  # Close the figure to free memory
            img_io.seek(0)  # Rewind the buffer

            # Convert the image to a base64 string
            img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

            # Prepare the response body with the base64-encoded image
            response_body = {
                'image': img_base64,
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
    
    # Check if both 'i1' and 'i2' are present but 'getImage' is not
    if 'i1' in query_params and 'i2' in query_params:
        i1 = query_params['i1']
        i2 = query_params['i2']
        
        # Logic to classify based on i1 and i2
        try:
            i1_val = float(i1)
            i2_val = float(i2)
        except ValueError:
            response_body = {
                'message': '"i1" and "i2" must be valid numbers.'
            }
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps(response_body)
            }

        # Example classification logic
        group = "CARRARA" if i1_val < i2_val else "PANTELIKON"
        
        # Return the marble classification group
        response_body = {
            'message': f'MARBLE CLASSIFICATION {group}',
        }

        return {
            'statusCode': 200,
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
