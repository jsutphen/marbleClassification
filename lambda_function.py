import os
import json
from matplotlib.patches import Patch, Ellipse
import io
import base64
import boto3
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import transforms, ticker  # added ticker for custom tick spacing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import chi2
from scipy.spatial import distance

# Use a non-interactive backend for matplotlib
matplotlib.use('Agg')

# Set MPLCONFIGDIR to /tmp/matplotlib
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
os.makedirs('/tmp/matplotlib', exist_ok=True)

# Initialize the S3 client
s3 = boto3.client('s3')

# S3 bucket and CSV file details
BUCKET_NAME = 'marbleisotopes'
CSV_KEY = 'DB_MarbleSign_Final_NT_1812_29_MG_18Feb25.xlsx'

# Initialize the dataframe as None; it will be loaded on the first invocation
db = None

# Define custom styles for each marble group.
# Colors are given as hex codes and line patterns follow:
#   - "durchgängig"  → solid ("-")
#   - "gestrichelt"  → dashed ("--")
#   - "gepunktet"    → dotted (":")
#   - For special cases (e.g., Usak and Vakif) we use dash-dot or alternate patterns.
class_styles = {
    'Afyon':            {'color': '#FFA500', 's': 5, 'alpha': 0.5, 'ellipse_linewidth': 1, 'ellipse_linestyle': '-'},
    'Altintas':         {'color': '#FFA500', 's': 5, 'alpha': 0.5, 'ellipse_linewidth': 1, 'ellipse_linestyle': ':'},
    'Aphrodisias':      {'color': '#8B0000', 's': 5, 'alpha': 0.5, 'ellipse_linewidth': 1, 'ellipse_linestyle': '-'},
    'Carrara':          {'color': '#00008B', 's': 5, 'alpha': 0.5, 'ellipse_linewidth': 1, 'ellipse_linestyle': '-'},
    'Denizli':          {'color': '#FFFF00', 's': 5, 'alpha': 0.5, 'ellipse_linewidth': 1, 'ellipse_linestyle': '-'},
    'Doliana':          {'color': '#39FF14', 's': 5, 'alpha': 0.5, 'ellipse_linewidth': 1, 'ellipse_linestyle': '-'},
    'Ephesos1':         {'color': '#8A2BE2', 's': 5, 'alpha': 0.5, 'ellipse_linewidth': 1, 'ellipse_linestyle': '-'},
    'Ephesos2':         {'color': '#8A2BE2', 's': 5, 'alpha': 0.5, 'ellipse_linewidth': 1, 'ellipse_linestyle': '--'},
    'Goktepe':          {'color': '#FF6666', 's': 5, 'alpha': 0.5, 'ellipse_linewidth': 1, 'ellipse_linestyle': '-'},
    'Heracleia':        {'color': '#936B09', 's': 5, 'alpha': 0.5, 'ellipse_linewidth': 1, 'ellipse_linestyle': '-'},
    'Hierapolis':       {'color': '#BDB76B', 's': 5, 'alpha': 0.5, 'ellipse_linewidth': 1, 'ellipse_linestyle': '-'},
    'Hymettos':         {'color': '#008000', 's': 5, 'alpha': 0.5, 'ellipse_linewidth': 1, 'ellipse_linestyle': '--'},
    'Mani':             {'color': '#39FF14', 's': 5, 'alpha': 0.5, 'ellipse_linewidth': 1, 'ellipse_linestyle': '-'},
    'Milas':            {'color': '#0000FF', 's': 5, 'alpha': 0.5, 'ellipse_linewidth': 1, 'ellipse_linestyle': '--'},
    'Miletus':          {'color': '#0000FF', 's': 5, 'alpha': 0.5, 'ellipse_linewidth': 1, 'ellipse_linestyle': '-'},
    'Naxos/Apollona':   {'color': '#B22222', 's': 5, 'alpha': 0.5, 'ellipse_linewidth': 1, 'ellipse_linestyle': '-'},
    'Naxos/Mel/Kin':    {'color': '#B22222', 's': 5, 'alpha': 0.5, 'ellipse_linewidth': 1, 'ellipse_linestyle': '--'},
    'Paros/Lychnites':  {'color': '#800080', 's': 5, 'alpha': 0.5, 'ellipse_linewidth': 1, 'ellipse_linestyle': '-'},
    'Paros/Chorodaki':  {'color': '#800080', 's': 5, 'alpha': 0.5, 'ellipse_linewidth': 1, 'ellipse_linestyle': '--'},
    'Penteli':          {'color': '#008000', 's': 5, 'alpha': 0.5, 'ellipse_linewidth': 1, 'ellipse_linestyle': '-'},
    'Proconnesos1':     {'color': '#000000', 's': 5, 'alpha': 0.5, 'ellipse_linewidth': 1, 'ellipse_linestyle': '-'},
    'Proconnesos2':     {'color': '#000000', 's': 5, 'alpha': 0.5, 'ellipse_linewidth': 1, 'ellipse_linestyle': '--'},
    'StBeat':           {'color': '#90EE90', 's': 5, 'alpha': 0.5, 'ellipse_linewidth': 1, 'ellipse_linestyle': '-'},
    'Seravezza':        {'color': '#1E90FF', 's': 5, 'alpha': 0.5, 'ellipse_linewidth': 1, 'ellipse_linestyle': ':'},
    'Thasos/Al':        {'color': '#006400', 's': 5, 'alpha': 0.5, 'ellipse_linewidth': 1, 'ellipse_linestyle': '-'},
    'Thiounta':         {'color': '#BDB76B', 's': 5, 'alpha': 0.5, 'ellipse_linewidth': 1, 'ellipse_linestyle': '--'},
    'Tinos':            {'color': '#39FF14', 's': 5, 'alpha': 0.5, 'ellipse_linewidth': 1, 'ellipse_linestyle': '-'},
    'Usak':             {'color': '#BDB76B', 's': 5, 'alpha': 0.5, 'ellipse_linewidth': 1, 'ellipse_linestyle': '-.'},
    'Vakif':            {'color': '#BDB76B', 's': 5, 'alpha': 0.5, 'ellipse_linewidth': 1, 'ellipse_linestyle': ':'}
}

def absolute_probability(x, y, sample):
    """
    x: values of samples in x dimension
    y: values of samples in y dimension
    sample: two-dimensional data point for which the absolute probability will be computed

    returns: the absolute probability (1 at the mean, approaching 0 far away)
    """
    mean = (np.mean(x), np.mean(y))
    cov = np.cov(x, y)
    inv_cov = np.linalg.inv(cov)
    mahalanobis = distance.mahalanobis(sample, mean, inv_cov)
    survival_prob = 1 - chi2.cdf(mahalanobis**2, df=2)
    return survival_prob

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of x and y.
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      facecolor=facecolor,
                      **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def load_csv_from_s3():
    """
    Loads the Excel file from S3 and returns a pandas DataFrame.
    """
    global db
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=CSV_KEY)
        data = response['Body'].read()  # read as bytes
        db = pd.read_excel(io.BytesIO(data))
        # Drop rows missing key data (modify column names if needed)
        db = db.dropna(subset=['d18O', 'd13C', 'MARBLE GROUP basic'])
        print(f"Successfully loaded data from {BUCKET_NAME}/{CSV_KEY}")
    except Exception as e:
        print(f"Error loading CSV from S3: {e}")
        db = None


def handler(event, context):
    global db
    # Extract query string parameters from the event
    query_params = event.get('queryStringParameters', {}) or {}
    headers = {
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'OPTIONS,POST,GET',
        'Content-Type': 'application/json'
    }
    
    if 'getImage' in query_params:
        if 'i1' in query_params and 'i2' in query_params:
            try:
                i1 = float(query_params['i1'])
                i2 = float(query_params['i2'])
            except ValueError:
                return {
                    'statusCode': 400,
                    'headers': headers,
                    'body': json.dumps({'message': '"i1" and "i2" must be valid numbers.'})
                }
            items_str = query_params.get('items', '')
            items = items_str.split(',') if items_str else []

            # Load CSV if not already loaded
            if db is None:
                load_csv_from_s3()
                if db is None:
                    return {
                        'statusCode': 500,
                        'headers': headers,
                        'body': json.dumps({'message': 'Failed to load data from S3.'})
                    }

            # Filter data based on requested classes (if provided)
            if items:
                filtered_db = db[db["MARBLE GROUP basic"].isin(items)]
            else:
                filtered_db = db.copy()

            # Fit LDA on the selected data
            X = np.column_stack((filtered_db["d18O"], filtered_db["d13C"]))
            y = filtered_db["MARBLE GROUP basic"].values
            clf = LinearDiscriminantAnalysis()
            clf.fit(X, y)

            test_values = [[i1, i2]]
            predicted_class = clf.predict(test_values)[0]
            probabilities = clf.predict_proba(test_values)[0]

            # Start plotting
            fig, ax = plt.subplots(figsize=(10, 8))
            # Use the 90% confidence scaling factor (~sqrt(4.605))
            n_std = np.sqrt(4.605)

            # Loop over each class from the fitted LDA model.
            for cls in clf.classes_:
                # Get style settings from class_styles; if not defined, use a default style.
                style = class_styles.get(cls, {
                    'color': 'C0', 's': 10, 'alpha': 0.6, 'ellipse_linewidth': 2, 'ellipse_linestyle': '-'
                })
                subset = filtered_db[filtered_db["MARBLE GROUP basic"] == cls]
                x_vals = subset['d18O']
                y_vals = subset['d13C']
                # Scatter plot using custom style
                ax.scatter(x_vals, y_vals, color=style['color'], s=style['s'], alpha=style['alpha'], label=cls)
                # Confidence ellipse with custom edge properties
                confidence_ellipse(x_vals, y_vals, ax, n_std=n_std,
                                   edgecolor=style['color'],
                                   linewidth=style['ellipse_linewidth'],
                                   linestyle=style['ellipse_linestyle'],
                                   facecolor='none')

            # Plot the input test point in red
            ax.scatter([i1], [i2], color='red', edgecolors='k')
            ax.set_xlim([-12, 4])
            ax.set_ylim([-4, 6])
            
            # Seitenverhältnis so einstellen, dass 1 Einheit auf der x-Achse gleich 1 Einheit auf der y-Achse ist.
            ax.set_aspect('equal', adjustable='box')

            # Setze Ticks alle 4 Einheiten auf der x-Achse und alle 2 auf der y-Achse
            ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.set_xlabel(r'$\delta^{18}O$' "\n(PDB ‰)")
            ax.set_ylabel(r'$\delta^{13}C$' "\n(PDB ‰)")

            # Create legend labels that include predicted probabilities
            legend_labels = [
                f"{cls} ({prob * 100:.2f}%)"
                for cls, prob in zip(clf.classes_, probabilities)
            ]
            legend_handles = [
                Patch(facecolor=class_styles.get(cls, {'color': 'C0'})['color'],
                      edgecolor='ghostwhite',
                      label=label)
                for cls, label in zip(clf.classes_, legend_labels)
            ]
            ax.legend(handles=legend_handles, title="Relative Probability", loc="best", framealpha=0.6, facecolor="ghostwhite")
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.text(1.0, 0.0, 'MarbleSignatures', transform=ax.transAxes, fontsize=20, color='gray', alpha=0.5,
                    ha='right', va='bottom')


            # Compute the absolute probability for the predicted class.
            class_mask = (filtered_db["MARBLE GROUP basic"] == predicted_class)
            X_class = np.column_stack((
                filtered_db.loc[class_mask, "d18O"],
                filtered_db.loc[class_mask, "d13C"]
            ))
            abs_prob = absolute_probability(X_class[:, 0], X_class[:, 1], sample=[i1, i2])
            print("Absolute probability for the predicted class:", abs_prob)

            # Save the figure to a BytesIO stream
            img_io = io.BytesIO()
            fig.savefig(img_io, format='png', bbox_inches='tight')
            plt.close(fig)
            img_io.seek(0)
            img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

            response_body = {
                'image': img_base64,
                'classes': list(clf.classes_),
                'probabilities': probabilities.tolist(),
                'absolute': np.round(abs_prob * 100, 2)
            }
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps(response_body)
            }
        else:
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({'message': 'Both "i1" and "i2" parameters are required when requesting an image.'})
            }
    
    return {
        'statusCode': 400,
        'headers': headers,
        'body': json.dumps({'message': 'Missing required parameters. Provide "getImage" or "i1" and "i2" parameters.'})
    }