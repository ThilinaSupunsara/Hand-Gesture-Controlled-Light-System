from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

if __name__ == '__main__':
    # This block is only for testing on your local machine
    # Render will use gunicorn to run the app
    app.run(debug=True, host='0.0.0.0', port=5000)