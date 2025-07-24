from flask import Flask, request, render_template, send_file, session, redirect, url_for, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import pandas as pd
from query_engine import handle_query, build_system_prompt, sanitize_prompt
from rag_engine import setup_rag_chain
from datetime import datetime, timedelta
import os
import random
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

load_dotenv()

app = Flask(__name__)

limiter = Limiter(key_func=get_remote_address, default_limits=["100/hour"])
limiter.init_app(app)

app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret")
app.permanent_session_lifetime = timedelta(minutes=30)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
UPLOAD_FOLDER = "uploaded_data"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # ensure the folder exists
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

qa_chain = setup_rag_chain()

def load_data(file_path="data/global_superstore.csv"):
    df = pd.read_csv(file_path)

    # Convert 'Order Date' to datetime only if it exists
    if "Order Date" in df.columns:
        df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    
    return pd.read_csv(file_path, encoding="ISO-8859-1", low_memory=False)


# Load your default dataset
global_superstore_df = pd.read_csv("data/global_superstore.csv", parse_dates=["Order Date"])

@app.route("/", methods=["GET", "POST"])
def index():
    # Initialize dropdown with just the default dataset
    return render_template("index.html", datasets=["Global Superstore"])

@app.route("/upload", methods=["POST"])
def upload():
    try:
        file = request.files["file"]
        dataset_name = request.form["dataset_name"]

        if not file or not dataset_name:
            return jsonify({"success": False, "error": "Missing file or dataset name"}), 400

        if "uploaded_datasets" not in session:
            session["uploaded_datasets"] = {}

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        session["uploaded_datasets"][dataset_name] = filepath

        # 🟢 Build the list of datasets to return
        datasets = ["global_superstore"] + list(session["uploaded_datasets"].keys())

        return jsonify({
            "success": True,
            "message": f"Dataset '{dataset_name}' uploaded successfully!",
            "datasets": datasets,
            "default_dataset": dataset_name
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/get_datasets", methods=["GET"])
def get_datasets():
    datasets = ["Global Superstore"]
    if "uploaded_datasets" in session:
        datasets.extend(session["uploaded_datasets"].keys())
    return jsonify({"datasets": datasets})

@app.route("/ask", methods=["POST"])
def ask():
    try:
        user_question = request.form["question"]
        dataset_name = request.form["dataset_name"]
        print("dataset_name: ",dataset_name)
        # Choose correct dataset
        if dataset_name == "Global Superstore":
            df = global_superstore_df
        elif "uploaded_datasets" in session and dataset_name in session["uploaded_datasets"]:
            filepath = session["uploaded_datasets"][dataset_name]
            df = pd.read_csv(filepath)
        else:
            return jsonify({"answer": "Dataset not found."})

        # For demo: respond with first 3 rows (replace this with real logic)
        preview = df.head(3).to_dict(orient="records")
        return jsonify({"answer": f"First few rows of {dataset_name}:", "preview": preview})

    except Exception as e:
        return jsonify({"answer": f"An error occurred: {str(e)}"})
    
@app.route("/clear", methods=["POST"])
def clear():
    session.pop("chat_history", None)
    session.pop("chart_filename", None)
    session.pop("uploaded_datasets", None)
    session.pop("selected_dataset", None)
    return redirect("/")

def is_doc_question(query):
    keywords = ["mean", "describe", "definition", "what is", "explain", "field", "column", "documentation"]
    return any(kw in query.lower() for kw in keywords)

if __name__ == "__main__":
    app.run(debug=True)
