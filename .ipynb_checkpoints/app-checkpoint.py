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
    result = ""
    chart = None
    session.permanent = True
    query = ""
    response = ""
    is_chart = False
    chart_path = None
    data_path = None
    code_path = None
    timestamp = None

    print("In app.py=>index()")
    if "chat_history" not in session:
        session["chat_history"] = []

    if request.method == "POST":
        query = request.form["question"]
        session.pop("chart_filename", None)
        df = load_data()

        
        # Track user question
        session["chat_history"].append({"role": "user", "content": query})
        conversation = session["chat_history"][-5:]

        if is_doc_question(query):
            # Route to RAG
            rag_result = qa_chain.run(query)
            result = rag_result
            is_chart = False
            chart_url = None
        else:
            columns_str = ", ".join(list(df.columns))
            messages = [{"role": "system", "content": build_system_prompt(columns_str)}]
            messages.extend(conversation)

            # Use one or the other, not both:
            # response_text, is_chart, chart_filename, chart_path, data_path, code_path = handle_query_with_history(messages, df)
            sanitize_prompt(request.form["question"])
            result, is_chart, chart_path = handle_query(query, df)
            print("Back in app.py")
            print("result: ", result)
            print("is_chart: ", is_chart)
            print("chart_path: ", chart_path)
            if chart_path:
                timestamp = datetime.now().timestamp()
                chart_url = url_for('static', filename=os.path.basename(chart_path)) + f"?t={timestamp}"
                session["chart_filename"] = chart_url

            chart_url = (url_for("static", filename=chart_path) + f"?t={datetime.now().timestamp()}"if chart_path else None)
            
            #print("timestamp: ", timestamp)
            #print("chart_url: ", chart_url)
            #print("session['chart_filename']: ", session["chart_filename"])
            #chart_basename = os.path.basename(chart_filename)
            #cache_buster = datetime.now().timestamp()
            #session["chart_filename"] = f"{chart_basename}?t={cache_buster}"
            #print("session['chart_filename'",session["chart_filename"])
            #print("chart: ",chart)

        
        session["chat_history"].append({"role": "assistant", "content": result})

    #chart_url = url_for('static', filename=chart_path)
    #chart_url = url_for('static', filename=chart_filename) + f"?t={timestamp}"

    
    chart_url = (url_for("static", filename=chart_path) + f"?t={datetime.now().timestamp()}"if chart_path else None)
    print("About to call render_template()")
    return render_template(
        "index.html",
        result=result,
        chart=chart,
        chart_url=chart_url,
        timestamp=datetime.now().timestamp(),
        is_chart=is_chart,
        chat_history=session.get("chat_history", []),
        session=session,
        response=result
    )

    
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
        question = request.form["question"]
        dataset_name = request.form["dataset_name"]
        print("dataset_name:", dataset_name)

        # Select dataset
        if dataset_name == "global_superstore":
            df = global_superstore_df
        elif "uploaded_datasets" in session and dataset_name in session["uploaded_datasets"]:
            filepath = session["uploaded_datasets"][dataset_name]
            df = pd.read_csv(filepath)
        else:
            return jsonify({"error": "Dataset not found"}), 400

        session.permanent = True
        session.setdefault("chat_history", [])
        session.pop("chart_filename", None)

        session["chat_history"].append({"role": "user", "content": question})
        conversation = session["chat_history"][-5:]

        # Determine if doc question
        if is_doc_question(question):
            result = qa_chain.run(question)
            chart_url = None
            is_chart = False
        else:
            columns_str = ", ".join(list(df.columns))
            messages = [{"role": "system", "content": build_system_prompt(columns_str)}]
            messages.extend(conversation)

            result, is_chart, chart_path = handle_query(question, df)

            chart_url = (
                url_for("static", filename=os.path.basename(chart_path)) + f"?t={datetime.now().timestamp()}"
                if chart_path else None
            )

            if chart_url:
                session["chart_filename"] = chart_url

        session["chat_history"].append({"role": "assistant", "content": result})

        return jsonify({
            "result": result,
            "chart_url": chart_url,
            "is_chart": is_chart,
            "timestamp": datetime.now().timestamp(),
            "chat_history": session["chat_history"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def is_doc_question(query):
    keywords = ["mean", "describe", "definition", "what is", "explain", "field", "column", "documentation"]
    return any(kw in query.lower() for kw in keywords)

@app.route("/chart")
def chart():
    return send_file("static/chart.png", mimetype='image/png')

@app.route("/clear", methods=["POST"])
def clear():
    session.pop("chat_history", None)
    session.pop("chart_filename", None)
    return redirect("/")

@app.route("/download/chart")
def download_chart():
    chart_filename = session.get("chart_filename")
    if not chart_filename:
        return "No chart available for download", 404
    return send_file(os.path.join("static", chart_filename), mimetype="image/png", as_attachment=True)

@app.route("/download/csv")
def download_csv():
    data_filename = session.get("data_filename")
    if not data_filename:
        return "No CSV available for download", 404
    return send_file(os.path.join("static", data_filename), mimetype="text/csv", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
