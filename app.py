from flask import Flask, request, render_template, send_file, session, redirect, url_for
# 1. Flask-Limiter (per IP)
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import pandas as pd
from query_engine import handle_query
from query_engine import handle_query_with_history
from query_engine import build_system_prompt
from query_engine import sanitize_prompt
from datetime import datetime
from datetime import timedelta
from rag_engine import setup_rag_chain
import os
import random
#import logging
from dotenv import load_dotenv

load_dotenv()  # this reads .env in the current working directory

qa_chain = setup_rag_chain()

print("Current working directory:", os.getcwd())

#logging.basicConfig(
#    level=logging.DEBUG,
#    format='%(asctime)s %(levelname)s:%(message)s'
#)

#logger = logging.getLogger(__name__)
#logger.info("Starting app.py")
print("print() works")
#logging.debug("logging.debug() works")

app = Flask(__name__)

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100/hour"]
)
limiter.init_app(app)

app.config['RANDOM_CACHE_BUSTER'] = str(random.randint(0, 999999))
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-default-dev-secret")
app.permanent_session_lifetime = timedelta(minutes=30)

# 2. Payload size cap (8 KB)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024  # bytes

# Load data once
df = pd.read_csv("data/global_superstore.csv", encoding='ISO-8859-1')

def load_data():
    return pd.read_csv("data/global_superstore.csv", encoding="ISO-8859-1", parse_dates=["Order Date"])

@app.route("/", methods=["GET", "POST"])
def index():
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
        query = request.form["query"]
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
            sanitize_prompt(request.form["query"])
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
#     return render_template(
#     "index.html",
#     result=result,
#     chart=session.get("chart_filename"),  # <- this enables `{% if chart %}` in the template
#     timestamp=datetime.now().timestamp(),
#     is_chart=is_chart,
#     chat_history=session.get("chat_history", []),
#     session=session,
#     response=result
# )

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

# @app.route("/download/code")
# def download_code():
#     code_filename = session.get("code_filename")
#     if not code_filename:
#         return "No Python code available for download", 404
#     return send_file(os.path.join("static", code_filename), mimetype="text/x-python", as_attachment=True)

def is_doc_question(query):
    keywords = ["mean", "describe", "definition", "what is", "explain", "field", "column", "documentation" ]
    return any(kw in query.lower() for kw in keywords)

if __name__ == "__main__":
    app.run(debug=True)