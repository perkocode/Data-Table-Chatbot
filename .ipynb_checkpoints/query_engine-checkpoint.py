import openai
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import squarify  # pip install squarify
import traceback
import re
import uuid
from openai import OpenAI
from dotenv import load_dotenv
from flask import session

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI

import os
from dotenv import load_dotenv
#import logging

#logging.basicConfig(
#    level=logging.DEBUG,
#    format='%(asctime)s %(levelname)s:%(message)s'
#)

#logger = logging.getLogger(__name__)
#logger.info("Starting query_engine.py")

load_dotenv()

matplotlib.use('Agg')
loader = TextLoader("docs/tableau_superstore_data_dictionary.txt", encoding="utf-8")

try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("client: ", client)
except Exception as e:
    print("Failed to create OpenAI client:", e)
    raise
    
#client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def build_system_prompt(columns_str):
    system_prompt = f"""You are a data analyst assistant. Given a question and the dataframe's columns, return only the pandas code to answer the question. The dataframe is named `df`. Avoid deprecated methods like `Series.append()`. Use pd.concat() instead of append() and avoid assigning directly to a Series with a new label. You may use:
    - `pd` for pandas
Rules:
- Use only these columns: [{columns_str}]
- Set the final data result in a variable named `result`.
- Set an appropriate chart type in a string variable named `chart_type`. Choose one of: "bar", "barh", "line", "pie", "scatter", "treemap".
- If the chart is a scatter plot, assign the x and y axis columns to `x_col` and `y_col` respectively.
- If the chart is a treemap, assign the label column to `label_col` and the size column to `size_col`.
- Do NOT plot anything or use plt.
- Do NOT import anything or print anything.
- Do NOT explain the code. Just return Python code that defines `result`, `chart_type`, and any relevant plotting parameters.
"""
    return system_prompt

def patch_deprecated_code(code: str) -> str:
    # Replace deprecated Series.append() with pd.concat
    pattern = r'(\w+)\s*=\s*\1\.append\((.*)\)'
    replacement = r'\1 = pd.concat([\1, \2])'
    return re.sub(pattern, replacement, code)

def prompt_gpt(query: str, df_columns: list) -> str:
    columns_str = ", ".join(df_columns)
    system_prompt = build_system_prompt(columns_str)
    print("system_prompt: ",system_prompt)
    user_prompt = f"Question: {query}"
    print("user_prompt: ",user_prompt)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0,
    )
    print("response: ",response)
    return response.choices[0].message.content
    #response["choices"][0]["message"]["content"] #response.choices[0].message.content

def handle_query(query, df):
    try:
        chart_filename = chart_path = data_path = code_path = None
        code = prompt_gpt(query, list(df.columns))
        #local_vars = {"df": df.copy()}
        local_vars = {"df": df.copy(), "plt": plt, "pd": pd}
        print("local_vars:", local_vars)
        code = patch_deprecated_code(code)
        code = code.replace("plt.show()", "")
        print("Generated code:", code)

        # Save code
        # code_filename = f"code_{uuid.uuid4().hex}.py"
        # code_path = os.path.join("static", code_filename)
        # with open(code_path, "w") as f:
        #     f.write(code)
        # session["code_filename"] = code_filename

        try:
            exec(code, {}, local_vars)
            #code_filename = f"code_{uuid.uuid4().hex}.py"
            #code_path = f"static/{code_filename}"
            #with open(code_path, 'w') as f:
            #    f.write(code)
        except Exception as e:
            traceback.print_exc()
            return f"Error: {str(e)}", False, None
        result = local_vars.get("result")
        print("Result from execution:", result)
        print("isinstance(result, pd.Series): ",isinstance(result, pd.Series))
        print("isinstance(result, pd.DataFrame): ",isinstance(result, pd.DataFrame))

        if isinstance(result, (pd.DataFrame, pd.Series)):
             data_filename = f"data_{uuid.uuid4().hex}.csv"
             data_path = os.path.join("static", data_filename)
             print("data_filename: ",data_filename)
             print("data_path: ",data_path)
             result.to_csv(data_path)
             session["data_filename"] = data_filename
             print("session[data_filename]: ",isinstance(result, pd.DataFrame))

        
        chart_type = local_vars.get("chart_type", "bar")  # Default to bar if not set
        print("chart_type: ",chart_type)
        print("isinstance(result, pd.DataFrame): ",isinstance(result, pd.DataFrame))
        
        if result is None or (hasattr(result, 'empty') and result.empty):
            return "No data found for the specified query. Try changing the filters.", False, None
        
        if isinstance(result, pd.Series) or isinstance(result, pd.DataFrame):
            print("Inside if isinstance(result, pd.Series) or isinstance(result, pd.DataFrame))")

            if isinstance(result, pd.Series):
                print("Inside isinstance(result, pd.Series)")
                print("result.empty: ",result.empty)
                if result.empty:
                    return "Result is empty and cannot be plotted.", False, None
                plt.clf()
                plt.close('all')
                plt.figure(figsize=(8, 5))

                if chart_type == "pie":
                    result.plot(kind="pie", autopct="%1.1f%%", legend=False)
                    plt.ylabel("")
                    print("Plotted pie")
                elif chart_type == "line":
                    result.plot(kind="line", marker="o")
                    print("Plotted line")
                elif chart_type == "barh":
                    result.plot(kind="barh", color="skyblue")
                    print("Plotted barh")
                elif chart_type == "scatter":
                    x_col = local_vars.get("x_col")
                    y_col = local_vars.get("y_col")
                    if x_col and y_col and x_col in df.columns and y_col in df.columns:
                        plt.scatter(df[x_col], df[y_col], alpha=0.7)
                        plt.xlabel(x_col)
                        plt.ylabel(y_col)
                    else:
                        raise ValueError("Missing or invalid 'x_col' and 'y_col' for scatter plot.")
                    print("Plotted scatter")
                elif chart_type == "treemap":
                    result = result.reset_index()  # Now has dimension and measure columns

                    # Default to first two columns (e.g., Category, Sales)
                    default_label_col = result.columns[0]
                    default_size_col = result.columns[1]
                    
                    # Try to override from local_vars, but only use if they are valid column names
                    label_col = local_vars.get("label_col")
                    size_col = local_vars.get("size_col")
                
                    # If user-defined vars are not strings or invalid, fall back to defaults
                    if not isinstance(label_col, str) or label_col not in result.columns:
                        label_col = default_label_col
                    if not isinstance(size_col, str) or size_col not in result.columns:
                        size_col = default_size_col
                
                    # Log helpful debug info
                    print("Resolved label_col:", label_col)
                    print("Resolved size_col:", size_col)
                    print("result.columns:", result.columns)
                
                    # Proceed with plotting
                    labels = result[label_col].astype(str) + "\n" + result[size_col].round(0).astype(str)
                    squarify.plot(sizes=result[size_col], label=labels, alpha=0.8)
                    plt.axis("off")
                    print("Plotted treemap")
                else:
                    result.plot(kind="bar", color="skyblue")
                    print("Plotted bar")

                
                plt.title("Generated Chart")
                plt.tight_layout()
                chart_filename = f"chart_{uuid.uuid4().hex}.png"
                chart_path = f"static/{chart_filename}"

                print("chart_filename: ", chart_filename)
                print("chart_path: ", chart_path)
                
                plt.savefig(chart_path)
                plt.close()
                return result.to_string(), True, chart_filename
        
            elif isinstance(result, pd.DataFrame):
                print("In isinstance(result, pd.DataFrame)")
                print("result.shape[1]: ",result.shape[1])
                print("chart_type: ",chart_type)
                if chart_type == "line" and result.shape[1] == 2:
                    print("In chart_type == 'line' and result.shape[1] == 2")
                    plt.clf()
                    plt.close('all')
                    plt.figure(figsize=(8, 5))
            
                    x = result.columns[0]
                    y = result.columns[1]
            
                    result = result.dropna(subset=[x, y])  # prevent NaN errors
            
                    plt.plot(result[x], result[y], marker="o")
                    plt.xlabel(x)
                    plt.ylabel(y)
                    plt.title("Generated Chart")
                    plt.tight_layout()
            
                    chart_filename = f"chart_{uuid.uuid4().hex}.png"
                    chart_path = f"static/{chart_filename}"
                    plt.savefig(chart_path)
                    plt.close()
            
                    return result.to_string(), True, chart_filename
                elif chart_type == "scatter":
                    x_col = local_vars.get("x_col")
                    y_col = local_vars.get("y_col")
                    print("Inside scatter chart_type check")
                    print("result.columns:", result.columns)
                    print("x_col:", x_col, "y_col:", y_col)
            
                    if (
                        x_col and y_col
                        and x_col in result.columns
                        and y_col in result.columns
                    ):
                        plt.clf()
                        plt.close('all')
                        plt.figure(figsize=(8, 5))
            
                        result = result.dropna(subset=[x_col, y_col])
                        plt.scatter(result[x_col], result[y_col], alpha=0.7)
                        plt.xlabel(x_col)
                        plt.ylabel(y_col)
                        plt.title("Generated Scatter Plot")
                        plt.tight_layout()
            
                        chart_filename = f"chart_{uuid.uuid4().hex}.png"
                        chart_path = f"static/{chart_filename}"
                        plt.savefig(chart_path)
                        plt.close()
            
                        return result.to_string(), True, chart_filename
                    else:
                        print("x_col or y_col not in result.columns — skipping scatter plot")
                return result.to_string(), False, None


            print("At bottom of isinstance(result, pd.Series) and about to return False for is_chart")
            return result.to_string(), False, None

        print("Returning False for is_chart")
        return result.to_string(), False, None
    
    except Exception as e:
        print("Does it get here? Exception")
        traceback.print_exc()
        return f"Error: {str(e)}", False, None

def handle_query_with_history(messages, df):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0,
        )
        code = response.choices[0].message.content
        local_vars = {"df": df.copy(), "pd": pd, "plt": plt}
        code = patch_deprecated_code(code)
        code = code.replace("plt.show()", "")
        exec(code, {}, local_vars)
        #code_filename = f"code_{uuid.uuid4().hex}.py"
        #code_path = f"static/{code_filename}"
        #with open(code_path, 'w') as f:
        #    f.write(code)
        result = local_vars.get("result")

        #data_filename = f"data_{uuid.uuid4().hex}.csv"
        #data_path = f"static/{data_filename}"
        #result.to_csv(data_path)
        
        if isinstance(result, pd.Series) or isinstance(result, pd.DataFrame):
            plt.clf()
            plt.close('all')
            if isinstance(result, pd.Series) and result.index.nlevels == 1:
                plt.clf()
                plt.close('all')
                plt.figure(figsize=(8, 5))
                result.plot(kind="bar", color="skyblue")
                plt.tight_layout()
                #filename = f"chart_{uuid.uuid4().hex}.png"

                chart_filename = f"chart_{uuid.uuid4().hex}.png"
                chart_path = f"static/{chart_filename}"
                plt.savefig(chart_path)
                
                #plt.savefig(f"static/{filename}")
                #session["chart_filename"] = filename
                session["chart_filename"] = chart_filename
                #plt.savefig("static/chart.png")
                return result.to_string(), True, chart_filename #, chart_path, data_path, code_path

            return result.to_string(), False, None #, None, data_path, code_path

        return result.to_string(), False, None #, None, data_path, code_path
    except Exception as e:
        traceback.print_exc()
        return f"Error: {str(e)}", False, None #, None, None, None
