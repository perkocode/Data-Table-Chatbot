import openai
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import traceback
import re
import uuid
from openai import OpenAI
from dotenv import load_dotenv
from flask import session
load_dotenv()

matplotlib.use('Agg')

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def build_system_prompt(columns_str):
    system_prompt = f"""You are a data analyst assistant. Given a question and column names from a dataframe, return only the Python pandas code needed to answer the question. Use only these columns: [{columns_str}]. The dataframe is named 'df'. Avoid deprecated methods like `Series.append()`. Use pd.concat() instead of append() and avoid assigning directly to a Series with a new label. You may use:
    - `pd` for pandas
    - `plt` for matplotlib plotting if visualization is needed. Important instructions:
    - Never use `plt.show()`
    - Always save charts with `plt.savefig("static/chart.png")`
    - Do not overwrite the result with the plot (i.e. avoid `result = result.plot(...)`)
    - You must assign the final result to a variable called `result`, and plot separately
    - Respond with code only, no explanations. Based on the user's question and the dataframe, generate Python code to answer the question. Use appropriate chart types:
    - Bar chart for comparing categories
    - Line chart for time trends
    - Pie chart only when asking for percentage share
    - Area chart for stacked time series
    - Scatter plot for correlation or numeric comparisons
    
    Use matplotlib or pandas `.plot()` and always assign the result to `result` if it's a DataFrame or Series. Save any plots using `plt.savefig(...)`. Do not include explanations. Respond with code only."""
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
        code = prompt_gpt(query, list(df.columns))
        #local_vars = {"df": df.copy()}
        local_vars = {"df": df.copy(), "plt": plt, "pd": pd}
        print("local_vars:", local_vars)
        code = patch_deprecated_code(code)
        code = code.replace("plt.show()", "")
        print("Generated code:", code)
        try:
            exec(code, {}, local_vars)
        except Exception as e:
            traceback.print_exc()
            return f"Error: {str(e)}", False, None
        result = local_vars.get("result")
        print("Result from execution:", result)
        print("isinstance(result, pd.Series): ",isinstance(result, pd.Series))
        print("isinstance(result, pd.DataFrame): ",isinstance(result, pd.DataFrame))

        if result is None or (hasattr(result, 'empty') and result.empty):
            return "No data found for the specified query. Try changing the filters.", False, None
        
        if isinstance(result, pd.Series) or isinstance(result, pd.DataFrame):
            # Optional: Add chart for Series

            if isinstance(result, pd.Series):
                if result.empty:
                    return "Result is empty and cannot be plotted.", False, None
                plt.clf()
                plt.close('all')
                plt.figure(figsize=(8, 5))
                result.plot(kind="bar", color="skyblue")
                plt.title("Generated Chart")
                plt.tight_layout()
                chart_filename = f"chart_{uuid.uuid4().hex}.png"
                chart_path = f"static/{chart_filename}"
                plt.savefig(chart_path)
                
                plt.close()
                return result.to_string(), True, chart_filename
        
            elif isinstance(result, pd.DataFrame):
                # Optionally, add DataFrame plotting support here
                return result.to_string(), False, None
            
            
            #print("Does it get here? Inside try and if")
            #if plt.get_fignums() and result.index.nlevels == 1: #isinstance(result, pd.Series) 
             #   plt.clf()
              #  plt.close('all')
              #  plt.figure(figsize=(8, 5))
              #  result.plot(kind="bar", color="skyblue")
              #  plt.tight_layout()
                #plt.savefig("static/chart.png")

               # chart_filename = f"chart_{uuid.uuid4().hex}.png"
               # chart_path = f"static/{chart_filename}"
               # plt.savefig(chart_path)
                
              #  plt.close()
              #  return result.to_string(), True, chart_filename

            return result.to_string(), False, None

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
        result = local_vars.get("result")

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
                return result.to_string(), True, chart_filename

            return result.to_string(), False, None

        return result.to_string(), False, None
    except Exception as e:
        traceback.print_exc()
        return f"Error: {str(e)}", False, None
