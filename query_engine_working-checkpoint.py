import openai
import os
import pandas as pd
import matplotlib.pyplot as plt
import traceback
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def prompt_gpt(query: str, df_columns: list) -> str:
    columns_str = ", ".join(df_columns)
    system_prompt = f"""You are a data analyst assistant. Given a question and column names from a dataframe, return only the Python pandas code needed to answer the question. Use only these columns: [{columns_str}]. The dataframe is named 'df'. Always assign your final output to a variable named 'result'. Do not include explanations. Respond with code only."""
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
        print("Generated code:", code)
        local_vars = {"df": df.copy()}
        print("local_vars:", local_vars)
        exec(code, {}, local_vars)
        result = local_vars.get("result")
        print("Result from execution:", result)
        print("isinstance(result, pd.Series): ",isinstance(result, pd.Series))
        print("isinstance(result, pd.DataFrame): ",isinstance(result, pd.DataFrame))
        if isinstance(result, pd.Series) or isinstance(result, pd.DataFrame):
            # Optional: Add chart for Series
            print("Does it get here? Inside try and if")
            if isinstance(result, pd.Series) and result.index.nlevels == 1:
                plt.figure(figsize=(8, 5))
                result.plot(kind="bar", color="skyblue")
                plt.tight_layout()
                plt.savefig("static/chart.png")
                return result.to_string(), True

            return result.to_string(), False

        return str(result), False
    
    except Exception as e:
        print("Does it get here? Exception")
        traceback.print_exc()
        return f"Error: {str(e)}", False
