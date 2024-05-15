import pandas
import json
import os
import streamlit as st
from config import openai_key
from raglang_utils import vectorstore
from uptrain import EvalLLM, Evals
from eval_utils import results
os.environ['OPENAI_API_KEY'] = openai_key
import openai

client = openai.OpenAI()


eval_llm = EvalLLM(openai_api_key=openai_key)

result = eval_llm.evaluate(
    data=results,
    checks=[Evals.CONTEXT_RELEVANCE, Evals.FACTUAL_ACCURACY, Evals.RESPONSE_COMPLETENESS,Evals.RESPONSE_RELEVANCE]
)



df=pandas.DataFrame(result)
st.write(df)
# Write the DataFrame to Excel
df.to_csv("evaluation_results.csv", index=False)  # Replace "output.xlsx" with the desired Excel file name