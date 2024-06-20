## Rag Chatbot using Langchain and LLamaindex for Versa.

# How To Run:
1) Git clone the latest branch i.e RagUnstructured2.1
2) Use " pip install -r requirements.txt" to install all the dependencies.
3) Enter your Openai Api key in "config.py" 
4) Once faiss index is created , it will show in your directory.
5) Run "streamlit run app.py" to run the application

### *Note*:
Evaluation pipeline automatically evaluates every query on backend and saves the data in evaluation_results.csv.
To run, add a .env file and add openai key, and langsmith key.

#### Files:
1) Avatar.py is responsible for generating avatar.
2) agent.py is responsible for trivia generation using agent.
3) secondarydb folder contains all the necessary files for secondary database.