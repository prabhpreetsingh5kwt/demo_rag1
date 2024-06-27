from flask import Flask, request, jsonify
from datetime import datetime
import uuid
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from secondarydb.semantic_srch_utils import search_with_faiss,create_embeddings_write_index
from app_utils import pdf_qa,sentiment,generate_unique_filename
from agent import preprocess
from avatar import get_avatar
import os

app = Flask(__name__)

langchain_api_key=os.getenv("langchain_api_key")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ['LANGCHAIN_PROJECT'] = "Versa" 

#trivia agent executor     
executor=preprocess()

@app.route('/userQuery', methods=['POST'])
def user_query():
    # Extract data from the request
    data = request.get_json()
    query = data.get('query', '')
    deal_id = data.get('dealId','')
    state = data.get('state', '')

    # Generate unique identifiers for user and session (if needed)
    user_id = str(uuid.uuid4())

    # Get current datetime
    created_at = datetime.now().isoformat()

    # Prepare response
    
    # Separate session for each user  
    session=deal_id
    memory = MongoDBChatMessageHistory(connection_string="mongodb+srv://prabhpreets5kwt:w5jtynqcht6sq3r9@cluster0.ma2wfua.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",session_id=session)
    

    #semantic search:
    response_faiss,video_path,score=search_with_faiss(query)


    # Response from RAG
    if score>0.50:
        
        response=response_faiss
        print("score"*10,score)
        print(response)
        memory.add_user_message(query)
        memory.add_ai_message(response)

    else:

        response=pdf_qa({"question": query,"chat_history": memory.messages})
        rephrased_query=response['generated_question']
        response=str(response['answer']).replace("$","\$")
        
        #sentiment analyzer and product extractor for user query which returns a query for trivia agent
        sentiment_response=sentiment(rephrased_query)
        if sentiment_response['sentiment']=='negative':
            product=sentiment_response['product']
            query=f"{product} in {state}"
            print(query)
        
            #executor block for trivia agent
            exec_response = executor.invoke({"input": query, "chat_history": []})
            exec_response=exec_response['output']
            exec_response=exec_response.replace("$","\$")
            
            #Add trivia to response
            response=response+' '+ '\n\n'+exec_response

        memory.add_user_message(rephrased_query)
        memory.add_ai_message(response)

    video_path = generate_unique_filename()

    #"""Download video and read dataframe"""

    # video=urllib.request.urlretrieve(result_url,video_path)
    # source='secondarydb/datasource.csv'
    # check_and_create_csv(source)

    #"""Append query response and video path to secondary database"""
    # append_to_csv(source,rephrased_query,response,video_path)
    # 
    # NEED TO CREATE EMBEDDINGS AGAIN AND UPDATE THE FAISS INDEX FOR SECONDARYDB
    # create_embeddings_write_index()

    
    jsonresponse = {
        "userId": user_id,
        "query": query,
        "dealId": deal_id,
        "state": state,
        "createdAt": created_at,
        "llm_output": response
    }

    # response=pdf_qa(query,sesion)
    return jsonify(jsonresponse), 200










if __name__ == '__main__':
    app.run(debug=True)
