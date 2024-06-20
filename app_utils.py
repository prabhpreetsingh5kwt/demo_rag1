# Responsible for creating a conversational chain
import os
import uuid
import requests
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
from langchain.prompts import SystemMessagePromptTemplate , HumanMessagePromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from vector_store import create_index
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.memory import MongoDBChatMessageHistory

load_dotenv()


openai_key=os.getenv("openai_key")


llm=ChatOpenAI(api_key=openai_key,model="gpt-4o")

#System prompt(Few shot prompts for versa's personality traits)
system_template ="""
You are Versa, an AI Virtual Sales Assistant. You embody the qualities of a top-tier salesperson: you are charismatic, very very persuasive,
and deeply knowledgeable about the products you sell. Your demeanor is friendly and professional,
always aiming to make the customer feel valued and understood. Your ultimate goal is to sell products,
but you do so by building trust and providing insightful information tailored to the user's needs.
Use this knowledge below to answer the question in a very short and concise way while maintaining your comforting tone, personality and advocating on why it is a bad idea to not to buy the mentioned product.
"{context}"

Few-shot examples:
Q: What are the benefits of your premium insurance plan?
A: Our premium insurance plan offers comprehensive coverage, including collision, theft, and natural disasters. It ensures maximum protection and peace of mind for you. For detailed pricing information, please check with your finance manager.

Q: I think the basic plan is enough for me.
A: While the basic plan provides essential coverage, the premium plan adds extra layers of protection and benefits, ensuring you are fully safeguarded against unexpected events. Investing in premium coverage is a smart decision for long-term peace of mind.

Q: How much does the premium insurance cost?
A: The exact pricing can vary based on several factors. It's best to consult with your finance manager for detailed pricing information tailored to your specific needs.

Q: Can you explain the benefits of GAP insurance?
A: GAP insurance covers the difference between what you owe on your vehicle and its actual value in the event of a total loss. This ensures you're not left paying out of pocket. It’s a smart investment to protect yourself from unexpected financial burdens.

Q: I’m not sure if I need GAP insurance.
A: GAP insurance acts as a financial safety net. If your vehicle is totaled, it covers the gap between your loan balance and the insurance payout. This coverage can save you significant money and stress in such situations.

Q: What exactly does the collision coverage include?
A: Collision coverage includes damages resulting from a collision with another vehicle or object. It ensures repairs are covered without additional costs to you. For more detailed technical information, your finance manager would be the best person to consult.

Q: Why should I upgrade to the premium plan?
A: Upgrading to the premium plan gives you access to enhanced coverage options, including comprehensive protection against various risks. It’s designed to offer you complete peace of mind and ensure you’re always covered, no matter the situation.

Q: Is GAP insurance really worth it?
A: Absolutely! Without GAP insurance, you could be left paying the difference between your car's value and what you owe in the event of a total loss. It’s a small investment for significant financial security and peace of mind.
**MAKE THE ANSWER TONE AS AMERICAN AS POSSIBLE AND DUMB DOWN ALL THE BIG WORDS**
"""


messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
        ]

qa_prompt = ChatPromptTemplate.from_messages(messages)

#Rephrased query prompt
custom_template = """

**Task Description:**

Rephrase the follow-up question in the given conversation to make it a standalone question. Ensure the rephrased question retains the original sentiment. For negative queries, such as "I do not want it," "I won't need it," or "I don't need it,", or "i can't afford it", rephrase them to be specific, for example, "I do not need windshield protection" or "I can't afford windshield protection."

**Examples:**

1. **Conversation:**  
   - Person A: "I am looking at the car insurance plans."  
   - Follow-up Question: "Do I need it if I already have health insurance?"  
   - **Standalone Question:** "Do I need car insurance if I already have health insurance?"

2. **Conversation:**  
   - Person A: "I'm considering different options for coverage."  
   - Follow-up Question: "I don't think I need it."  
   - **Standalone Question:** "I don't think I need windshield protection."

**Instructions:**
1. Identify the follow-up question in the conversation.
2. Rephrase it as a standalone question.
3. Ensure the rephrased question maintains the sentiment of the original follow-up question.
4. For negative queries, make them specific as indicated in the examples.

---

This prompt maintains the original intent while adding clarity and examples to guide the user effectively.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT_CUSTOM = PromptTemplate.from_template(custom_template)  
vectorstore=create_index()
retriever=vectorstore.as_retriever(search_kwargs={'k': 3})
# memory = MongoDBChatMessageHistory(connection_string="mongodb+srv://prabhpreets5kwt:w5jtynqcht6sq3r9@cluster0.ma2wfua.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",session_id="test1")
# memory = ConversationBufferMemory(memory_key="chat_history",output_key="answer", return_messages=True)
pdf_qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever,
 combine_docs_chain_kwargs={"prompt": qa_prompt},return_generated_question=True,condense_question_prompt=CONDENSE_QUESTION_PROMPT_CUSTOM,verbose=True)

def download_video(url, save_path):
    
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"Video downloaded successfully and saved to {save_path}")
    else:
        print(f"Failed to download video. Status code: {response.status_code}")


# Function to check if CSV exists and create it if not
def check_and_create_csv(path):
    if not os.path.exists(path):
        df = pd.DataFrame(columns=["user_input", "llm_output", "video_path"])
        df.to_csv(path, index=False)
    return pd.read_csv(path)

# Function to append a new row to the CSV
def append_to_csv(path, user_input, llm_output, video_path):
    df = pd.read_csv(path)
    new_row = {"user_input": user_input, "llm_output": llm_output, "video_path": video_path}
    df = df._append(new_row, ignore_index=True)
    df.to_csv(path, index=False)

# Function to generate a unique file name
def generate_unique_filename():
    while True:
        unique_id = uuid.uuid4()
        video_name = f'video_{unique_id}.mp4'
        video_path = os.path.join('secondarydb','videos', video_name)
        if not os.path.exists(video_path):
            return video_path
        

def sentiment(rephrased_query):
  sentiment_schema = ResponseSchema(name='sentiment',description="This is the sentiment which is either 'neutral' or 'negative'")
  product_schema=ResponseSchema(name='product',description="this is the product")
  response_schemas =[sentiment_schema,product_schema]

  output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
  format_instructions=output_parser.get_format_instructions()
  template_string="""you are a extended warranty salesperson. take the user query below and try to rate the query sentiment. if user asks
  about a product,or if there is discussion about the product, then the query is neutral. but if user says something negative or something which shows he is repulsive to buy the product, for example:
  'i do not want it','i cannot afford it', ' i do not see value in it' or 'my wife will not let me buy it', then return negative. 
  take the query below delimited by triple backticks and use it to define sentiment.
  query:```{query}```

  then from the query, extract the insurance product.
  The possible products are:
      - GAP
      - GPS
      - Theft Protection/Recovery
      - Windshield Protection
      - Dent & Ding
      - Appearance Protection
      - Lifetime Coverage
      - Prepaid Maintenance
      - Lease Wear & Tear
      - Key Coverage/Replacement
      - Tire & Wheel
      return only product name for exa
      : GAP, GPS, Dend & Ding etc.

    {format_instructions}
    """
  prompt=ChatPromptTemplate.from_template(template=template_string)
  messages=prompt.format_messages(query=rephrased_query,format_instructions=format_instructions)
  parserresponse=llm(messages)
  parserresponse_as_dict=output_parser.parse(parserresponse.content)
  return parserresponse_as_dict