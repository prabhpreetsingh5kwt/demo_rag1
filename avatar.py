import requests
import time
import os
from dotenv import load_dotenv
load_dotenv()
did_authorization=os.getenv("did_authorization")

url = "https://api.d-id.com/talks"
headers = {
        "accept": "application/json",
        "content-type": "application/json",
         "authorization": did_authorization
    }

def check_status_get(task_id):
    
    status_url = f"{url}/{task_id}"  # assuming the status URL includes the task_id
    response = requests.get(status_url, headers=headers)
    return response

def get_avatar(prompt):

    url = "https://api.d-id.com/talks"

    payload = { "source_url": "https://i.ibb.co/StrDHM2/versa.png"
            ,"script": {
            "type": "text",
            "subtitles": "false",
            "provider": {
                "type": "microsoft",
                "voice_id": "en-US-JennyNeural"
            }
        ,"input": prompt} }
    # headers = {
    #     "accept": "application/json",
    #     "content-type": "application/json",
    #     "authorization": did_authorization
    # }

    response1 = requests.post(url, json=payload, headers=headers)

    if response1.status_code == 201:
        response_data = response1.json()
        task_id = response_data['id']  # assuming the response includes a task_id
    else:
        print(f"Failed to start task: {response1.status_code} {response1.text}")
        exit(1)


    while True:
        get_response = check_status_get(task_id)
        if get_response.status_code == 200:
            status_data = get_response.json()
            if status_data['status'] == 'done':
                result_url=status_data['result_url']
                print("result_url=======",status_data['result_url'])
                print("Task completed successfully")
                return result_url

                break
            elif status_data['status'] == 'failed':
                print(f"Task failed: {status_data['error_message']}")
                break
            else:
                print(f"Task status: {status_data['status']}")
        else:
            print(f"Failed to check status: {get_response.status_code} {get_response.text}")
            break
        
        # Wait before the next check
        time.sleep(1.0)

    print('-' * 20)

# response=get_avatar("5kwebtech")
# print('response===',response)