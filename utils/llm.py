import os
import base64
import requests
from openai import OpenAI

import time

class LLMClient:
    def __init__(self):
        self.api_key = os.environ.get('OPENAI_API_KEY')
        self.base_url = os.environ.get('OPENAI_BASE_URL')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        if not self.base_url:
            raise ValueError("OPENAI_BASE_URL environment variable is not set")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def get_response_vlm(self, prompt, image_path, model, max_retries=3, retry_delay=2):
        """
        Get response from VLM model with single image input
        """
        base64_image = self.encode_image(image_path=image_path)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": f"{model}", 
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        }

        retries = 0
        while retries <= max_retries:
            try:
                response = requests.post(url=f"{self.base_url}/chat/completions", headers=headers, json=payload)
                
                if response.status_code == 200:
                    try:
                        response_json = response.json()
                        content = response_json['choices'][0]['message']['content']
                        return content
                    except ValueError as e:
                        print(f"Failed to parse JSON response: {e}")
                else:
                    print(f"Request failed with status code: {response.status_code}")
                    print(f"Response content: {response.text}")
            
            except requests.exceptions.RequestException as e:
                print(f"Request failed with exception: {e}")
            
            retries += 1
            if retries <= max_retries:
                print(f"Retrying... Attempt {retries}/{max_retries}")
                time.sleep(retry_delay)
        
        print("Max retries reached. Request failed.")
        return None