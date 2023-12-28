#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
import json
from requests.packages.urllib3.util.ssl_ import create_urllib3_context

url = 'https://127.0.0.1:5000'
text1 = input("input text1: ")
text2 = input("input text2: ")

data = {'text1': text1, 'text2': text2}
headers = {'Content-Type': 'application/json'}

try:
    # Make a POST request with JSON data
    response = requests.post(url, headers=headers, json=data, verify=False)

    # Check the status code
    if response.status_code == 200:
        print('Request successful!')
        print('Response content:', response.text)
    else:
        print('Request failed with status code:', response.status_code)
except requests.exceptions.RequestException as e:
    print('Request failed with error:', e)

