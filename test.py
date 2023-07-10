import requests
import json

url = 'http://104.198.158.118/predict'
files = {'image': open('xray_images/CHNCXR_0089_0.png', 'rb')}
r = requests.post(url, files=files)
print(r.json())
