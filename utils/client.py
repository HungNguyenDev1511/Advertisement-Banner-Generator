import requests

url = "http://127.0.0.1:5000/generate"
data = {"prompt": "A cat sitting on a laptop"}

response = requests.post(url, json=data)
if response.status_code == 200:
    with open("output.jpg", "wb") as f:
        f.write(response.content)
    print("Image saved as output.jpg")
else:
    print("Error:", response.json())
