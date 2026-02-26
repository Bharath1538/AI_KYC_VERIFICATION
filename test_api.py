import requests

url = "http://localhost:5001/api/v1/kyc/verify/document"
headers = {
    "X-API-Key": "sk_test_1234567890abcdef"
}
files = {
    "file": ("test.jpg", open("/Users/av/Capstone/AI_KYC_VERIFICATION/AI_KYC_VERIFICATION/aadhar1.jpg", "rb"), "image/jpeg")
}

response = requests.post(url, headers=headers, files=files)
print(f"Status Code: {response.status_code}")
print(f"Response: {response.json()}")
