import requests
import json

# 测试API登录
print("Testing API login...")
login_url = "http://127.0.0.1:8000/api/login/"
login_data = {
    "username": "testuser",
    "password": "testpass123"
}

response = requests.post(login_url, json=login_data)
print(f"Login response status: {response.status_code}")
print(f"Login response: {response.text}")

# 如果登录成功，测试JWT认证
if response.status_code == 200:
    login_result = response.json()
    token = login_result.get('token')
    
    if token:
        print("\nTesting JWT authentication...")
        test_url = "http://127.0.0.1:8000/api/test-jwt/"
        headers = {
            "Authorization": f"Bearer {token}"
        }
        
        test_response = requests.get(test_url, headers=headers)
        print(f"JWT test response status: {test_response.status_code}")
        print(f"JWT test response: {test_response.text}")
    else:
        print("No token found in login response")
else:
    print("Login failed, cannot test JWT authentication")
