import requests


url = 'http://localhost:9696/predict'

customer_id = 'xyz-123'
customer = {
    'age':10,
    'workclass':'State-gov',
    'fnlwgt':77516.0,
    'education':'Bachelors',
    'education-num':13.0,
    'marital-status':'Never-married',
    'occupation':'Exec-managerial',
    'relationship':'Not-in-family',
    'race':'White',
    'sex':'Male',
    'capital-gain':2174.0,
    'capital-loss':0.0,
    'hours-per-week':40.0,
    'native-country':'United-States'
}

response = requests.post(url, json=customer).json()
print(response)

if bool(response) == True:
    print('Customer income > 50k  %s' % customer_id)
else:
    print('Customer income <= 50k %s' % customer_id)