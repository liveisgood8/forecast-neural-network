#Модуль для подгрузки данных с сервиса
# https://www.ncdc.noaa.gov/cdo-web/webservices/v2

import requests
import json


def getData():
    response = requests.get(
        'https://www.ncdc.noaa.gov/cdo-web/api/v2/data?datasetid=GHCND&locationid=CITY:AE000001&startdate=2010-05-01&enddate=2010-05-15&limit=50',
        headers = {'token': 'IKMNEfkXQSpsVUOVHnWRoQFcEJXpdnGw'}
    )

    #Градусы возвращаются с десятыми, т.е. TMAX=337 ~ TMAX=33.7
    #datasetid - GHCND - Daily Summaries
    #locationid - CITY
    #{
    #    "mindate": "1983-01-01",
    #    "maxdate": "2018-10-09",
    #    "name": "Abu Dhabi, AE",
    #    "datacoverage": 1,
    #    "id": "CITY:AE000001"
    #},

    data = json.loads(response.text)

    return data
