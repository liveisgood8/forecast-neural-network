import requests
from data.dictionaries import *

def get_detectors_sn(id, station_id):
    if id == "liquidPrecipitation":
        if station_id == 5:
            return ["50000018", "50000018_961"]
    elif id == "profileTemperature":
        if station_id == 5:
            return ["50000018"]
        elif station_id == 8:
            return ["40000225PT"]
        elif station_id == 9:
            return ["40000225PT"]
    elif id == "amkFull":
        if station_id == 1:
            return ["15407AMK-03", "15408AMK-03", "15409AMK-03"]
    elif id == 'amkInstant':
        if station_id == 1:
            return ["15407AMK-03", "15408AMK-03", "15409AMK-03"]
        elif station_id == 2:
            return ["15412AMK-03", "15413AMK-03", "15414AMK-03"]
        elif station_id == 3:
            return ["15416AMK-03", "15417AMK-03", "15418AMK-03"]
        elif station_id == 4:
            return ["15419AMK-03", "15420AMK-03", "15421AMK-03"]
        elif station_id == 6:
            return ["12427AMK-03", "14401AMK-03"]
        elif station_id == 7:
            return ["104AMK-03"]
        elif station_id == 8:
            return ["15420AMK-03", "18411AMK-03", "18412AMK-03", "18413AMK-03"]
        elif station_id == 9:
            return ["18411AMK-03"]
        elif station_id == 10:
            return ["12430AMK-03"]
    elif id == "amkR":
        if station_id == 5:
            return ["16402AMK-03"]
    elif id == "dptInstant":
        if station_id == 1:
            return ["15001DTV-01"]
        elif station_id == 2:
            return ["15002DTV-01"]
        elif station_id == 3:
            return ["15003DTV-01"]
        elif station_id == 4:
            return ["15004DTV-01"]
    elif id == "gammaBackground":
        if station_id == 1:
            return ["05425GAMMA"]
        elif station_id == 2:
            return ["01326GAMMA"]
        elif station_id == 3:
            return ["01315GAMMA"]
        elif station_id == 4:
            return ["01195GAMMA"]
        elif station_id == 5:
            return ["50000018_961"]
    elif id == "precipitationMetr":
        if station_id == 1:
            return ["OPT15001OPTIOS"]
    elif id == "radiationFlux":
        if station_id == 9:
            return ["40000225PAR"]
    elif id == "soilHumidity":
        if station_id == 9:
                return ["400002251VLP", "400002252VLP"]


def build_url(start_time, end_time, param_id, station_id, detector_sn, thining = -1):
    urlfinal = "http://mon.imces.ru/t/"
    if thining != -1:
        urlfinal += str(thining) + '/'
    urlfinal += param_id + "/*/tm/"
    urlfinal += start_time + '/'
    urlfinal += end_time + '?'
    urlfinal += "ids_group=" + str(station_id)
    urlfinal += "&sn=" + detector_sn

    return urlfinal


def load_data(url):
    print("Load data from: " + url)
    response = requests.get(url)

    data = response.content.decode("utf-8")[:-2]
    if not data:
        return None
    else:
        return data.split('\n')

