from urllib.request import urlopen
import json

class LanguageFetch:

    def __init__(self):
        access_key = ''
        ip = urlopen('http://ip.42.pl/anything').read().decode('UTF-8') # test UK 185.86.151.11
        #ip = '185.86.151.11'
        webcall_str = "http://api.ipstack.com/" + ip + "?access_key=" + access_key
        response = urlopen(webcall_str)

        json_string = response.read()
        self.json_lib = json.loads(json_string.decode('UTF-8'))
        response.close()

    def getLanguage_based_on_loc(self):

        language = self.json_lib['location']['languages'][0]['name']
        print(language)
        return language

    def getCountry(self):
        country = self.json_lib['country_name']
        return country

    def getCity(self):
        city = self.json_lib['city']
        return city




