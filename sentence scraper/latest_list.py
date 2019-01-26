import requests
import zipfile
import io
from pyperclip import paste
from selenium import webdriver


r = requests.get('https://chromedriver.storage.googleapis.com/2.45/chromedriver_win32.zip', stream=True)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()

f_name = "list.txt"
site = "https://www.proxy-list.download/HTTPS"

driver = webdriver.Chrome()
driver.get(site)
button = driver.find_element_by_id("btn3")
button.click()
driver.close()
txt = paste()

with open(f_name, "w") as o:
    for l in txt.split("\n"):
        o.write("https://" + l)

