- Import flat files from the web
	- using `urllib` library
		```python
		from urllib.request import urlretrieve
		urlretrieve(url, "data.csv") #saves it to data.csv
		#performs a get request
		```
	- or just working with pandas and passing the url to pd.read_csv() 
- get requests using urlib
```python
from urllib.request import urlopen, Request
request = Request(url)
response = urlopen(request)
html = response.read()
response.close()
```

- `requests` package
```python
import requests
r = requests.get(url)
text = r.text
```
- Scraping the web in python with beautifulsoup
```python
from bs4 import BeautifulSoup
import requests
html_doc  =requests.get(url).text
soup = BeautifulSoup(html_doc)
print(soup.prettify())#properly indented html
```
	- soup.title gets title of the page
	- soup.get_text() gets all the text
	- for link in soup.find_all('a') : print(link.get"href"))
 