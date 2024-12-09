import requests
from bs4 import BeautifulSoup

def scrape_medium(tag):
    url = f"https://medium.com/tag/{tag}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    articles = soup.find_all('h3')

    for i, article in enumerate(articles):
        print(f"{i+1}: {article.text}")

if __name__ == "__main__":
    scrape_medium("machine-learning")

