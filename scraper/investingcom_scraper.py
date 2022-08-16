from bs4 import BeautifulSoup
import requests
import pandas as pd
import json

headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36'}     # to avoid 404 request

# create dictionary to contain results, initiate iteratables keys
news_dict = {}
news_dict['url'] = []
news_dict['headline'] = []
news_dict['raw_article'] = []


# set up loop to get appropriate sample
# check https://www.investing.com/news/stock-market-news/XXXX to get this right

SAMPLE_END = 9750       # corresponds to Jan. 07 2016
SAMPLE_START = 11015  #11015    # corresponds to Dec.27-2011

for i in range(SAMPLE_END, SAMPLE_START):
    try:
        print(f"Scraping page No. {i}")
        html_text = requests.get(f"https://www.investing.com/news/stock-market-news/{i}", headers=headers).text
        soup = BeautifulSoup(html_text, 'lxml')

        articles = soup.find_all('a', class_="title")
        for article in articles:
            try:
                url = "https://www.investing.com" + article.get('href')

                if 'news/stock-market-news' in url and url not in news_dict['url']:                 # to make sure we don't pick up commodity, ads, analysis, etc... | avoid dublicates

                    headline = article.text
                    news_dict['url'].append(url)
                    news_dict['headline'].append(headline)

                    # now: go to url and parse it
                    article_html = requests.get(url, headers=headers).text
                    article_soup = BeautifulSoup(article_html, 'lxml')

                    article_div = article_soup.find('div', class_='WYSIWYG articlePage')
                    paragraphs = article_div.find_all('p')

                    raw_article = "".join([paragraph.text for paragraph in paragraphs])

                    # check if article is from the International Business Times (different parsing structure)
                    linkedd = paragraphs[0].find('a')
                    if linkedd:
                        if 'ibtimes' in linkedd.get('href'):
                            raw_article = "(IBT) " + raw_article

                    news_dict['raw_article'].append(raw_article)

                    # check if everything has the same length
                    if not len(news_dict['url']) == len(news_dict['headline']) == len(news_dict['raw_article']):
                        print("NOT ALL EQUAL LENGTHS")
                        print("LEN URL: ", len(news_dict['url']))
                        print("LEN HL: ", len(news_dict['headline']))
                        print("LEN RA: ", len(news_dict['raw_article']))


                        # get the maximal length one
                        max_len = max([len(news_dict['url']),len(news_dict['headline']), len(news_dict['raw_article'])])
                        print("MAX LEN: ", max_len)

                        # extend all lists with NaN to make them the same length
                        news_dict['url'].extend(['NaN'] * (max_len - len(news_dict['url'])))
                        news_dict['headline'].extend(['NaN'] * (max_len - len(news_dict['headline'])))
                        news_dict['raw_article'].extend(['NaN'] * (max_len - len(news_dict['raw_article'])))

                        print("LISTS HAVE BEEN UPDATED WITH NaN...")
                        print("LEN URL: ", len(news_dict['url']))
                        print("LEN HL: ", len(news_dict['headline']))
                        print("LEN RA: ", len(news_dict['raw_article']))

            except:
                continue
    except:
        continue

# show working example
print(news_dict['headline'][0])
print(news_dict['url'][0])
print(news_dict['raw_article'][0])

# another check for same lenght
if not len(news_dict['url']) == len(news_dict['headline']) == len(news_dict['raw_article']):
    max_len = max([len(news_dict['url']), len(news_dict['headline']), len(news_dict['raw_article'])])
    # extend all lists with NaN to make them the same length
    news_dict['url'].extend(['NaN'] * (max_len - len(news_dict['url'])))
    news_dict['headline'].extend(['NaN'] * (max_len - len(news_dict['headline'])))
    news_dict['raw_article'].extend(['NaN'] * (max_len - len(news_dict['raw_article'])))

# save as json for backup
with open('news_collection_03.json', 'w') as outfile:
    json.dump(news_dict, outfile)


# convert to pandas and save as csv
news_df = pd.DataFrame(news_dict)
news_df.to_csv('news_collection_03.csv')

