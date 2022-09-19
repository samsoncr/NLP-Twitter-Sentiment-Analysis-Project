import requests
import os
import json
from pprint import pprint


class TwitterRequest:
    def __init__(self, bearer_token):
        self.bearer_token = bearer_token
        self.params = {}
        self.expansions = []

    def execute(self):
        url = 'https://api.twitter.com/2/tweets/search/recent'

        self.params['expansions'] = ",".join(sorted(self.expansions))

        response = requests.request(
            "GET",
            url,
            params=self.params,
            headers={
                'Content-Type':'application/json',
                'Authorization': 'Bearer {}'.format(self.bearer_token)
            }
        )

        return response


    def geo_place_id(self):
        self.expansions.append('geo.place_id')
        return self

    def author_id(self):
        self.expansions.append('author_id')
        return self

    def attachments_media_keys(self):
        self.expansions.append('attachments.media_keys')
        return self

    def all_expansions(self):
        self.geo_place_id()
        self.author_id()
        self.attachments_media_keys()
        return self

    def max_results(self, limit):
        self.params['max_results'] = limit
        return self

    def query(self, query):
        self.params['query'] = query
        return self


class QueryBuilder():
    def init(self):
        pass



def main():
    # How to send a request to twitter
    query = 'has:media goodbye'
    bearer_token = 'AAAAAAAAAAAAAAAAAAAAAG94gwEAAAAAwzQcTDdzBIeteH4muf5Gyu8cfhA%3DE4oA9XannUhfFnPKsnAnvWXzY5SJrg9EsgglejnNCEnDrQlRQM'

    tr = TwitterRequest(bearer_token)
    tr.query(query)
    tr.max_results(10)
    tr.all_expansions()

    response = tr.execute()

    print(response.status_code)
    pprint(response.json())


if __name__ == "__main__":
    main()