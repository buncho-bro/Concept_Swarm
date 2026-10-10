"""
【ファイル概要】
web_search.py

必要に応じて外部のウェブ検索APIを叩き、取得したテキストを即座に概念ベクトルへ変換してブラックボードに投下するツール。
"""

import urllib.request
import json
import urllib.parse

class WebSearchTool:
    def __init__(self):
        self.name = "Web Search"

    def search(self, query, max_results=3):
        """
        Perform a simple web search using Wikipedia API as a fallback,
        since DuckDuckGo/Google requires external libraries or API keys.
        This provides 'raindrops' (context) for the Ripple Engine.
        """
        # Windowsのコンソール(cp932)で絵文字を出力するとクラッシュするため絵文字を削除
        print(f"[Web Search] Searching for: {query}")
        try:
            url = f"https://ja.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&utf8=&format=json"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode('utf-8'))
                
            results = []
            for item in data['query']['search'][:max_results]:
                snippet = item['snippet'].replace('<span class="searchmatch">', '').replace('</span>', '')
                results.append(f"Title: {item['title']}\nSnippet: {snippet}")
            
            if not results:
                return "No results found."
            
            return "\n\n".join(results)
            
        except Exception as e:
            return f"Search failed: {str(e)}"
