from apify_client import ApifyClient
import pandas as pd

def get_twitter_data(api_token, keyword, max_items=10):
    """
    Fungsi untuk scraping Twitter menggunakan Apify Client.
    """
    client = ApifyClient(api_token)

    run_input = {
        "searchTerms": [keyword],
        "maxItems": max_items,
        "queryType": "Latest",
        "lang": "en", 
        "filter:replies": False,
        "filter:quote": False,
    }

    try:
        run = client.actor("CJdippxWmn9uRfooo").call(run_input=run_input)

        dataset_items = client.dataset(run["defaultDatasetId"]).list_items().items
        
        if not dataset_items:
            return pd.DataFrame()

        return pd.DataFrame(dataset_items)
    
    except Exception as e:
        raise Exception(f"Error saat scraping: {str(e)}")