# ==============================================================================
# STEP 1: WEB SCRAPING - 4 CATEGORIES X 1,000 = 4,000 NEWS HEADLINES
# ==============================================================================
# Target Categories:
# 1. Politics
# 2. Finance
# 3. Weather
# 4. Sport
#
# Exactly 1,000 unique articles are collected per category.
# Total: 4,000 labeled and perfectly balanced samples.
# ==============================================================================

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

TOPICS = {
    "Politics": [
        "politics news",
        "white house politics",
        "congress elections",
        "senate political news",
        "government policy legislation",
        "foreign policy geopolitics",
        "political election campaign",
        "parliament prime minister",
        "supreme court ruling politics",
        "presidential election news",
        "diplomacy summit united nations",
        "political debate democracy",
        "cabinet ministers politics",
        "global politics international relations",
        "governor election politics"
    ],
    "Finance": [
        "finance news",
        "stock market wall street",
        "federal reserve interest rates",
        "banking financial markets",
        "inflation central bank economy",
        "corporate earnings revenue",
        "cryptocurrency bitcoin market",
        "global economic growth trade",
        "treasury bond yields",
        "hedge fund investment portfolio",
        "business profit mergers acquisitions",
        "fintech banking shares",
        "oil commodities market prices",
        "recession economic indicators",
        "venture capital startup investment"
    ],
    "Weather": [
        "weather forecast alert",
        "severe weather storm warning",
        "hurricane tropical storm advisory",
        "tornado watch warning weather",
        "heatwave record temperature",
        "flood rainfall warning weather",
        "snowstorm winter blizzard warning",
        "drought climate weather report",
        "national weather service forecast",
        "thunderstorm heavy rain advisory",
        "arctic blast freeze warning",
        "typhoon cyclone weather path",
        "atmospheric river storm forecast",
        "cold front temperature drop",
        "wildfire weather wind conditions"
    ],
    "Sport": [
        "sports news",
        "football premier league soccer",
        "nba basketball playoffs game",
        "tennis grand slam tournament",
        "formula 1 grand prix racing",
        "champions league match result",
        "baseball mlb news home run",
        "nhl hockey news game",
        "world cup championship sport",
        "olympics games athletes",
        "boxing ufc fight night",
        "golf masters pga championship",
        "motogp motorcycle racing news",
        "college football sports",
        "nfl super bowl football"
    ]
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def clean_news_title(title):
    """
    Cleans publisher suffixes at the end of titles (e.g., '- BBC', '- Reuters').
    """
    if not title:
        return ""
    cleaned = re.sub(r'\s*-\s*[^-]+$', '', title).strip()
    return cleaned if len(cleaned) > 10 else title.strip()

def scrape_news(target_per_category=1000):
    """
    Collects at least target_per_category unique news headlines for each topic.
    """
    all_news = []

    print("==================================================")
    print("           NEWS SCRAPING PROCESS STARTED          ")
    print(f"      Target: 4 Categories x {target_per_category} = {4 * target_per_category} Headlines")
    print("      Categories: Politics, Finance, Weather, Sport")
    print("==================================================\n")

    for category, queries in TOPICS.items():
        print(f"\n---> Fetching Category: [{category.upper()}]...")
        category_news = []
        seen_titles = set()

        for query in queries:
            if len(category_news) >= target_per_category:
                print(f"     [!] Target count ({target_per_category}) reached. Moving to next category.")
                break

            encoded_query = query.replace(" ", "+")
            url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

            try:
                response = requests.get(url, headers=HEADERS, timeout=10)
                soup = BeautifulSoup(response.content, "xml")
                items = soup.find_all("item")

                new_count = 0
                for item in items:
                    raw_title = item.title.text if item.title else ""
                    clean_title = clean_news_title(raw_title)

                    if len(clean_title) > 15 and clean_title not in seen_titles:
                        seen_titles.add(clean_title)
                        category_news.append({
                            "text": clean_title,
                            "label": category
                        })
                        new_count += 1
                        if len(category_news) >= target_per_category:
                            break

                print(f"     * '{query}' -> +{new_count} headlines (Category Total: {len(category_news)}/{target_per_category})")
                time.sleep(0.3)

            except Exception as error:
                print(f"     * Error in query '{query}': {error}")

        # Equalize to exact target count
        category_news = category_news[:target_per_category]
        all_news.extend(category_news)
        print(f"     [OK] Exactly {len(category_news)} headlines ready for [{category}].")

    return all_news

if __name__ == "__main__":
    TARGET_COUNT = 1000
    news_data = scrape_news(target_per_category=TARGET_COUNT)

    df = pd.DataFrame(news_data)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    # Ensure exactly TARGET_COUNT per category
    balanced_df = df.groupby("label").head(TARGET_COUNT).reset_index(drop=True)

    print("\n==================================================")
    print("                SCRAPING COMPLETED                ")
    print("==================================================")
    print(f"Total News Articles: {len(balanced_df)}")
    print("\nCategory Distribution (Balanced):")
    print(balanced_df["label"].value_counts())

    output_file = "news_dataset_4000.csv"
    balanced_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\nSaved successfully: {output_file}")
