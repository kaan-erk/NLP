# Individual Contribution Report

**Student Name:** Kaan Erk Toraman  
**Student ID:** 220901564  
**GitHub Username:** [kaan-erk](https://github.com/kaan-erk)  
**Assigned Module:** Web Scraping, Data Collection, and Dataset Construction (`scraper.py`)  

---

## 1. Overview of Contribution
In this project, my primary responsibility was the complete design, implementation, and execution of the **Data Acquisition & Web Scraping Pipeline** (`scraper.py`). I constructed the foundational raw dataset (`news_dataset_4000.csv`) consisting of exactly 4,000 balanced news headlines across four distinct domains: **Politics**, **Finance**, **Weather**, and **Sport**.

---

## 2. Detailed Technical Contributions in `scraper.py`

### 2.1 Domain & Query Engineering
- Curated **60 targeted search keywords** (15 queries per category) designed to capture diverse, real-world, and domain-representative news headlines.
- Categories handled:
  - **Politics**: Presidential elections, White House, Congress, Parliament, diplomacy, policy legislation, international relations.
  - **Finance**: Wall Street, Federal Reserve interest rates, inflation, corporate earnings, cryptocurrency, commodity prices, venture capital.
  - **Weather**: Severe storms, hurricanes, tornadoes, heatwaves, blizzards, floods, cold fronts, droughts.
  - **Sport**: Premier League, NBA, Tennis Grand Slams, Formula 1, UEFA Champions League, MLB, UFC, Olympics.

### 2.2 Automated RSS Scraping Engine
- Integrated Google News RSS feeds (`https://news.google.com/rss/search?q=...`) using `requests` and XML parsing with `BeautifulSoup` (`lxml`/`xml` backend).
- Configured custom HTTP browser headers (`User-Agent`) to prevent bot detection and rate limits.
- Built rate-limiting mechanisms (`time.sleep(0.3)`) and exception handling to ensure uninterrupted data collection.

### 2.3 Headline Sanitization & Cleaning
- **Publisher Suffix Stripping:** Implemented regular expression cleaning (`re.sub(r'\s*-\s*[^-]+$', '', title)`) to dynamically detect and remove news source tags (such as ` - BBC`, ` - Reuters`, ` - CNN`, ` - The Guardian`) from headline tails. This prevents models from memorizing publisher names instead of topic content.
- **Length & Quality Filters:** Filtered out excessively short titles (< 15 characters) and empty artifacts.

### 2.4 Deduplication & Class Balancing
- Implemented real-time `set`-based deduplication during fetching to eliminate redundant coverage across overlapping news events.
- Structured an automated class quota manager guaranteeing **strictly 1,000 unique records per category**, creating an ideal balanced dataset with no class imbalance bias.
- Serialized the final collected dataset to `news_dataset_4000.csv` with `utf-8-sig` encoding for cross-platform compatibility.

---

## 3. Summary of Deliverables & Files Produced
- **Script:** [`scraper.py`](../scraper.py) — Complete, modular, reproducible Python scraping script.
- **Dataset:** [`news_dataset_4000.csv`](../news_dataset_4000.csv) — 4,000 labeled and balanced news headlines serving as the input for preprocessing.
- **Documentation:** Inline comments and docstrings detailing the scraping logic, query configurations, and cleaning steps.
