import asyncio
import aiohttp
import json
import re
from bs4 import BeautifulSoup
import time
import random
from pathlib import Path

BASE_URL = "https://ethglobal.com"
SHOWCASE_URL_TEMPLATE = BASE_URL + "/showcase?page={}"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Configuration
MAX_CONCURRENT_REQUESTS = 10  # Reduced from 100
REQUEST_DELAY = 0.5  # Seconds between requests
MAX_RETRIES = 3
TIMEOUT = 30  # Seconds

async def fetch_with_retry(session, url, max_retries=MAX_RETRIES):
    """Fetch URL with retry logic and rate limiting."""
    for attempt in range(max_retries):
        try:
            # Add random delay to avoid overwhelming the server
            await asyncio.sleep(0.01)
            
            async with session.get(url, headers=HEADERS, timeout=TIMEOUT) as response:
                if response.status == 200:
                    return await response.text()
                elif response.status == 429:  # Rate limited
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Rate limited on {url}, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                    await asyncio.sleep(0.01)
                    continue
                else:
                    print(f"HTTP {response.status} for {url}")
                    if attempt == max_retries - 1:
                        return None
        except asyncio.TimeoutError:
            print(f"Timeout on {url}, attempt {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                await asyncio.sleep(0.01)
        except Exception as e:
            print(f"Error fetching {url}: {e}, attempt {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                await asyncio.sleep(0.01)
    
    return None

async def get_showcase_links(session, page_number):
    """Get showcase links from a specific page."""
    url = SHOWCASE_URL_TEMPLATE.format(page_number)
    print(f"Fetching showcase listing page {page_number}")
    
    html = await fetch_with_retry(session, url)
    if not html:
        return []
    
    soup = BeautifulSoup(html, "html.parser")
    links = []
    
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if re.match(r"^/showcase/[^/]+$", href):
            links.append(BASE_URL + href)
    
    unique_links = list(set(links))
    print(f"Found {len(unique_links)} unique links on page {page_number}")
    return unique_links

async def parse_showcase_page(session, url):
    """Parse a single showcase page."""
    html = await fetch_with_retry(session, url)
    if not html:
        print(f"Failed to fetch {url}")
        return None
    
    soup = BeautifulSoup(html, "html.parser")
    
    try:
        # Extract title
        title_el = soup.find("h1", class_=re.compile(r"text-4xl"))
        title = title_el.get_text(strip=True) if title_el else ""
        
        # Extract short description
        short_desc_el = title_el.find_next_sibling("p") if title_el else None
        short_desc = short_desc_el.get_text(strip=True) if short_desc_el else ""
        
        # Extract created at
        created_at_div = soup.find("h3", string=re.compile("Created At", re.I))
        created_at = ""
        if created_at_div:
            next_div = created_at_div.find_next("div")
            created_at = next_div.get_text(strip=True) if next_div else ""
        
        # Extract project description
        project_desc = ""
        proj_desc_h3 = soup.find("h3", string=re.compile("Project Description", re.I))
        if proj_desc_h3:
            desc_div = proj_desc_h3.find_next_sibling("div")
            project_desc = desc_div.get_text(" ", strip=True) if desc_div else ""
        
        # Extract how it's made
        how_made = ""
        how_made_h3 = soup.find("h3", string=re.compile("How.*Made", re.I))
        if how_made_h3:
            how_div = how_made_h3.find_next_sibling("div")
            how_made = how_div.get_text(" ", strip=True) if how_div else ""
        
        return {
            "url": url,
            "project_title": title,
            "short_description": short_desc,
            "created_at": created_at,
            "project_description": project_desc,
            "how_its_made": how_made
        }
    
    except Exception as e:
        print(f"Error parsing {url}: {e}")
        return None

async def process_in_batches(session, urls, batch_size=MAX_CONCURRENT_REQUESTS):
    """Process URLs in batches to avoid overwhelming the server."""
    all_results = []
    
    for i in range(0, len(urls), batch_size):
        batch = urls[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}: {len(batch)} URLs")
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(batch_size)
        
        async def process_url(url):
            async with semaphore:
                return await parse_showcase_page(session, url)
        
        tasks = [process_url(url) for url in batch]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and None results
        valid_results = [r for r in batch_results if r is not None and not isinstance(r, Exception)]
        all_results.extend(valid_results)
        
        print(f"Batch completed: {len(valid_results)} successful, {len(batch_results) - len(valid_results)} failed")
        
        # Small delay between batches
        if i + batch_size < len(urls):
            await asyncio.sleep(0.01)
    
    return all_results

async def save_progress(data, filename="ethglobal_showcase_projects.json"):
    """Save progress to file."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} projects to {filename}")

async def main():
    # Create session with better configuration
    connector = aiohttp.TCPConnector(
        limit=MAX_CONCURRENT_REQUESTS,
        limit_per_host=MAX_CONCURRENT_REQUESTS,
        ttl_dns_cache=300,
        use_dns_cache=True,
    )
    
    timeout = aiohttp.ClientTimeout(total=TIMEOUT)
    
    async with aiohttp.ClientSession(
        connector=connector, 
        timeout=timeout,
        headers=HEADERS
    ) as session:
        
        # 1. Find all showcase links across pages
        print("=== Collecting showcase links ===")
        page_number = 1
        all_links = []
        consecutive_empty_pages = 0
        
        while consecutive_empty_pages < 3:  # Stop after 3 consecutive empty pages
            links = await get_showcase_links(session, page_number)
            
            if not links:
                consecutive_empty_pages += 1
                print(f"No links found on page {page_number} (consecutive empty: {consecutive_empty_pages})")
            else:
                consecutive_empty_pages = 0
                all_links.extend(links)
            
            page_number += 1
            
            # Add a small delay between page requests
            await asyncio.sleep(0.01)
        
        # Remove duplicates
        all_links = list(set(all_links))
        print(f"Found total {len(all_links)} unique showcase pages.")
        
        if not all_links:
            print("No showcase links found. Exiting.")
            return
        
        # 2. Fetch all showcase project pages in batches
        print("=== Fetching project details ===")
        all_data = await process_in_batches(session, all_links)
        
        # 3. Save to JSON
        await save_progress(all_data)
        
        print(f"Scraping completed! Successfully scraped {len(all_data)}/{len(all_links)} projects.")

if __name__ == "__main__":
    asyncio.run(main())