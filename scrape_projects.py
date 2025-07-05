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

# Optimized Configuration
MAX_CONCURRENT_REQUESTS = 100  # Increased for processing
REQUEST_DELAY = 0  # No delay for maximum speed
MAX_RETRIES = 2  # Reduced retries for speed
TIMEOUT = 15  # Reduced timeout
BATCH_SIZE = 200  # Much larger batches for processing
DISCOVERY_BATCH_SIZE = 30  # Separate batch size for discovery

async def fetch_with_retry(session, url, max_retries=MAX_RETRIES):
    """Fetch URL with retry logic - optimized for speed."""
    for attempt in range(max_retries):
        try:
            # Only add delay if we have REQUEST_DELAY configured
            if REQUEST_DELAY > 0:
                await asyncio.sleep(REQUEST_DELAY)
            
            async with session.get(url, headers=HEADERS) as response:
                if response.status == 200:
                    return await response.text()
                elif response.status == 429:  # Rate limited
                    wait_time = 1 + attempt  # Linear backoff for speed
                    print(f"Rate limited on {url}, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    # Don't retry on other HTTP errors for speed
                    return None
        except asyncio.TimeoutError:
            if attempt < max_retries - 1:
                await asyncio.sleep(0.1)  # Very short retry delay
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(0.1)  # Very short retry delay
    
    return None

async def get_showcase_links_with_thumbnails(session, page_number):
    """Get showcase links and their thumbnails from a specific page."""
    url = SHOWCASE_URL_TEMPLATE.format(page_number)
    print(f"Fetching showcase listing page {page_number}")
    
    html = await fetch_with_retry(session, url)
    if not html:
        return []
    
    soup = BeautifulSoup(html, "html.parser")
    links_with_thumbnails = []
    
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if re.match(r"^/showcase/[^/]+$", href):
            # Find thumbnail image within the link
            thumbnail = ""
            img = a.find("img")
            if img and img.get("src"):
                thumbnail = img["src"]
                # Handle relative URLs
                if thumbnail.startswith("/"):
                    thumbnail = BASE_URL + thumbnail
            
            links_with_thumbnails.append({
                "url": BASE_URL + href,
                "thumbnail": thumbnail
            })
    
    # Remove duplicates based on URL
    seen_urls = set()
    unique_links = []
    for item in links_with_thumbnails:
        if item["url"] not in seen_urls:
            seen_urls.add(item["url"])
            unique_links.append(item)
    
    print(f"Found {len(unique_links)} unique links on page {page_number}")
    return unique_links

async def discover_all_pages_parallel(session, max_pages=999999, batch_size=DISCOVERY_BATCH_SIZE):
    """Discover all pages in parallel batches for maximum speed."""
    print(f"=== Discovering pages in parallel (checking up to {max_pages} pages) ===")
    
    all_links_with_thumbnails = []
    
    # Process pages in batches
    for start_page in range(1, max_pages + 1, batch_size):
        end_page = min(start_page + batch_size - 1, max_pages)
        pages_in_batch = list(range(start_page, end_page + 1))
        
        print(f"Checking pages {start_page}-{end_page} in parallel...")
        
        # Fetch all pages in this batch concurrently
        tasks = [get_showcase_links_with_thumbnails(session, page) for page in pages_in_batch]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        empty_pages_in_batch = 0
        for i, result in enumerate(batch_results):
            if isinstance(result, list) and result:
                all_links_with_thumbnails.extend(result)
            else:
                empty_pages_in_batch += 1
        
        print(f"Batch {start_page}-{end_page}: Found links on {len(batch_results) - empty_pages_in_batch} pages")
        
        # If entire batch is empty, we've likely reached the end
        if empty_pages_in_batch == len(batch_results):
            print(f"No more pages found after page {start_page - 1}")
            break
    
    return all_links_with_thumbnails

async def parse_showcase_page(session, url_data):
    """Parse a single showcase page - optimized for speed."""
    url = url_data["url"] if isinstance(url_data, dict) else url_data
    thumbnail = url_data.get("thumbnail", "") if isinstance(url_data, dict) else ""
    
    html = await fetch_with_retry(session, url)
    if not html:
        return None
    
    try:
        soup = BeautifulSoup(html, "html.parser")
        
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
            "how_its_made": how_made,
            "thumbnail": thumbnail
        }
    
    except Exception as e:
        print(f"Error parsing {url}: {e}")
        return None

async def process_in_batches_optimized(session, url_data_list, batch_size=BATCH_SIZE):
    """Process URLs in batches - heavily optimized for speed."""
    all_results = []
    total_batches = (len(url_data_list) + batch_size - 1) // batch_size
    
    print(f"Processing {len(url_data_list)} URLs in {total_batches} batches of {batch_size}")
    
    for i in range(0, len(url_data_list), batch_size):
        batch = url_data_list[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        print(f"Processing batch {batch_num}/{total_batches}: {len(batch)} URLs")
        start_time = time.time()
        
        # Process entire batch concurrently with no delays
        tasks = [parse_showcase_page(session, url_data) for url_data in batch]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and None results
        valid_results = [r for r in batch_results if r is not None and not isinstance(r, Exception)]
        all_results.extend(valid_results)
        
        elapsed = time.time() - start_time
        rate = len(batch) / elapsed if elapsed > 0 else 0
        
        print(f"Batch {batch_num} completed in {elapsed:.2f}s: {len(valid_results)} successful, {len(batch_results) - len(valid_results)} failed (Rate: {rate:.1f} pages/sec)")
        
        # No delay between batches for maximum speed
    
    return all_results

async def save_progress(data, filename="ethglobal_showcase_projects.json"):
    """Save progress to file."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} projects to {filename}")

async def main():
    total_start_time = time.time()
    
    # Create session with aggressive configuration for maximum speed
    connector = aiohttp.TCPConnector(
        limit=300,  # Much higher connection limit
        limit_per_host=150,  # Higher per-host limit
        ttl_dns_cache=300,
        use_dns_cache=True,
        enable_cleanup_closed=True,
        keepalive_timeout=30,  # Keep connections alive longer
        force_close=False,  # Don't force close connections
    )
    
    timeout = aiohttp.ClientTimeout(total=TIMEOUT, connect=5)
    
    async with aiohttp.ClientSession(
        connector=connector, 
        timeout=timeout,
        headers=HEADERS
    ) as session:
        
        # 1. Find all showcase links across pages IN PARALLEL
        discovery_start = time.time()
        all_links_with_thumbnails = await discover_all_pages_parallel(session, max_pages=999999, batch_size=DISCOVERY_BATCH_SIZE)
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_links_with_thumbnails = []
        for item in all_links_with_thumbnails:
            if item["url"] not in seen_urls:
                seen_urls.add(item["url"])
                unique_links_with_thumbnails.append(item)
        
        discovery_time = time.time() - discovery_start
        print(f"Discovery completed in {discovery_time:.2f}s: Found {len(unique_links_with_thumbnails)} unique showcase pages.")
        
        if not unique_links_with_thumbnails:
            print("No showcase links found. Exiting.")
            return
        
        # 2. Fetch all showcase project pages in optimized batches
        print("=== Fetching project details (OPTIMIZED) ===")
        processing_start = time.time()
        all_data = await process_in_batches_optimized(session, unique_links_with_thumbnails)
        processing_time = time.time() - processing_start
        
        # 3. Save to JSON
        await save_progress(all_data)
        
        total_time = time.time() - total_start_time
        success_rate = len(all_data) / len(unique_links_with_thumbnails) * 100
        
        print(f"\n=== SCRAPING COMPLETED ===")
        print(f"Total time: {total_time:.2f}s")
        print(f"Discovery time: {discovery_time:.2f}s")
        print(f"Processing time: {processing_time:.2f}s")
        print(f"Successfully scraped: {len(all_data)}/{len(unique_links_with_thumbnails)} projects ({success_rate:.1f}%)")
        print(f"Average processing rate: {len(all_data)/processing_time:.1f} pages/sec")

if __name__ == "__main__":
    asyncio.run(main())