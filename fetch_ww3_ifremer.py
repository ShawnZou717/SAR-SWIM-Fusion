import os
import argparse
import requests
import re
from urllib.parse import urljoin
from bs4 import BeautifulSoup

# Strictly using your provided base URL
BASE_URL = "https://data-cersat.ifremer.fr/projects/iwwoc/colocations/swim/model/ww3/swi_l2____/"

# Compile the regex pattern for the exact filename format you specified
# \d{2} matches XX (2 digits), \d{8} matches yyyymmdd (8 digits), \d{6} matches HHMMSS (6 digits)
FILE_PATTERN = re.compile(r"^CFO_OP\d{2}_SWI_L2_____F_\d{8}T\d{6}_\d{8}T\d{6}\.nc$")

def get_directory_links(url):
    """Scrape the HTTP index page for directory or file links."""
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error accessing {url}: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    links = []
    
    for a_tag in soup.find_all('a'):
        href = a_tag.get('href')
        # Filter out navigation links like 'Parent Directory' or query params
        if href and not href.startswith('?') and not href.startswith('/'):
            links.append(href)
            
    return links

def download_file(url, output_path):
    """Download a file in chunks."""
    if os.path.exists(output_path):
        print(f"      -> File already exists, skipping: {output_path}")
        return

    print(f"      -> Downloading {url} ...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(output_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"      -> Saved to {output_path}")
    except requests.exceptions.RequestException as e:
        print(f"      -> Failed to download {url}: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)

def main():
    parser = argparse.ArgumentParser(description="Download WW3 NetCDF data from IFREMER server.")
    parser.add_argument("-y", "--years", required=True, type=str, help="Comma-separated list of years (e.g., 2019,2020)")
    parser.add_argument("-o", "--outdir", required=True, help="Output directory to save the files")
    
    args = parser.parse_args()
    years_list = [y.strip() for y in args.years.split(',')]
    os.makedirs(args.outdir, exist_ok=True)
    
    print(f"Scanning base URL for version folders: {BASE_URL}")
    
    # 1. Get {version_name} folders
    base_links = get_directory_links(BASE_URL)
    version_dirs = [v for v in base_links if v.endswith('/')]
    
    if not version_dirs:
        print("No version directories found at the base URL.")
        return

    for version in version_dirs:
        version_url = urljoin(BASE_URL, version)
        clean_version = version.strip('/')
        print(f"\n=== Entering Version: {clean_version} ===")
        
        # 2. Iterate over requested {year}s
        for year in years_list:
            year_url = urljoin(version_url, f"{year}/")
            
            # 3. Get {seq_num} folders
            year_links = get_directory_links(year_url)
            seq_dirs = [d for d in year_links if d.endswith('/')]
            
            if not seq_dirs:
                print(f"  No sequential directories found for year {year} in {clean_version}.")
                continue
                
            print(f"  Found {len(seq_dirs)} sequential folders for year {year}.")
            
            for seq_dir in seq_dirs:
                seq_url = urljoin(year_url, seq_dir)
                clean_seq = seq_dir.strip('/')
                
                # 4. Get files in {seq_num} folder
                seq_files = get_directory_links(seq_url)
                
                # Filter strictly using the Regular Expression
                target_files = [f for f in seq_files if FILE_PATTERN.match(f)]
                
                if not target_files:
                    continue
                    
                # Mirror directory structure locally: outdir/version/year/seq_num/
                local_dir = os.path.join(args.outdir, clean_version, str(year), clean_seq)
                os.makedirs(local_dir, exist_ok=True)
                
                # Download
                for file_name in target_files:
                    file_url = urljoin(seq_url, file_name)
                    local_file_path = os.path.join(local_dir, file_name)
                    download_file(file_url, local_file_path)

if __name__ == "__main__":
    main()