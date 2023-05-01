
# From https://github.com/zenodo/zenodo/issues/1888
import os 
import requests

ACCESS_TOKEN = "ZENODO_ACCESS_TOKEN"
record_id = "7883353"

r = requests.get(f"https://zenodo.org/api/records/{record_id}", params={'access_token': ACCESS_TOKEN})
download_urls = [f['links']['self'] for f in r.json()['files']]
filenames = [f['key'] for f in r.json()['files']]

print(r.status_code)
print(download_urls)

os.makedirs("example_data/asinh.align_manual.CD3_Gate", exist_ok=True)
for filename, url in zip(filenames, download_urls):
    print("Downloading:", filename)
    r = requests.get(url, params={'access_token': ACCESS_TOKEN})
    with open(filename, 'wb') as f:
        f.write(r.content)