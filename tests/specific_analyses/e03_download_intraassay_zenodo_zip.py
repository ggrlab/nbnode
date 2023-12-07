# From https://github.com/zenodo/zenodo/issues/1888
import io
import zipfile

import requests

# ACCESS_TOKEN = "ZENODO_ACCESS_TOKEN"  # is public now
record_id = "7890571"

r = requests.get(
    f"https://zenodo.org/api/records/{record_id}",
    # params={"access_token": ACCESS_TOKEN}
)
# print(r.json())
download_urls = [f["links"]["self"] for f in r.json()["files"]]
filenames = [f["key"] for f in r.json()["files"]]
# outdir = "example_data/asinh.align_manual.CD3_Gate"
# os.makedirs(outdir, exist_ok=True)

# for filename, url in zip(filenames, download_urls):
#     print("Downloading:", filename)
#     r = requests.get(url, params={"access_token": ACCESS_TOKEN})
#     with open(os.path.join(outdir, filename), "wb") as f:
#         f.write(r.content)

for single_json_file in r.json()["files"]:
    if single_json_file["key"].endswith(".zip"):
        my_zipfile = {
            "filename": single_json_file["key"],
            "url": single_json_file["links"]["self"],
            "checksum": single_json_file["checksum"],
        }
    # print(single_json_file)

print("Downloading:", my_zipfile["filename"])
r = requests.get(
    my_zipfile["url"],
    # params={"access_token": ACCESS_TOKEN}
)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall("example_data")
