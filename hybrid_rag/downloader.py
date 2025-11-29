import requests
import os

data = "data"
urls = {
    "MSC.385(94)_Polar_Code_Safety.pdf": "https://wwwcdn.imo.org/localresources/en/KnowledgeCentre/IndexofIMOResolutions/MSCResolutions/MSC.385(94).pdf",
    "MEPC.264(68)_Polar_Code_Env.pdf": "https://wwwcdn.imo.org/localresources/en/KnowledgeCentre/IndexofIMOResolutions/MEPCDocuments/MEPC.264(68).pdf"
}

if not os.path.exists(data):
    os.makedirs(data)
    print(f"Created directory: {data}")

for filename, url in urls.items():
    filepath = os.path.join(data, filename)
    print(f"Downloading {filename}...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded to {filepath}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {filename}: {e}")

print("Download process complete.")