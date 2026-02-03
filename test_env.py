from dotenv import load_dotenv
import os

load_dotenv()

print("=== Environment Variables ===")
print(f"SERPAPI_KEY: {os.getenv('SERPAPI_KEY')}")
print(f"IMGBB_API_KEY: {os.getenv('IMGBB_API_KEY')}")
print(f"SERPAPI exists: {bool(os.getenv('SERPAPI_KEY'))}")
print(f"Length: {len(os.getenv('SERPAPI_KEY', ''))}")
