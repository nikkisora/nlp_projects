

import re
from pathlib import Path

def get_dataset(files_dir, chunk_size=100):
    files = Path(files_dir).iterdir()
    corp = ''

    for f in files:
        with open(f) as file:
            corp += file.read().lower()

    corp = re.sub(r'[^A-Za-z\- ]', '', corp).split()

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    return list(chunks(corp, chunk_size)), set(corp)