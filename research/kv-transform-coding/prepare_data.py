"""Download GloVe 300-d (gensim-data GitHub release) and save the first 120k vectors as glove120k.npy."""
import gzip, os, urllib.request
import numpy as np

URL = "https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-300/glove-wiki-gigaword-300.gz"
if not os.path.exists("glove300.gz"):
    urllib.request.urlretrieve(URL, "glove300.gz")  # ~390 MB
rows = []
with gzip.open("glove300.gz", "rt") as f:
    next(f)  # header: "400000 300"
    for i, line in enumerate(f):
        if i >= 120000:
            break
        rows.append(np.asarray(line.rstrip().split(" ")[1:], dtype=np.float32))
np.save("glove120k.npy", np.stack(rows))
