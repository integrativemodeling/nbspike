import argparse
import pandas as pd
from Bio import SeqIO

_FASTA_FILE = "../data/spike_nanobody.fasta.txt"
_CDR_FILE = "../data/imgt_cdr.csv"

def _get_boundary(seq, subseq):
    start = seq.find(subseq) + 1
    stop = start + len(subseq) - 1
    return (start, stop)

# user input
parser = argparse.ArgumentParser(
    description="CDR boundaries from CDR subsequences")

parser.add_argument("nanobody_name", help="Name of nanobody")

args = parser.parse_args()
nb = args.nanobody_name

# parse all nb sequences
records = {}
for r in SeqIO.parse(_FASTA_FILE, format="fasta"):
    if "delta" in r.id or "omicron" in r.id: continue
    records[r.id] = r.seq._data.decode("utf-8")

assert nb in records

# parse CDR sequences
df = pd.read_csv(_CDR_FILE, comment="#", index_col=0)
nb_df = df.loc[nb]
for k in ["cdr1", "cdr2", "cdr3"]:
    print(k, _get_boundary(seq=records[nb], subseq=nb_df[k]))
    
