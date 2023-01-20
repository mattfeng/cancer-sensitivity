#!/usr/bin/env python

from chemspipy import ChemSpider
import argparse
import os
import csv

try:
    cs = ChemSpider(os.environ["CHEMSPIDER_API_KEY"])
except KeyError:
    print("add CHEMSPIDER_API_KEY as an environment variable")
    quit()

def get_chemspider_record_id(query):
    results = cs.search(query)

    if len(results) == 0:
        return None
    
    return results[0]

def main(compound_list_fname, out_fname):
    # read in list of compound names
    cmpds = []
    with open(compound_list_fname) as f:
        for line in f:
            line = line.strip()
            cmpds.append(line)
    
    with open(out_fname, "w", newline="") as fout:
        writer = csv.writer(fout)

        for cmpd in cmpds:
            print(f"searching for: {cmpd}")
            record = get_chemspider_record_id(cmpd)
            record_id = record.record_id
            writer.writerow([cmpd, record_id])
    
    print("[i] done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("compound_list")
    parser.add_argument("outfile")

    args = parser.parse_args()

    main(args.compound_list, args.outfile)
    main()