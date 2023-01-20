#!/usr/bin/env python

from chemspipy import ChemSpider
import argparse
import json
import os
import csv

try:
    cs = ChemSpider(os.environ["CHEMSPIDER_API_KEY"])
except KeyError:
    print("add CHEMSPIDER_API_KEY as an environment variable")
    quit()

def get_compound_info_batch(record_ids):
    results = cs.get_details_batch(record_ids)
    return results

def main(records_csv, out_fname):
    # read in list of compound names
    cmpds = {}
    with open(records_csv, newline="") as f:
        reader = csv.reader(f)
        for line in reader:
            name, record_id = line
            cmpds[int(record_id)] = name
    
    record_ids = cmpds.keys()

    with open(out_fname, "w", newline="") as fout:
        writer = csv.writer(fout)

        for batch_idx in range(0, len(record_ids), 100):
            print(f"working on records {batch_idx}-{batch_idx+100}")
            batch_ids = record_ids[batch_idx:batch_idx + 100]
            details = get_compound_info_batch(batch_ids)

            for det in details:
                det_json = json.dumps(det)
                writer.writerow([cmpds[det.id], det.id, det_json])
    
    print("[i] done")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("records_csv")
    parser.add_argument("outfile")

    args = parser.parse_args()

    main(args.records_csv, args.outfile)