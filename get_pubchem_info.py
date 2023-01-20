#!/usr/bin/env python

import argparse
import json
import pubchempy as pcp
import csv
import time

def main(pubchemids_fname, out_fname):
    cids = []
    with open(pubchemids_fname) as fin:
        for line in fin:
            line = line.strip()
            cids.append(int(line))


    with open(out_fname, "w", newline="") as fout:
        writer = csv.writer(fout)

        for cid in cids:
            print(f"searching cid: {cid}")
            success = False

            while not success:
                try:
                    cmpd = pcp.Compound.from_cid(cid)
                    cmpd_json = json.dumps(cmpd.to_dict())
                    success = True
                except pcp.PubChemHTTPError:
                    time.sleep(3)

            writer.writerow([cid, cmpd_json])

    print("[i] done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pubchemids_fname")
    parser.add_argument("out_fname")
    args = parser.parse_args()

    main(args.pubchemids_fname, args.out_fname)