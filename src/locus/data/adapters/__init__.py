"""Converters from a third party's corpus format into the JSONL contract.

Each module here is a one-way adapter: it reads whatever files its upstream
source ships and writes `locus.data.ingest`'s contracted layout, so the
harness never needs to know a second input format exists.
"""
