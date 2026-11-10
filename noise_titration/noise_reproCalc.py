#!/usr/bin/env python3
"""Compatibility entry point for the current v2.4/T1w noise analysis.

Paths, model choice, failure accounting, and plotting now live together in
noise_hippunfold.py. Both clean and degraded outputs must come from that run.
"""
from noise_hippunfold import plot_results


if __name__ == '__main__':
    plot_results()
