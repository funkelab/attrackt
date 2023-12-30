from typing import List
from attrackt.scripts.load_csv_ilp_result import load_csv_ilp_result


def identify_valid_sequences(sequences: List[str], supervised_csv: str):
    valid_sequences = []
    for sequence in sequences:
        if len(load_csv_ilp_result(supervised_csv, [sequence])) > 0:
            valid_sequences.append(sequence)
    return valid_sequences
