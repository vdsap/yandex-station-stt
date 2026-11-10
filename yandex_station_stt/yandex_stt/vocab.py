"""Vocabulary and BPE token handling for Yandex Station ASR."""

import os
from typing import Dict, List


class SubwordVocab:
    """Manages the 1004-token vocabulary of the Conformer RNN-T model.

    Token layout:
    - ID 0: <unk>
    - ID 1..999: 998 Russian BPE subwords from subword_dict.txt
    - ID 1000..1002: Context / special tokens
    - ID 1003: <blank> (skip token)
    """

    BLANK_ID = 1003
    UNK_ID = 0

    def __init__(self, dict_path: str) -> None:
        self.dict_path = dict_path
        self.id_to_subword: Dict[int, str] = {
            self.UNK_ID: "<unk>",
            self.BLANK_ID: "<blank>",
        }
        self.subword_to_id: Dict[str, int] = {
            "<unk>": self.UNK_ID,
            "<blank>": self.BLANK_ID,
        }

        if not os.path.exists(dict_path):
            raise FileNotFoundError(f"Subword dictionary not found at {dict_path}")

        with open(dict_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                parts = line.strip().split()
                if parts:
                    subword = parts[0]
                    token_id = idx + 1
                    self.id_to_subword[token_id] = subword
                    self.subword_to_id[subword] = token_id

    def decode(self, token_ids: List[int]) -> str:
        """Decodes an integer token sequence into clean human-readable Russian text."""
        subwords = []
        for tid in token_ids:
            if tid == self.BLANK_ID or tid >= 1000:
                continue
            subword = self.id_to_subword.get(tid, "")
            subwords.append(subword)

        raw_text = "".join(subwords)
        # ▁ is the word boundary marker in Russian BPE vocabulary
        clean_text = raw_text.replace("▁", " ").strip()
        # Collapse multiple spaces
        return " ".join(clean_text.split())

    def __len__(self) -> int:
        return len(self.id_to_subword)
