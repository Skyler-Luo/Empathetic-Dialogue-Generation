import csv
import os

# 设置数据集缓存路径
DATASETS_CACHE = os.environ.get('DATASETS_CACHE', os.path.join(os.path.dirname(__file__), '.cache', 'datasets'))

os.makedirs(DATASETS_CACHE, exist_ok=True)
os.environ['HF_DATASETS_CACHE'] = DATASETS_CACHE

import datasets


_CITATION = """\
@inproceedings{rashkin2019towards,
  title = {Towards Empathetic Open-domain Conversation Models: a New Benchmark and Dataset},
  author = {Hannah Rashkin and Eric Michael Smith and Margaret Li and Y-Lan Boureau},
  booktitle = {ACL},
  year = {2019},
}
"""

_DESCRIPTION = """\
PyTorch original implementation of Towards Empathetic Open-domain Conversation Models: a New Benchmark and Dataset
"""
_URL = "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz"


class EmpatheticDialogues(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.1.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "conv_id": datasets.Value("string"),
                    "utterance_idx": datasets.Value("int32"),
                    "context": datasets.Value("string"),
                    "prompt": datasets.Value("string"),
                    "speaker_idx": datasets.Value("int32"),
                    "utterance": datasets.Value("string"),
                    "selfeval": datasets.Value("string"),
                    "tags": datasets.Value("string")
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/facebookresearch/EmpatheticDialogues",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        archive = dl_manager.download(_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"files": dl_manager.iter_archive(archive), "split_file": "empatheticdialogues/train.csv"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"files": dl_manager.iter_archive(archive), "split_file": "empatheticdialogues/valid.csv"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"files": dl_manager.iter_archive(archive), "split_file": "empatheticdialogues/test.csv"},
            ),
        ]

    def _generate_examples(self, files, split_file):
        """Yields examples."""
        for path, f in files:
            if split_file == path:
                data = csv.DictReader(line.decode("utf-8") for line in f)
                for id_, row in enumerate(data):
                    utterance = row["utterance"]
                    speaker_id = int(row["speaker_idx"])
                    context = row["context"]
                    conv_id = row["conv_id"]
                    tags = row["tags"] if row["tags"] else ""
                    selfeval = row["selfeval"] if row["selfeval"] else ""
                    utterance_id = int(row["utterance_idx"])
                    prompt = row["prompt"]
                    yield id_, {
                        "utterance": utterance,
                        "utterance_idx": utterance_id,
                        "context": context,
                        "speaker_idx": speaker_id,
                        "conv_id": conv_id,
                        "selfeval": selfeval,
                        "prompt": prompt,
                        "tags": tags,
                    }
                break


if __name__ == "__main__":
    builder = EmpatheticDialogues()
    builder.download_and_prepare()
    ds = builder.as_dataset()
    for split in ["train", "validation", "test"]:
        print(f"{split} examples: {len(ds[split])}")
