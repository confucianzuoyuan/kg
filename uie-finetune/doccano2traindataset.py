import os
import time
import argparse
import json
import numpy as np

from utils import set_seed, convert_ext_examples


def do_convert():
    set_seed(1000)

    splits = [0.8, 0.1, 0.1]

    tic_time = time.time()
    if not os.path.exists(args.doccano_file):
        raise ValueError("Please input the correct path of doccano file.")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    with open(args.doccano_file, "r", encoding="utf-8") as f:
        raw_examples = f.readlines()

    def _create_ext_examples(examples,
                             negative_ratio,
                             shuffle=False,
                             is_train=True):
        entities, relations = convert_ext_examples(
            examples, negative_ratio, is_train)
        examples = entities + relations
        if shuffle:
            indexes = np.random.permutation(len(examples))
            examples = [examples[i] for i in indexes]
        return examples

    def _save_examples(save_dir, file_name, examples):
        count = 0
        save_path = os.path.join(save_dir, file_name)
        if not examples:
            print("Skip saving %d examples to %s." % (0, save_path))
            return
        with open(save_path, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                count += 1
        print("Save %d examples to %s." % (count, save_path))

    indexes = np.random.permutation(len(raw_examples))
    index_list = indexes.tolist()
    raw_examples = [raw_examples[i] for i in indexes]

    i1, i2, _ = splits
    p1 = int(len(raw_examples) * i1)
    p2 = int(len(raw_examples) * (i1 + i2))

    train_ids = index_list[:p1]
    dev_ids = index_list[p1:p2]
    test_ids = index_list[p2:]

    with open(os.path.join(args.save_dir, "sample_index.json"), "w") as fp:
        maps = {
            "train_ids": train_ids,
            "dev_ids": dev_ids,
            "test_ids": test_ids
        }
        fp.write(json.dumps(maps))

    if args.task_type == "ext":
        train_examples = _create_ext_examples(raw_examples[:p1],
                                              args.negative_ratio,
                                              args.is_shuffle)
        dev_examples = _create_ext_examples(raw_examples[p1:p2],
                                            -1,
                                            is_train=False)
        test_examples = _create_ext_examples(raw_examples[p2:],
                                             -1,
                                             is_train=False)

    _save_examples(args.save_dir, "train.txt", train_examples)
    _save_examples(args.save_dir, "dev.txt", dev_examples)
    _save_examples(args.save_dir, "test.txt", test_examples)

    print('Finished! It takes %.2f seconds' % (time.time() - tic_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--doccano_file", default="./data/doccano.json",
                        type=str, help="The doccano file exported from doccano platform.")
    parser.add_argument("-s", "--save_dir", default="./data",
                        type=str, help="The path of data that you wanna save.")
    parser.add_argument("--task_type", choices=['ext', 'cls'], default="ext", type=str,
                        help="Select task type, ext for the extraction task and cls for the classification task, defaults to ext.")

    args = parser.parse_args()

    args.negative_ratio = 5
    args.prompt_prefix = '情感倾向'
    args.options = ['正向', '负向']
    args.separator = '##'
    args.is_shuffle = True

    do_convert()
