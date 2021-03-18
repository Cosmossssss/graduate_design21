"""
-*- coding: utf-8 -*-
@Name   : pos_tagging-dataset_process.py
@Time   : 2021/3/16 10:59
@Author : 软工1701 李澳 U201716958
@Desc   : 测试数据集预处理的相关函数
"""

import unittest
from utils.dataset.processor import *


class PTBTestCase(unittest.TestCase):
    """测试 utils.dataset.processor.py 中的相关函数"""

    # Build Time: 2021/3/16
    def test_generate_ptb_vocab(self):
        project_root_dir = ".\\..\\"
        # noinspection PyBroadException
        try:
            generate_ptb_vocab(project_root_dir + "ptbdataset\\ptb.train.txt",
                               project_root_dir + "ptbdataset_process\\ptb.vocab")
            result = True
        except Exception:
            result = False
        self.assertTrue(result)

    # Build Time: 2021/3/18
    def test_vocab_transform_index(self):
        project_root_dir = ".\\..\\"
        # noinspection PyBroadException
        try:
            vocab_transform_index(project_root_dir + "ptbdataset_process\\ptb.vocab",
                                  project_root_dir + "ptbdataset\\ptb.train.txt",
                                  project_root_dir + "ptbdataset\\ptb.test.txt",
                                  project_root_dir + "ptbdataset\\ptb.valid.txt",
                                  project_root_dir + "ptbdataset_process\\ptb.train",
                                  project_root_dir + "ptbdataset_process\\ptb.test",
                                  project_root_dir + "ptbdataset_process\\ptb.valid")
            result = True
        except Exception:
            result = False
        self.assertTrue(result)
