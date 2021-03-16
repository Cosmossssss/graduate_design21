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
    """测试 utils.dataset.processor.py """
    def test_generate_ptb_vocab(self):
        try:
            generate_ptb_vocab("D:\\Document\\Python\\pos_tagging\\ptbdataset\\ptb.train.txt", "D:\\Document\\Python\\pos_tagging\\ptbdataset_process\\ptb.vocab")
            result=True
        except Exception as e:
            result=False
        self.assertTrue(result)
