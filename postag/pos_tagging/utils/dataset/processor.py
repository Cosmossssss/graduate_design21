"""
-*- coding: utf-8 -*-
@Name   : pos_tagging-processor.py
@Time   : 2021/3/16 0:46
@Author : 软工1701 李澳 U201716958
@Desc   : 预处理如 PTB，CTB，人民日报语料库 等数据集的函数工具包，目前先以处理 PTB 数据集的函数集合作为代表
"""


# 生成词频降序排列的词汇表文件
def generate_ptb_vocab(src_train_data, vocab_file):
    """

    Args:
        src_train_data: ptb训练集数据作为输入
        vocab_file: 输出的词汇表文件

    Returns:
        null

    References:
        https://blog.csdn.net/qq_23031939/article/details/79759344
        https://chehongshu.blog.csdn.net/article/details/85288590?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-4.control&dist_request_id=&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-4.control

    """

    import codecs
    import collections
    from operator import itemgetter

    # 统计单词出现的频率
    counter = collections.Counter()
    with codecs.open(src_train_data, "r", "utf-8") as f:
        for line in f:
            for word in line.strip().split():
                counter[word] += 1

    # 按照词频顺序对单词进行排序
    # Counter 集成于 dict 类，因此也可以使用字典的方法，此类返回一个以元素为 key 、元素个数为 value 的 Counter 对象集合
    # 依据key排序　itermgetter(1)为降序
    sorted_word_to_cnt = sorted(counter.items(), key=itemgetter(1), reverse=True)

    # 转换成单词string的list
    sorted_words_list = [x[0] for x in sorted_word_to_cnt]

    # 因为稍后需要在文本换行处加入句子结束符"<eos>"，这里预先将其加入词汇表
    # 而因为在 ptb 数据集中，输入数据已经将低频词汇替换为了<unk>，可以作为一个“单词”在以上步骤中被一并处理，
    # 所以不需要另外在单词的list中加入低频词汇/<unk>的项
    sorted_words_list = ["<eos>"] + sorted_words_list

    # 将降序后的单词序列写入到 vocab_file的词汇表文件中
    with codecs.open(vocab_file, 'w', 'utf-8') as file_output:
        for word in sorted_words_list:
            file_output.write(word + '\n')
