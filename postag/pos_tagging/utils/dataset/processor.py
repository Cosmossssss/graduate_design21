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

    Build Time:
        2021/3/16 1:30

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

    # 因为稍后需要在文本换行处加入句子结束符"<eos>"(或者说行结束符)，这里预先将其加入词汇表
    # 而因为在 ptb 数据集中，输入数据已经将低频词汇替换为了<unk>，可以作为一个“单词”在以上步骤中被一并处理，
    # 所以不需要另外在单词的list中加入低频词汇/<unk>的项
    sorted_words_list = ["<eos>"] + sorted_words_list

    # 将降序后的单词序列写入到 vocab_file的词汇表文件中
    with codecs.open(vocab_file, 'w', 'utf-8') as file_output:
        for word in sorted_words_list:
            file_output.write(word + '\n')


# 将数据集中的训练、测试、验证的数据文件编号化
def vocab_transform_index(vocab, src_train_data, src_test_data, src_valid_data, train_index_data, test_index_data,
                          valid_index_data):
    """

    Args:
        vocab: 创建好的词汇降序表
        src_train_data: ptb 数据集中的训练数据
        src_test_data: ptb 数据集中的测试数据
        src_valid_data: ptb 数据集中的验证数据
        train_index_data: 经过编号化后的训练数据
        test_index_data: 经过编号化后的测试数据
        valid_index_data: 经过编号化后的验证数据

    Returns:
        null

    References:
        https://blog.csdn.net/qq_23031939/article/details/79759344
        https://chehongshu.blog.csdn.net/article/details/85288590?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-4.control&dist_request_id=&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-4.control

    Build Time:
        2021/3/18 0:08

    """
    import codecs
    with codecs.open(vocab, 'r', 'utf-8') as vocab_f:
        # 获取降序后的单词列表 vocab_list
        vocab_list = [single_vocab.strip() for single_vocab in vocab_f.readlines()]
    # 生成单词和行数的 dict
    word_index_dict = {k: v for (k, v) in zip(vocab_list, range(len(vocab_list)))}

    # 根据单词获取其在字典中的编号 index 的函数 get_word_index_from_dict
    def get_word_index_from_dict(word_content):
        return word_index_dict[word_content] if word_content in word_index_dict else word_index_dict['<unk>']

    # 编号化单个数据集文件的函数 src_to_index
    def src_to_index(src_data, index_data):
        # 指向原始格式数据集文件和编号化后的数据集文件指针 f_in, f_out
        f_in = codecs.open(src_data, 'r', 'utf-8')
        f_out = codecs.open(index_data, 'w', 'utf-8')
        for line in f_in:
            # 每一行/每一个句子中出现的单词集合（外加句子结束符"<eos>"）
            words = line.strip().split() + ["<eos>"]
            # 单词集合转换成对应的 index 集合
            index_line = ' '.join([str(get_word_index_from_dict(word)) for word in words]) + '\n'
            # 按行转换后写道对应的文件中
            f_out.write(index_line)
        f_in.close()
        f_out.close()

    # 将训练数据，测试数据，验证数据分别编号化
    src_to_index(src_train_data, train_index_data)
    src_to_index(src_test_data, test_index_data)
    src_to_index(src_valid_data, valid_index_data)
