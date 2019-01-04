# coding:utf_8
import re

# """
#   通过与黄金标准文件对比分析中文分词效果.

#   使用方法：
#           python crf_tag_score.py test_gold.utf8 your_tagger_output.utf8

#   分析结果示例如下:
#     标准词数：104372 个，正确词数：96211 个，错误词数：6037 个
#     标准行数：1944，正确行数：589 ，错误行数：1355
#     Recall: 92.1808531024%
#     Precision: 94.0957280338%
#     F MEASURE: 93.1284483593%


#   参考：中文分词器分词效果的评测方法
#   http://ju.outofmemory.cn/entry/46140

# """


def read_line(f):
    '''
        读取一行，并清洗空格和换行
    '''
    line = f.readline()
    return line.strip()


def prf_score(real_text_file, pred_text_file, prf_file, epoch):
    file_gold = open(real_text_file, 'r', encoding='utf8')
    # file_gold = codecs.open(r'../corpus/msr_test_gold.utf8', 'r', 'utf8')
    # file_tag = codecs.open(r'pred_standard.txt', 'r', 'utf8')
    file_tag = open(pred_text_file, 'r', encoding='utf8')

    line1 = read_line(file_gold)
    N_count = 0  # 将正类分为正或者将正类分为负
    e_count = 0  # 将负类分为正
    c_count = 0  # 正类分为正
    e_line_count = 0
    c_line_count = 0

    while line1:
        line2 = read_line(file_tag)

        list1 = re.split('\s+', line1.strip())
        list2 = re.split('\s+', line2.strip())

        count1 = len(list1)  # 标准分词数
        N_count += count1
        if line1 == line2:
            c_line_count += 1  # 分对的行数
            c_count += count1  # 分对的词数
        else:
            e_line_count += 1
            count2 = len(list2)

            arr1 = []
            arr2 = []

            pos = 0
            for w in list1:
                arr1.append(tuple([pos, pos + len(w)]))  # list1中各个单词的起始位置
                pos += len(w)

            pos = 0
            for w in list2:
                arr2.append(tuple([pos, pos + len(w)]))  # list2中各个单词的起始位置
                pos += len(w)

            for tp in arr2:
                if tp in arr1:
                    c_count += 1
                else:
                    e_count += 1

        line1 = read_line(file_gold)

    R = float(c_count) / N_count
    P = float(c_count) / (c_count + e_count)
    F = 2. * P * R / (P + R)
    ER = 1. * e_count / N_count

    print("result:")
    print('标准词数：%d个，词数正确率：%f个，词数错误率：%f \n' % (N_count, c_count / N_count, e_count / N_count))
    print('标准行数：%d，行数正确率：%f，行数错误率：%f \n' % (c_line_count + e_line_count, c_line_count / (c_line_count + e_line_count),
                                            e_line_count / (c_line_count + e_line_count)))
    print('Recall: %f' % (R))
    print('Precision: %f' % (P))
    print('F MEASURE: %f' % (F))
    print('ERR RATE: %f' % (ER))

    # print P,R,F

    f = open(prf_file, 'a', encoding='utf-8')
    f.write('result-(epoch:%s):\n' % epoch)
    f.write('标准词数：%d，词数正确率：%f，词数错误率：%f \n' % (N_count, c_count / N_count, e_count / N_count))
    f.write('标准行数：%d，行数正确率：%f，行数错误率：%f \n' % (c_line_count + e_line_count, c_line_count / (c_line_count + e_line_count),
                                              e_line_count / (c_line_count + e_line_count)))
    f.write('Recall: %f\n' % (R))
    f.write('Precision: %f\n' % (P))
    f.write('F MEASURE: %f\n' % (F))
    f.write('ERR RATE: %f\n' % (ER))
    f.write('====================================\n')

    return F


if __name__ == '__main__':
    prf_score('../data/gold.utf8', '../data/pred_text.utf8', '../data/prf_tmp.txt', 35)
