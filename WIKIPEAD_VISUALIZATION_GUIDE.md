# WIKIPEAD_ALL.PY ���ݿ��ӻ�����ʹ��ָ��

��Ŀ¼��������ר��Ϊ `wikipead_all.py` ���������ƵĿ��ӻ����ߡ�

## ���߸���

### 1. `draw_wikipead_all_chart.py` - ͼ����ƹ���
ר��Ϊ `wikipead_all.py` ������������ɸ��ֿ��ӻ�ͼ��

### 2. `generate_wikipead_analysis_report.py` - ��������������
������ϸ���ı��������棬����ͳ��ժҪ�����Ʒ������Ż�����

## ��Ҫ��������

### ͼ����ƹ���
- **�ĵ�Ƶ�ʷֲ��Ա�ͼ**: ��ʾTop 10%�ĵ��ڲ�ͬ���ݼ��ļ��ж�
- **N-gram���з���ͼ**: ��������2,3,4���ĵ��Ե�Ƶ�ʷֲ�
- **HNSW�߲�ڵ�ֲ�ͼ**: ��ʾ�����ĵ���HNSW�����еĲ㼶�ֲ�
- **��ϸ�㼶�ֲ�ͼ**: չʾTop-10�����ĵ��ľ���㼶�ֲ�
- **�ۺ϶Ա��Ǳ��**: �����з������������һ���ۺ�ͼ����

### �������湦��
- **ͳ��ժҪ**: �ؼ�ָ�����ֵͳ��
- **�ֲ���������**: ���жȱ�ֵ������ϵ������������Ե�
- **�ȽϷ���**: �����ݼ��ĶԱȶ���
- **�Ż�����**: ���ڷ��������ʵ�ý���

## ʹ�÷���

### ����ʹ��

```bash
# �����������ݼ������õ�ͼ��
python draw_wikipead_all_chart.py

# �����������ݼ������õķ�������
python generate_wikipead_analysis_report.py
```

### ָ�����ݼ�������

```bash
# ������MMLU���ݼ�����������
python draw_wikipead_all_chart.py --dataset mmlu

# �������������ݼ���top10����
python draw_wikipead_all_chart.py --topk 10

# �����ض����ݼ������õ����
python draw_wikipead_all_chart.py --dataset hotpotqa --topk 5

# �����ض����õķ�������
python generate_wikipead_analysis_report.py --dataset nq --topk 10
```

### ����˵��

- `--dataset`: ѡ�����ݼ�
  - `mmlu`: MMLU���ݼ�
  - `nq`: Natural Questions���ݼ�  
  - `hotpotqa`: HotpotQA���ݼ�
  - `triviaqa`: TriviaQA���ݼ�
  - `all`: �������ݼ� (Ĭ��)

- `--topk`: ѡ��Top-K����
  - `1`: Top-1����
  - `5`: Top-5����
  - `10`: Top-10����
  - `all`: ���п������� (Ĭ��)

## ����ļ�

### ͼ���ļ� (������ `output/charts/` Ŀ¼)
- `wikipead_frequency_distribution.png` - �ĵ�Ƶ�ʷֲ��Ա�ͼ
- `wikipead_ngram_distribution.png` - N-gram�ֲ��Ա�ͼ
- `wikipead_high_level_distribution.png` - HNSW�߲�ڵ�ֲ�ͼ
- `wikipead_level_distribution_detail.png` - ��ϸ�㼶�ֲ�ͼ
- `wikipead_comprehensive_dashboard.png` - �ۺ϶Ա��Ǳ��

### ���������ļ�
- `wikipead_analysis_report.txt` - ��ϸ���ı���������

### ���Ի����
��ָ���ض����ݼ�������ʱ������ļ����������Ӧ��׺��
- ����: `wikipead_frequency_distribution_mmlu_top10.png`
- ����: `wikipead_analysis_report_hotpotqa_top5.txt`

## ����Ҫ��

��Щ������Ҫ `wikipead_all.py` ���ɵ��������͵�ͳ���ļ���

### �����ļ�
- `freq_stats_{dataset}_top{k}.txt` - �ĵ�Ƶ��ͳ��
- `ngram_stats_n{n}_{dataset}_top{k}.txt` - N-gram����ͳ�� (n=2,3,4)
- `high_level_stats_{dataset}_top{k}.txt` - HNSW�߲�ڵ�ͳ��

### ֧�ֵ����ݼ����������
- ���ݼ�: mmlu, nq, hotpotqa, triviaqa
- Top-K: 1, 5, 10

## ͼ������

### ����֧��
- ͼ����⡢���ǩ��ͼ���Ⱦ�֧��������ʾ
- ������SimHei��Microsoft YaHei����������

### ���������
- 300 DPI�ֱ��ʣ��ʺ����ĺͱ���ʹ��
- �Զ�����ͼ���ֺʹ�С
- ֧����ֵ��ǩ��ʾ

### �������ݴ���
- �Զ������õ�ͳ���ļ�
- ��������ļ������ʽ (UTF-8, GB2312��)
- ���Ŵ���ȱʧ����

## ������������

���ɵķ���������������½ڣ�

1. **ִ��ժҪ** - �ؼ����ָ���
2. **�ĵ�Ƶ�ʷֲ�����** - ��ϸ��Ƶ��ͳ��
3. **N-gram���з���** - �����ĵ��Ե�ģʽ����
4. **HNSW������������** - �����㼶�ͽṹ����
5. **�ؼ�����** - �����ݼ��ıȽϷ���
6. **�Ż�����** - ���ڷ�����ʵ�ý���
7. **����˵��** - ָ����ͺͼ��㷽��

## �����ų�

### ��������

1. **�Ҳ���ͳ���ļ�**
   - ȷ�������� `wikipead_all.py` ����ͳ���ļ�
   - ����ļ�����ʽ�Ƿ���ȷ

2. **ͼ����ʾ�쳣**
   - ����Ƿ�װ����������
   - ȷ��matplotlib�汾����

3. **�������**
   - ���߻��Զ����Զ��ֱ����ʽ
   - ���������⣬���ͳ���ļ��ı���

### ����Ҫ��

```bash
pip install matplotlib numpy
```

## �߼��÷�

### ��������
���Ա�д�ű��������ɲ�ͬ���õ�ͼ��

```bash
# �������е������ݼ���ͼ��
for dataset in mmlu nq hotpotqa triviaqa; do
    python draw_wikipead_all_chart.py --dataset $dataset
done

# �����������õı���
for topk in 1 5 10; do
    python generate_wikipead_analysis_report.py --topk $topk
done
```

### �Զ������
�����޸Ľű��е��ļ�·���������ʽ�������ض�����

---

��Щ����ר��������ڷ��� `wikipead_all.py` ��������ṩ�˱�ԭ�й��߸��ḻ�ķ���ά�Ⱥ͸���ϸ�Ķ��졣