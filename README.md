# hotpaper-text

һ������RAG��������ǿ���ɣ����ĵ��ȶȷֲ��������ߣ����ڷ�����ͬ��ѯ���ݼ����ĵ�������Ƶ�ʷֲ�ģʽ��

## ? ��Ŀ���

����Ŀʵ����һ��������RAGϵͳ�������ߣ������о��ڲ�ͬ��ѯ���ݼ��ϼ���Wikipedia�ĵ�ʱ���ȶȷֲ����ɡ�ʹ���Ƚ���BAAI/bge-large-en-v1.5Ƕ��ģ�ͺ�FAISS HNSW�����������и�Ч�����������ͳ�Ʒ�����

## ? ���Ĺ���

- ? **�����ݼ�֧��**: ֧��MMLU��Natural Questions��HotpotQA��TriviaQA�������ʴ����ݼ�
- ? **Ƶ�ʷ���**: ͳ���ĵ�����Ƶ�ʷֲ���ʶ��"��β�ֲ�"ģʽ
- ? **����Top-K**: ֧�ֲ�ͬ�ļ�����������
- ? **���ӻ�**: ����Log-Log�߶ȵ�Ƶ�ʷֲ�ͼ��
- ? **��Ч����**: ʹ��FAISS HNSW����ʵ�ֿ����������ƶ�����
- ? **��Ϸ���**: �����ĵ���ϵ�������������ģʽ
- ? **�����ݼ��Ա�**: ֧�ֶ����ݼ�����Աȷ���

## ?? ����ջ

| ��� | ���� | ˵�� |
|:---|:---|:---|
| **Ƕ��ģ��** | BAAI/bge-large-en-v1.5 | 1024ά��רΪRAG�Ż����ı�Ƕ��ģ�� |
| **�������ݿ�** | FAISS HNSW | ��Ч�Ľ���������������� |
| **֪ʶ��** | rag-mini-wikipedia | Wikipedia���Ӽ����� |
| **���ӻ�** | Matplotlib | ����Log-Log�߶ȵ�Ƶ�ʷֲ�ͼ |
| **���ݴ���** | Pandas, NumPy | ���ݼ��������ֵ���� |
| **���ݼ�����** | datasets | HuggingFace datasets�� |

## ? ��װ��ʹ��

### ����Ҫ��

- Python 3.7+
- �Ƽ��ڴ� 8GB+
- NumPy 1.x�汾������FAISS��

### ������װ

```bash
pip install -r requirements.txt
```

### ���Ľű�ʹ��

#### 1. �ĵ�Ƶ�ʷ��� (`hot.py`)

���������ĵ��ڼ����е��ȶȷֲ���

```bash
# �����÷�
python hot.py --dataset [mmlu|nq|hotpotqa|triviaqa] --topk [K]

# ʾ������
python hot.py --dataset mmlu --topk 1       # MMLU���ݼ���Top-1�ĵ�
python hot.py --dataset nq --topk 5         # Natural Questions��Top-5�ĵ�
python hot.py --dataset hotpotqa --topk 10  # HotpotQA��Top-10�ĵ�
python hot.py --dataset triviaqa --topk 1   # TriviaQA��Top-1�ĵ�
```

#### 2. �ĵ���Ϸ��� (`hotpair.py`)

�����ĵ���ϵ�������������ģʽ��

```bash
# �����÷�
python hotpair.py --dataset [mmlu|nq|hotpotqa|triviaqa] --topk [K]

# ʾ������
python hotpair.py --dataset mmlu --topk 3
python hotpair.py --dataset nq --topk 5
```

#### 3. �����ݼ�ͼ������ (`draw_combo_chart.py`)

���ɿ����ݼ��ĶԱȷ���ͼ��

```bash
python draw_combo_chart.py  # �Զ���ȡ�����ɵ�ͳ���ļ�
```

### ֧�ֵ����ݼ�

| ���ݼ� | ��ʶ�� | ���� |
|:---|:---|:---|
| MMLU | `mmlu` | Massive Multitask Language Understanding |
| Natural Questions | `nq` | Google��Ȼ�������ݼ� |
| HotpotQA | `hotpotqa` | ���������ʴ����ݼ� |
| TriviaQA | `triviaqa` | �ٿ�֪ʶ�ʴ����ݼ� |

### ����ļ�˵��

#### ���ĵ�������� (`hot.py`)

- `output/charts/hot_docs_distribution_{dataset}_top{k}.png`: �ĵ�Ƶ�ʷֲ����ӻ�ͼ��
- `data/stats/freq_stats_{dataset}_top{k}.txt`: ��ϸƵ��ͳ����Ϣ������Top-10�����ĵ�

#### ��Ϸ������ (`hotpair.py`)

- `output/charts/ordered_combo_distribution_{dataset}_top{k}.png`: �������Ƶ�ʷֲ�ͼ
- `output/charts/unordered_combo_distribution_{dataset}_top{k}.png`: �������Ƶ�ʷֲ�ͼ
- `data/stats/ordered_combo_stats_{dataset}_top{k}.txt`: �������ͳ����Ϣ
- `data/stats/unordered_combo_stats_{dataset}_top{k}.txt`: �������ͳ����Ϣ

#### ϵͳ�ļ�

- `doc_embeddings.npy`: Wikipedia�ĵ�Ƕ�루�״��������ɣ�
- `hnsw_index.bin`: FAISS HNSW�����ļ����״��������ɣ�

## ? ��Ŀ�ṹ

```
hotpaper-text/
������ data/stats/           # ͳ�ƽ���ļ���
��   ������ ordered_combo_stats_*.txt
��   ������ unordered_combo_stats_*.txt
������ output/charts/        # ͼ������ļ���
������ hot.py               # ���ķ����ű�
������ hotpair.py           # ��Ϸ����ű�
������ draw_chart.py        # �����ݼ�ͼ�����
������ draw_combo_chart.py  # �����ݼ��Ա�ͼ��
������ dateset.py           # ���ݼ�����ģ��
������ requirements.txt     # Python����
������ commands.txt         # ʾ������
������ README.md           # ��Ŀ�ĵ�
```

## ? Ӧ�ó���

�����߿��԰����о��ߣ�

1. **ʶ�����ż����ĵ�**: ����RAGϵͳ�б�Ƶ��������Wikipedia�ĵ�
2. **�����ֲ�ģʽ**: ͨ��Log-Logͼ�۲��Ƿ���ѭ��β�ֲ�
3. **��������������**: �˽��������ļ��г̶�
4. **�Ż�RAGϵͳ**: �����ȶȷֲ����ɸĽ���������
5. **�����ݼ��Ƚ�**: ������ͬ�ʴ����ݼ��ļ������Բ���

## ? ����ԭ��

### ��������

```mermaid
flowchart TD
    A[����Wikipedia֪ʶ��] --> B[����/�����ĵ�Ƕ��]
    B --> C[����/����FAISS HNSW����]
    C --> D[���ز�ѯ���ݼ�]
    D --> E[ִ��Top-K����]
    E --> F[ͳ���ĵ�Ƶ��]
    F --> G[����Ƶ�ʷֲ�ͼ]
    G --> H[���ͳ�Ʊ���]
```

### �����㷨

1. **����Ƕ��**: ʹ��BAAI/bge-large-en-v1.5ģ�ͽ��ĵ��Ͳ�ѯת��Ϊ1024ά����
2. **���ƶȼ���**: ����FAISS HNSW�������и�Ч�Ľ������������
3. **Ƶ��ͳ��**: ͳ��ÿ���ĵ�ID�����в�ѯ��������еĳ���Ƶ��
4. **�ֲ�����**: ʹ��Log-Log����ϵ���ӻ�Ƶ�ʷֲ���ʶ��β����

## ?? ע������

- **�״�����**: ��Ҫ���ز�����Wikipedia���ݼ���Ƕ�룬Ԥ����Ҫ�ϳ�ʱ��
- **�ڴ�Ҫ��**: Ƕ���ļ��������ļ��ϴ�Լ25MB+�����Ƽ�8GB+�ڴ滷��
- **NumPy�汾**: ����ʹ��NumPy 1.x�汾����ΪFAISSģ����NumPy 2.x������
- **ģ�ͻ���**: ����ʹ�ñ��ػ����HuggingFaceģ�ͣ�������ڣ�
- **�ļ�����**: ����ͳ���ļ�����ʹ��GB2312���룬��ȡʱ��ע��

## ? �����ų�

### ��������

1. **NumPy�汾��ͻ**: ������NumPy 1.x�汾
2. **�ڴ治��**: ����ϵͳ�ڴ��ʹ�ø�С�����ݼ�
3. **ģ������ʧ��**: ����������ӻ�ʹ�ñ���ģ�ͻ���
4. **������ʾ����**: ȷ����װ��SimHei��Microsoft YaHei����

## ? ���֤

����Ŀʹ��MIT���֤ - ���LICENSE�ļ���

## ? ����

��ӭ�ύIssue��Pull Request���Ľ���Ŀ��

## ? ��ϵ

����������飬��ͨ��GitHub Issues��ϵ��