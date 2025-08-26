# hotpaper-text

һ������RAG��������ǿ���ɣ����������·ֲ��������ߣ����ڷ�����ͬ���ݼ����ĵ�������Ƶ�ʷֲ�ģʽ��

## ��Ŀ����

����Ŀʵ����һ��������RAGϵͳ�����ڷ����ڲ�ͬ��ѯ���ݼ��ϼ���Wikipedia�ĵ�ʱ�����Ŷȷֲ�����ʹ����BAAI/bge-large-en-v1.5Ƕ��ģ�ͺ�FAISS HNSW���������и�Ч������������

## ��Ҫ����

- ? **�����ݼ�֧��**: ֧��MMLU��Natural Questions��HotpotQA��TriviaQA�ȳ����ʴ����ݼ�
- ? **Ƶ�ʷ���**: �������������ĵ�Ƶ�ʷֲ���ʶ��"��������"����
- ? **������Top-K**: ֧�ֲ�ͬ�ļ�����������
- ? **���ӻ�**: ����Log-Log�߶ȵ�Ƶ�ʷֲ�ͼ
- ? **��Ч����**: ʹ��FAISS HNSW�������п�������������

## ����ջ

- **Ƕ��ģ��**: BAAI/bge-large-en-v1.5 (1024ά��רΪRAG�Ż�)
- **��������**: FAISS HNSW����
- **֪ʶ��**: rag-mini-wikipedia (Wikipedia�Ӽ�)
- **���ӻ�**: Matplotlib
- **���ݴ���**: Pandas, NumPy

## ��װ��ʹ��

### ����Ҫ��

```bash
pip install -r requirements.txt
```

### ����ʹ��

```bash
# ʹ��MMLU���ݼ�������Top-1�ĵ�
python hot.py --dataset mmlu --topk 1

# ʹ��Natural Questions���ݼ�������Top-5�ĵ�
python hot.py --dataset nq --topk 5

# ʹ��HotpotQA���ݼ�������Top-10�ĵ�
python hot.py --dataset hotpotqa --topk 10

# ʹ��TriviaQA���ݼ�������Top-1�ĵ�
python hot.py --dataset triviaqa --topk 1
```

### ֧�ֵ����ݼ�

- `mmlu`: MMLU (Massive Multitask Language Understanding)
- `nq`: Natural Questions
- `hotpotqa`: HotpotQA
- `triviaqa`: TriviaQA

### ����ļ�

������ɺ�����������ļ���

- `hot_docs_distribution_{dataset}_top{k}.png`: Ƶ�ʷֲ����ӻ�ͼ
- `freq_stats_{dataset}_top{k}.txt`: ��ϸ��Ƶ��ͳ����Ϣ
- `doc_embeddings.npy`: Wikipedia�ĵ�Ƕ�루�״��������ɣ�
- `hnsw_index.bin`: FAISS HNSW�����������״��������ɣ�

## �������

�ù��߿��԰�������

1. **����������������**: ʶ����RAGϵͳ�б�Ƶ��������Wikipedia����
2. **�����ֲ�ģʽ**: ͨ��Log-Logͼ�۲��Ƿ�������ɷֲ�
3. **��������������**: �˽��������ļ��г̶�
4. **�Ż�RAGϵͳ**: �������Ŷȷֲ�������������

## ʾ������

�ο� `commands.txt` �ļ��е�ʾ�����

```bash
python hot.py --dataset mmlu --topk 1
python hot.py --dataset nq --topk 5
python hot.py --dataset hotpotqa --topk 10
python hot.py --dataset triviaqa --topk 1
```

## ע������

- �״�������Ҫ���غʹ���Wikipedia���ݼ���������Ҫ�ϳ�ʱ��
- Ƕ���ļ��������ļ��ϴ�~25MB�����״����ɺ�᱾�ػ���
- �Ƽ������㹻�ڴ�Ļ��������У�����8GB+��

## ���֤

����Ŀ����MIT���֤ - ���LICENSE�ļ���