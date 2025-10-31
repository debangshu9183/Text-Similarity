# 🧠 Text Similarity using Transformer (TensorFlow)

This project demonstrates how to build a **text similarity model** using a **Transformer-based neural network** built from scratch in TensorFlow/Keras.  
It determines whether two sentences/questions convey the same meaning — similar to the **Quora Question Pairs** problem.

---

## 🚀 Features
- Uses **custom Transformer Encoder blocks** (Multi-Head Attention + Feed Forward).
- Learns semantic similarity between text pairs.
- Trains on the **Quora Question Pairs dataset**.
- Computes **cosine similarity** between encoded representations.
- Visualizes **accuracy and loss** over training epochs.
- Supports **custom text input** for similarity prediction.

---

## 📂 Dataset
The model is trained on the **Quora Question Pairs** dataset (`questions.csv.zip`),  
containing pairs of questions labeled as *duplicate* (1) or *not duplicate* (0).

Example:
| question1 | question2 | is_duplicate |
|------------|------------|--------------|
| How to invest in stock market? | What are the steps to invest in shares? | 1 |

---

## 🏗️ Model Architecture
1. **Tokenizer & Padding:** Convert text to sequences and pad them.  
2. **Positional Encoding:** Adds sequential context to embeddings.  
3. **Transformer Encoder:** Multi-head attention + feedforward network.  
4. **Cosine Similarity Layer:** Measures semantic closeness.  
5. **Sigmoid Output:** Predicts probability of duplication.

---

## 🧩 Model Summary
- Embedding dimension (`d_model`): 64  
- Transformer heads (`num_heads`): 4  
- Feed-forward dimension (`dff`): 128  
- Sequence length (`maxlen`): 50  
- Vocabulary size: 10,000  

---

## 🧪 Training
```bash
python main.py
