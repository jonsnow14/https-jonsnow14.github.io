---
layout: post
title: "How RAG Works: Part I"
date: 2026-06-18
categories: [AI, RAG, Deep-Dive]
---

*An introduction to Retrieval-Augmented Generation — why plain LLMs fall short, and how RAG fixes it by retrieving before generating.*

---

## 1. What is RAG?

Imagine you ask a knowledgeable friend a question. They know a lot in general, but if you ask something very specific — like what's in a particular research paper or a private company document — they'd need to look it up first. RAG gives an AI model the same ability: **look something up before answering**.

---

## 2. The Problem with Plain LLMs

A large language model (LLM) like Gemma is trained on a massive snapshot of text from the internet. That training gives it broad general knowledge, but it has a hard cutoff — it knows nothing about documents you wrote yesterday, your company's internal data, or anything outside its training window.

More critically, when an LLM doesn't know something, it doesn't say "I don't know." It generates a plausible-sounding answer anyway. These confident but wrong answers are called **hallucinations**.

```
User question ──► LLM ──► Response
                          (may be wrong, outdated, or fabricated — no source to check)
```

---

## 3. The RAG Solution: Retrieve Before You Generate

RAG (Retrieval-Augmented Generation) fixes this by giving the LLM access to a document store at query time. Instead of relying solely on memorized training data, the model is shown the most relevant excerpts from your actual documents before it answers.

```
Your documents ──► Index (chunk → embed → store in vector DB)
                                        │
User question ──► Embed ──► Retrieve ───┘
                             top-k chunks
                                 │
                    Augmented Prompt = question + retrieved chunks
                                 │
                                LLM ──► Grounded response
```

The response is now grounded in real text you provided — text you can trace back to a source. The LLM doesn't hallucinate facts that aren't in your documents; it works with what it's shown.

---

## 4. The RAG Pipeline, Step by Step

```
─────────────────── INDEXING (done once at setup) ───────────────────

   ┌──────────┐     ┌──────────────┐     ┌──────────────┐
   │          │     │    Embed     │     │    Vector    │
   │ Sources  ├────►│   Sources   ├────►│    Store    │
   │    ①    │     │      ②      │     │      ③      │
   └──────────┘     └──────────────┘     └──────┬───────┘
                                                 │
─────────────────── QUERYING (every question) ───┼────────────────────
                                                 │
   ┌──────────┐   ┌──────────────┐   ┌──────▼───────┐   ┌──────────────┐   ┌──────────┐
   │          │   │    Embed     │   │              │   │  Retrieved   │   │          │
   │  Prompt  ├──►│   Prompt    ├──►│  Retriever  ├──►│    Text     ├──►│   LLM    ├──► Response
   │    ④    │   │      ⑤      │   │      ⑥      │   │    ⑦  +     │   │    ⑧    │
   └────┬─────┘   └──────────────┘   └─────────────┘   └──────────────┘   └──────────┘
        │                                                                        ▲
        └──────────────────── original prompt (passed directly) ────────────────┘
```

| # | Step | What happens |
|---|---|---|
| ① | Gather Sources | Collect documents — PDFs, policies, reports — that will serve as the knowledge base |
| ② | Embed Sources | Pass each text chunk through an embedding model, converting it to a fixed-length numeric vector that captures its meaning |
| ③ | Store Vectors | Save the vectors in a vector database optimized for similarity search |
| ④ | Obtain User Prompt | Receive the user's question |
| ⑤ | Embed User Prompt | Embed the question using the same model as step ②, so it lives in the same vector space |
| ⑥ | Retrieve Relevant Data | Find the top-k vectors in the store closest to the question vector; return those text chunks |
| ⑦ | Create Augmented Prompt | Combine the retrieved chunks with the original prompt into a single enriched input |
| ⑧ | Obtain Response | Feed the augmented prompt to the LLM; it generates an answer grounded in the retrieved context |

RAG has two phases: **indexing** (done once) and **retrieval + generation** (done on every query).

---

## 5. Phase 1 — Indexing

### Step 1: Load your document

The source document — a PDF, a web page, a text file — is loaded and converted to raw text. A loader like `PyPDFLoader` reads each page of a PDF and passes the text downstream.

### Step 2: Split into chunks

Long documents are broken into smaller overlapping chunks (e.g., 500-word chunks with 50-word overlap). This is essential because retrieval compares the user's question against individual chunks, not the entire document.

**Smaller chunks = more precise matching.**

### Step 3: Embed each chunk

Every chunk is passed through an **embedding model** — a neural network that converts text into a fixed-length list of numbers called a vector. The key property: semantically similar text produces numerically similar vectors.

A common local model is `all-MiniLM-L6-v2`, which produces 384-dimensional vectors and runs fully on your machine.

### Step 4: Store in a vector database

The vectors are saved in a **vector store** — a database optimized for similarity search over thousands of vectors in milliseconds. Popular options:

| Vector Store | Type |
|---|---|
| ChromaDB | In-memory / local |
| FAISS | In-memory / local |
| Milvus | Self-hosted |
| Pinecone | Managed cloud |

---

## 6. Phase 2 — Retrieval + Generation

### Step 5: Embed the user's question

The incoming question is embedded using the **same model** used during indexing. This puts the question and the stored chunks in the same vector space so they can be meaningfully compared.

### Step 6: Retrieve the top-k most relevant chunks

The vector store finds chunks whose vectors are **closest to the question vector** (via cosine similarity). The top-k chunks are returned — not the whole document, just the most relevant passages.

### Step 7: Build the augmented prompt

The retrieved chunks are combined with the user's original question to form a single input — the **augmented prompt**. A structured template tells the LLM how to use the context:

```
Use the following context to answer the question.
If the answer isn't in the context, say you don't know — don't guess.

Context:
  [chunk 1 text]
  [chunk 2 text]
  [chunk 3 text]
  [chunk 4 text]

Question: What is the main finding of this paper?
```

Frameworks like LangChain handle this automatically via `RetrievalQA` and `ConversationalRetrievalChain` — they retrieve the chunks, slot them into a `PromptTemplate`, and pass the final string to the LLM, so you never have to assemble the prompt manually.

### Step 8: Send to the LLM and get a response

The augmented prompt is sent to the LLM. The model reads the retrieved context and generates a response **grounded in your document** rather than its training memory.

---

## Summary

| Phase | What happens | When it runs |
|---|---|---|
| Indexing | Load → chunk → embed → store | Once, at setup |
| Retrieval | Embed question → find top-k chunks | Every query |
| Generation | Build augmented prompt → send to LLM | Every query |

The core insight of RAG is simple: **don't ask the model to remember — ask it to read first, then answer.** By grounding every response in retrieved text, you get answers that are traceable, up-to-date, and far less likely to hallucinate.

---

*Part II will cover: chunking strategies, embedding model choices, re-ranking, and how to evaluate retrieval quality.*
