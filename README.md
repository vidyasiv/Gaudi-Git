## Gaudi-v2: For recommendation models

This repository is designed for implementing and testing recommendation models using Gaudi-v2.

Currently, we have implemented three recommendation models:
- [**PMF**](https://github.com/Sein-Kim/Gaudi-Git/tree/main/MF-gaudi): A collaborative filtering-based recommendation model.
- [**SASRec**](https://github.com/Sein-Kim/Gaudi-Git/tree/main/SASRec-gaudi): Another collaborative filtering-based recommendation model.
- [**A-LLMRec**](https://github.com/Sein-Kim/Gaudi-Git/tree/main/A-LLMRec-gaudi): An LLM-based recommendation model.

Checklists for improving implementation of recommendation models
- [x] Distributed Data Parallel: A-LLMRec Stage 1
- [x] Distributed Data Parallel: A-LLMRec Stage 2
- [x] Distributed Data Parallel: A-LLMRec Inference
- [ ] Distributed Data Parallel: Automatically find `world_size` (initialize_distributed_hpu)
- [ ] `nn.Parameter` -> `nn.Embedding`