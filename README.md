# LearnAct: Few-Shot Mobile GUI Agent with a Unified Demonstration Benchmark

<div align="center">
  <img src="assets/teaser-final.drawio.png" alt="LearnAct Framework" width="100%">
</div>

[📄 Paper](https://arxiv.org/abs/2504.13805) | [🤗 Dataset](https://huggingface.co/datasets/lgy0404/LearnGUI) | [🌐 Project Page](https://lgy0404.github.io/LearnAct/)

## Overview

Mobile GUI agents show promise in automating tasks but face generalization challenges in diverse real-world scenarios. Traditional approaches using pre-training or fine-tuning with massive datasets struggle with the diversity of mobile applications and user-specific tasks. We propose enhancing mobile GUI agent capabilities through human demonstrations, focusing on improving performance in unseen scenarios rather than pursuing universal generalization through larger datasets.

To realize this paradigm, we introduce LearnGUI, the first comprehensive dataset specifically designed for studying demonstration-based learning in mobile GUI agents. It comprises 2,252 offline tasks and 101 online tasks with high-quality human demonstrations. We further develop LearnAct, a sophisticated multi-agent framework that automatically extracts knowledge from demonstrations to enhance task completion. This framework integrates three specialized agents: DemoParser for knowledge extraction, KnowSeeker for relevant knowledge retrieval, and ActExecutor for demonstration-enhanced task execution.

## 📊 LearnGUI Benchmark

Our framework leverages the LearnGUI benchmark, the first comprehensive dataset specifically designed for studying demonstration-based learning in mobile GUI agents. It comprises:

- 2,252 offline tasks and 101 online tasks with high-quality human demonstrations
- Coverage of 73 diverse mobile applications
- Average of 13.2 steps per task
- Rich few-shot learning support with k-shot combinations (k=1,2,3)
- Multi-dimensional similarity metrics across instruction, UI, and action dimensions

### Comparison with Existing Datasets

LearnGUI offers several advantages over existing GUI datasets:

| Dataset                   | # Inst.         | # Apps       | # Step         | Env. | HL | LL | GT | FS |
| ------------------------- | --------------- | ------------ | -------------- | ---- | -- | -- | -- | -- |
| PixelHelp                 | 187             | 4            | 4.2            | ❌   | ✅ | ❌ | ✅ | ❌ |
| MoTIF                     | 276             | 125          | 4.5            | ❌   | ✅ | ✅ | ✅ | ❌ |
| UIBert                    | 16,660          | -            | 1              | ❌   | ❌ | ✅ | ✅ | ❌ |
| UGIF                      | 523             | 12           | 6.3            | ❌   | ✅ | ✅ | ✅ | ❌ |
| AITW                      | 30,378          | 357          | 6.5            | ❌   | ✅ | ❌ | ✅ | ❌ |
| AITZ                      | 2,504           | 70           | 7.5            | ❌   | ✅ | ✅ | ✅ | ❌ |
| AndroidControl            | 15,283          | 833          | 4.8            | ❌   | ✅ | ✅ | ✅ | ❌ |
| AMEX                      | 2,946           | 110          | 12.8           | ❌   | ✅ | ❌ | ✅ | ❌ |
| MobileAgentBench          | 100             | 10           | -              | ❌   | ✅ | ❌ | ❌ | ❌ |
| AppAgent                  | 50              | 10           | -              | ❌   | ✅ | ❌ | ❌ | ❌ |
| LlamaTouch                | 496             | 57           | 7.01           | ✅   | ✅ | ❌ | ✅ | ❌ |
| AndroidWorld              | 116             | 20           | -              | ✅   | ✅ | ❌ | ❌ | ❌ |
| AndroidLab                | 138             | 9            | 8.5            | ✅   | ✅ | ❌ | ❌ | ❌ |
| **LearnGUI (Ours)** | **2,353** | **73** | **13.2** | ✅   | ✅ | ✅ | ✅ | ✅ |

*Note: # Inst. (number of instructions), # Apps (number of applications), # Step (average steps per task), Env. (supports environment interactions), HL (has high-level instructions), LL (has low-level instructions), GT (provides ground truth trajectories), FS (supports few-shot learning).*

### Dataset Statistics

| Split         | K-shot | Tasks | Apps | Step actions | Avg Ins `<sub>`Sim `</sub>` | Avg UI `<sub>`Sim `</sub>` | Avg Act `<sub>`Sim `</sub>` | UI `<sub>`SH `</sub>`Act `<sub>`SH `</sub>` | UI `<sub>`SH `</sub>`Act `<sub>`SL `</sub>` | UI `<sub>`SL `</sub>`Act `<sub>`SH `</sub>` | UI `<sub>`SL `</sub>`Act `<sub>`SL `</sub>` |
| ------------- | ------ | ----- | ---- | ------------ | ------------------------------- | ------------------------------ | ------------------------------- | --------------------------------------------------- | --------------------------------------------------- | --------------------------------------------------- | --------------------------------------------------- |
| Offline-Train | 1-shot | 2,001 | 44   | 26,184       | 0.845                           | 0.901                          | 0.858                           | 364                                                 | 400                                                 | 403                                                 | 834                                                 |
| Offline-Train | 2-shot | 2,001 | 44   | 26,184       | 0.818                           | 0.898                          | 0.845                           | 216                                                 | 360                                                 | 358                                                 | 1,067                                               |
| Offline-Train | 3-shot | 2,001 | 44   | 26,184       | 0.798                           | 0.895                          | 0.836                           | 152                                                 | 346                                                 | 310                                                 | 1,193                                               |
| Offline-Test  | 1-shot | 251   | 9    | 3,469        | 0.798                           | 0.868                          | 0.867                           | 37                                                  | 49                                                  | 56                                                  | 109                                                 |
| Offline-Test  | 2-shot | 251   | 9    | 3,469        | 0.767                           | 0.855                          | 0.853                           | 15                                                  | 42                                                  | 55                                                  | 139                                                 |
| Offline-Test  | 3-shot | 251   | 9    | 3,469        | 0.745                           | 0.847                          | 0.847                           | 10                                                  | 36                                                  | 49                                                  | 156                                                 |
| Online-Test   | 1-shot | 101   | 20   | 1,423        | -                               | -                              | -                               | -                                                   | -                                                   | -                                                   | -                                                   |

## 🚀 LearnAct Framework

The LearnAct framework consists of three specialized agents:

1. **DemoParser**: Extracts usable knowledge from demonstration trajectories
2. **KnowSeeker**: Retrieves demonstration knowledge relevant to current tasks
3. **ActExecutor**: Combines instructions, GUI environment, and demonstration knowledge

<div align="center">
  <img src="assets/learnact-pipline.drawio.png" alt="LearnAct Pipeline" width="80%">
</div>

## 📈 Benchmark Results

Our experimental results show significant performance gains in both offline and online evaluations:

### Offline Evaluation

| Models           | Method   | Supports | Average      | Gmail | Booking | Music | SHEIN | NBC  | CityMapper | ToDo | Signal | Yelp |
| ---------------- | -------- | -------- | ------------ | ----- | ------- | ----- | ----- | ---- | ---------- | ---- | ------ | ---- |
| SPHINX-GUI Agent | AMEX     | 0-shot   | 67.2         | 45.9  | 64.5    | 74.4  | 71.8  | 70.3 | 67.4       | 79.3 | 64.9   | 66.3 |
| gemini-1.5-pro   | Baseline | 0-shot   | 19.3         | 20.1  | 16.4    | 24.5  | 10.2  | 35.6 | 14.1       | 17.4 | 27.9   | 15.2 |
| gemini-1.5-pro   | LearnAct | 1-shot   | 51.7 [+32.4] | 55.5  | 47.1    | 60.0  | 35.7  | 56.4 | 54.7       | 60.6 | 63.1   | 54.6 |
| gemini-1.5-pro   | LearnAct | 2-shot   | 55.6 [+36.3] | 57.5  | 53.2    | 55.3  | 39.6  | 56.1 | 58.2       | 68.1 | 69.7   | 60.0 |
| gemini-1.5-pro   | LearnAct | 3-shot   | 57.7 [+38.4] | 58.4  | 56.6    | 54.6  | 43.9  | 53.9 | 69.4       | 69.2 | 70.5   | 57.6 |
| UI-TARS-7B-SFT   | Baseline | 0-shot   | 77.5         | 68.1  | 81.0    | 81.1  | 72.9  | 80.9 | 70.6       | 66.0 | 92.6   | 82.4 |
| UI-TARS-7B-SFT   | LearnAct | 1-shot   | 82.8 [+5.3]  | 79.9  | 82.9    | 86.6  | 75.7  | 86.3 | 79.4       | 84.0 | 89.3   | 83.0 |
| UI-TARS-7B-SFT   | LearnAct | 2-shot   | 81.9 [+4.4]  | 80.1  | 80.7    | 86.2  | 76.1  | 87.2 | 80.0       | 83.7 | 84.4   | 84.2 |
| UI-TARS-7B-SFT   | LearnAct | 3-shot   | 82.1 [+4.6]  | 79.9  | 80.9    | 86.2  | 75.7  | 86.9 | 81.2       | 85.8 | 84.4   | 84.2 |
| Qwen2-VL-7B      | Baseline | 0-shot   | 71.8         | 60.8  | 73.9    | 76.0  | 65.5  | 75.5 | 62.9       | 78.7 | 82.8   | 69.1 |
| Qwen2-VL-7B      | LearnAct | 1-shot   | 77.3 [+5.5]  | 75.0  | 77.5    | 77.8  | 69.8  | 83.5 | 72.9       | 78.0 | 83.6   | 78.8 |
| Qwen2-VL-7B      | LearnAct | 2-shot   | 78.5 [+6.7]  | 75.0  | 78.0    | 77.8  | 73.3  | 86.0 | 73.5       | 81.9 | 87.7   | 77.6 |
| Qwen2-VL-7B      | LearnAct | 3-shot   | 79.4 [+7.6]  | 75.0  | 78.8    | 78.6  | 72.6  | 87.8 | 77.1       | 82.6 | 87.7   | 80.6 |

### Online Evaluation

| Input          | Models                    | # Params | LearnGUI-Online SR |
| -------------- | ------------------------- | -------- | ------------------ |
| Image + AXTree | GPT-4o                    | -        | 34.5               |
| Image + AXTree | Gemini-Pro-1.5            | -        | 22.8               |
| Image          | Claude Computer-Use       | -        | 27.9               |
| Image          | Aguvis                    | 72B      | 26.1               |
| Image          | Qwen2-VL-7B + 0-shot      | 7B       | 9.9                |
| Image          | Qwen2-VL-7B + LearnAct    | 7B       | 21.1 [+11.2]       |
| Image          | UI-TARS-7B-SFT + 0-shot   | 7B       | 18.1               |
| Image          | UI-TARS-7B-SFT + LearnAct | 7B       | 32.8 [+14.7]       |

Key highlights:

- Gemini-1.5-Pro accuracy increases from 19.3% to 51.7% (a 198.9% relative improvement)
- CityMapper app accuracy improves from 14.1% to 69.4%
- To-Do apps accuracy increases from 17.4% to 69.2%
- UI-TARS-7B-SFT's task success rate improves from 18.1% to 32.8% with LearnAct
- Our 7B parameter models enhanced with LearnAct match or exceed the performance of much larger commercial models

## 🔗 Citation

If you find our work helpful, please consider citing our paper:

```bibtex
@article{liu2025learnact,
  title={LearnAct: Few-Shot Mobile GUI Agent with a Unified Demonstration Benchmark},
  author={Liu, Guangyi and Zhao, Pengxiang and Liu, Liang and Chen, Zhiming and Chai, Yuxiang and Ren, Shuai and Wang, Hao and He, Shibo and Meng, Wenchao},
  journal={arXiv preprint arXiv:2504.13805},
  year={2025}
}
```

## 📄 License

This dataset is licensed under Apache License 2.0.
