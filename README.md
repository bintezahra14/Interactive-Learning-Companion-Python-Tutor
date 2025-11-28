# Interactive Learning Companion – Python Tutor
**Course:** ITAI-2376 – AI Agent Systems  
**Team:** Team 1 – Interactive Learning Companion  
**Members:**  
- Binte Zahra  
- Waseem  

---

## 1. Project Overview

This repository contains the implementation of an AI agent system called the **Interactive Learning Companion**, a Python programming tutor that adapts to individual learners.

The agent:
- Answers Python questions at different difficulty levels
- Generates explanations and examples
- Uses tools to execute Python code and perform web search (stub)
- Maintains a learner profile and adjusts difficulty over time
- Includes safety, input validation, and RL-style feedback and policy improvement

---

## 2. Repository Structure

```text
interactive-learning-companion/
├── README.md                  # Setup & usage instructions
├── requirements.txt           # Python dependencies
├── config_example.json        # Example config (no secrets)
├── src/
│   ├── memory.py              # MemorySystem & LearnerProfile classes
│   ├── tools.py               # Tool implementations (run_python, web_search_stub)
│   ├── agent.py               # LearningAgent (ReAct + RL-style logic)
│   └── main.py                # CLI entry point / demo loop
├── notebooks/
│   └── FN_Notebook_BinteZahra_Waseem_Team1_ITAI2376.ipynb  # Colab notebook with experiments
└── docs/
    └── architecture_diagram.png  # (optional) System architecture diagram
