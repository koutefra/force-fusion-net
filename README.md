# 🧠 ForceFusionNet

> **Official implementation of the paper**  
> **“Deep Learning Approach to Force-Based Modeling of Pedestrian Flow in Bottleneck Scenarios”**  
> *František Dušek, Pavel Hrabák, Daniel Vašata*  
> Faculty of Information Technology, Czech Technical University in Prague, 2025

---

<p align="center">
  <img src="demo.gif" width="85%" alt="ForceFusionNet simulation on b160 bottleneck">
</p>

<p align="center">
  <em>ForceFusionNet simulation of pedestrian flow through a 1.6 m bottleneck (b160 scene).</em>
</p>

---

### 🧩 Overview
**ForceFusionNet** is a hybrid force-based model for simulating pedestrian flow through bottlenecks.  
It integrates **Graph Neural Network–inspired aggregation** into the classical **Social Force Model (SFM)**,  
learning goal-directed, interaction, and obstacle forces directly from real-world trajectory data.

Compared to purely knowledge-based or deep models, ForceFusionNet achieves:
- 💡 *Interpretability* — each force module corresponds to a physical component  
- ⚙️ *Physical realism* — dynamics governed by second-order equations of motion  
- 🚶‍♂️ *Predictive power* — accurately reproduces unseen bottleneck scenes

---

## ⚡ Quick Start

### 1️⃣ Clone the repository
```bash
git clone https://github.com/koutefra/force-fusion-net.git
cd force-fusion-net
```

---

### 2️⃣ Create and activate the environment
```bash
conda env create -f environment.yml
conda activate force-fusion-net
```

---

### 3️⃣ Download the dataset
Make the script executable and run it:
```bash
chmod 755 ./data/datasets/download.sh
./data/datasets/download.sh
```

This will automatically fetch the **Jülich Bottleneck Caserne Dataset**  
and extract it into:
```
data/datasets/julich_bottleneck_caserne/
```

---

### 4️⃣ Train the model
Train either the **DirectForceNet** or **ForceFusionNet** model:

```bash
python src/train.py \
  --model_type fusion_net \
  --batch_size 64 \
  --epochs 5 \
  --pred_steps 32 \
  --device cuda
```

---

### 5️⃣ Evaluate performance
Use pretrained weights or your trained model:

```bash
python src/evaluate.py \
  --scene_file b160.txt \
  --scene_name b160 \
  --model_folder ./data/weights \
  --model_file fusion_net_2025-01-06_194434.pth \
  --model_type fusion_net \
  --device cuda
```

---

### 6️⃣ Visualize trajectories and create videos
```bash
python src/create_vid.py \
  --scene_file b160.txt \
  --scene_name b160 \
  --model_folder ./data/weights \
  --model_file fusion_net_2025-01-06_194434.pth \
  --model_type fusion_net \
  --simulation_steps 300 \
  --time_scale 2.0
```

---

## 🧹 Optional: Skip large result files

The `results/` folder contains precomputed plots, metrics, and videos  
for all bottleneck scenes. It’s **not required** for running or reproducing the project.  
If you want to **clone the repository faster**, you can skip these large files using Git’s sparse checkout:

```bash
git clone --filter=blob:none --sparse https://github.com/koutefra/force-fusion-net.git
cd force-fusion-net
git sparse-checkout set --no-cone data src environment.yml
```

This will only download the essential files (`data/`, `src/`, `environment.yml`).  
You can always fetch the rest later if needed:
```bash
git sparse-checkout set results
```

---

## 🧰 Repository Structure
```
force-fusion-net/
├── data/
│   ├── datasets/
│   │   └── download.sh
│   └── weights/
│       ├── fusion_net_2025-01-06_194434.pth
│       ├── direct_net_2025-10-07_111752_epoch1.pth
│       └── sfm.json
├── src/
│   ├── data/
│   ├── entities/
│   ├── evaluation/
│   ├── models/
│   ├── utils/
│   ├── train.py
│   ├── evaluate.py
│   └── create_vid.py
├── results/
│   ├── b090/
│   ├── b160/
│   │   ├── gt/
│   │   ├── fusion_net/
│   │   ├── direct_net/
│   │   └── sfm_160/
├── demo.gif
└── environment.yml
```

## 🧭 Contact
For questions or suggestions, feel free to open an issue or [reach out](https://fit.cvut.cz/en/faculty/people/19200-frantisek-dusek)


