# REST-MEX 25'
#### Sentiment Analysis and Classification of Mexican Tourist Reviews

@Iberlef25'

---

## 🏆 The challenge

Given a dataset of Spanish-language reviews about Mexican tourist destinations, your task is to:

- ✅ Determine the sentiment polarity on a scale from 1 (very negative) to 5 (very positive).
- ✅ Classify the type of destination (hotel, restaurant, or attraction).
- ✅ Identify the specific Magical Town from a list of 60 locations.

Perks

- 🔹 Access to a dataset with 200,000+ real reviews.
- 🔹 A unique opportunity to apply and test state-of-the-art NLP models.
- 🔹 Publication of results in **CEUR-WS** as part of IberLEF 2025.
- 🔹 Collaboration with academia and industry on a challenge with real-world impact on the tourism sector.

---

## 📂 Project Structure

### **1. Data**

- Contains datasets for training and testing, such as `train.csv` and `test.csv`, tokenized and embedding files.

### **2. Models**

- **Path**: `/models/baseline`
- Contains a baseline model for sentiment analysis and classification tasks using ML algorithms.

- **Path**: `/models/tabularisai-distilbert/`
- Includes a fine-tuned DistilBERT tokenizer and vocabulary files (`tokenizer.json`, `vocab.txt`) for processing Spanish-language text.

- **Path**: `/models/embeddings/`

- Contains ML models for sentiment analysis and classification tasks using `Alibaba-NLP/gte-multilingual-base` embeddings.

### **3. Code**

#### **3.1 Baseline**

- **Path**: `/baseline/`
- Contains the baseline model for sentiment analysis and classification tasks using ML algorithms.

#### **3.2 DistilBERT**

- **Path**: `/transformers-sentiment/`
- Contains the code for training and evaluating the model using a fine-tuned DistilBERT tokenizer and vocabulary files (`tokenizer.json`, `vocab.txt`) for processing Spanish-language text.

#### **3.3 Embeddings**

- **Path**: `/transformers-embeddings/`
- Contains the code for training and evaluating the model using `Alibaba-NLP/gte-multilingual-base` embeddings.

#### **3.4 Utils**

- **Path**: `/utils/config.py`
- setConfig(): Sets GPU or CPU configuration for training and evaluation.
- **Path**: `/utils/metrics.py`
- Calculates the REST-MEX evaluation metrics for sentiment analysis and classification tasks.
- **Path**: `/utils/run.py`
- Main function to run the training and evaluation process.

---

### Usage 

0. Prerequisites

- Miniforge or Anaconda installation.

```bash
# To install Miniforge in Unix-like platforms (macOS, Linux, & WSL)
wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b
```

1. Clone the repository:

```bash
git clone https://github.com/roicort/REST-MEX25.git
cd REST-MEX25
```

2. Install the required dependencies:

```bash
conda env create -f environment.yml
conda activate restmex
```

3. Install GPU dependencies

This script will detect your operating system, 
the device (MPS, CUDA or CPU), CUDA version (If applicable), 
and install the required dependencies.

```bash
bash install-transformers.sh
```

4. Run eval script

```bash
python main.py 
```

---

### Notes

- Some models where trained using M3 Pro GPU trough MPS (Metal Performance Shaders) while others were trained using a NVIDIA RTX6000 ADA GPU.
- The code is compatible with both CPU and GPU (CIUDA or MPS) configurations. 
- Just make sure to verify the correct device is being used.

---

### License

- This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
- Please refer to other licenses for third-party libraries and models used in this project.
- Please note that the dataset is for research purposes only and should not be used for commercial applications without permission.
- The dataset is provided "as is" without any warranties or guarantees of any kind.
- The authors are not responsible for any damages or losses resulting from the use of the dataset.
- If this is useful for you, please consider citing the paper:

```bibtex
@inproceedings{NLque?-restmex2025,
  title={REST-MEX 2025: Sentiment Analysis and Classification of Mexican Tourist Reviews},
  author={Rodrigo S. Cortez-Madrigal},
  booktitle={CEUR Workshop Proceedings},
  year={2025},
  publisher={CEUR-WS.org}
}
```

---

## 🤝 Organized by

- 🏛️ CIMAT Monterrey
- 🏛️ Universidad de Guanajuato (UG) Salamanca
- 🏛️ Universidad Iberoamericana, Mexico City
- 🏛️ CENATAV, Havana, Cuba

---

🌟 An opportunity to contribute to the future of smart tourism in Mexico! 🇲🇽✨

#RESTMex2025 #IberLEF #NLP #SentimentAnalysis #Tourism #ArtificialIntelligence #DataScience #CIMAT #UG #Ibero #CENATAV