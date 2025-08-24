# Resume Classification System using DistilBERT  

## Project Overview  
This project implements a *resume classification system* that automatically categorizes resumes into different job roles using *DistilBERT*, a transformer-based deep learning model.  
The model achieves *78% accuracy*, making it effective for filtering and organizing resumes across diverse domains.  

## Features  
- Collected and preprocessed resumes from diverse sources.  
- Leveraged *DistilBERT* for robust text classification.  
- End-to-end pipeline: data collection → preprocessing → training → evaluation.  
- Achieved *78% accuracy* on classification tasks.  

##  Tools & Technologies  
- *Python*  
- *Hugging Face Transformers (DistilBERT)*  
- *Scikit-learn, Pandas, NumPy*  
- *GitHub API, Google Gemini API*  
- *Jupyter Notebook* for development & experiments  

##  Dataset  
- Resumes collected from:  
  - *Public GitHub repositories*  
  - *Synthetic data generation* (to balance categories)  
  - *Multiple online sources*  
- Data preprocessing included cleaning, tokenization, and formatting for DistilBERT input.  

## Results  
- Model Accuracy: *78%*  
- Classification across multiple job roles/domains(44 Categories).  

##  Installation & Usage  
```bash
# Clone repository
git clone https://github.com/Sooraj12Sreekumar/Resume-Classification-System-Using-BERT.git
cd resume-classification
```

# Install dependencies
```bash
pip install -r requirements.txt
```
# Run training and evaluation
```bash
python -m model_BERT.train
```
```bash
python -m model_BERY.evaluate
```
