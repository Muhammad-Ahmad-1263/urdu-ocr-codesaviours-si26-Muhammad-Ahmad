Urdu OCR Project | Code Saviours SI-26 | Muhammad Ahmad

## About This Project
This repository contains my Week 1 to Week 5 work for the Urdu OCR (Optical Character Recognition) project, completed as part of the Code Saviours ML/AI Internship Programme, Batch SI-26.

## Week 1 — Research Summary

**What is OCR (Optical Character Recognition)?**
OCR is a technology that converts images of printed or handwritten text into machine-readable, editable text. It works by analyzing the shapes and patterns of characters in an image and matching them to known letterforms, then outputting the result as digital text a computer can process, search, or edit. OCR is widely used in applications like scanning documents, digitizing books, and reading text from photos.

**Why is Urdu OCR harder than English OCR?**
Urdu OCR is significantly harder than English OCR for several reasons. First, Urdu uses a cursive, connected script (Nastaliq or Naskh style) where letters change shape depending on their position in a word, unlike the mostly fixed letterforms in English. Second, Urdu is written right-to-left and often has overlapping diacritical marks and stacked characters, especially in Nastaliq script, which makes character segmentation much more difficult. Third, there is far less labeled Urdu training data publicly available compared to English, making it harder to train accurate models.

**What are 2 real-world situations where Urdu OCR would be useful?**
One real-world use case is digitizing historical Urdu literature, newspapers, and government archives so they can be searched, indexed, and preserved digitally instead of existing only as physical or scanned copies. Another use case is enabling accessibility tools, such as apps that read out Urdu signboards, documents, or handwritten notes for visually impaired users, or translation apps that need to first recognize Urdu text before translating it.

## Week 1 — Data Collection Summary
For this week's data collection task, I built a dataset of 137 labeled Urdu text images:
- 100 images sourced from an existing curated Urdu poetry dataset (Ghalib and Iqbal ghazals)
- 37 real screenshots of Urdu news headlines collected from a Pakistani news website

Images are organized into `data/raw/{newspaper, books, signboards, synthetic, other}`, with ground-truth text for every image stored in `data/labels.csv`.

## Repository Structure
```
├── SI26_Week1_Muhammad_Ahmad.ipynb   # Week 1 notebook (setup, research, data collection)
├── data/
│   ├── raw/
│   │   ├── newspaper/
│   │   ├── books/
│   │   ├── signboards/
│   │   ├── synthetic/
│   │   └── other/
│   └── labels.csv
└── README.md
```
# Week 2 – Image Preprocessing

## Overview
This week focused on building a robust image preprocessing pipeline for the Urdu OCR project. The objective was to standardize the dataset and improve image quality before model training. Preprocessing helps reduce noise, normalize image dimensions, and enhance text visibility, enabling the OCR model to learn more effectively from consistent input data.

## Implementation
A preprocessing pipeline was developed using Python and OpenCV to perform the following operations:

- Converted all images to grayscale to eliminate unnecessary color information.
- Resized every image to a fixed resolution of **512 × 128 pixels** for dataset consistency.
- Applied noise reduction using the **Fast Non-Local Means Denoising** algorithm.
- Performed binary thresholding to enhance text contrast and improve character visibility.
- Saved all processed images to the `data/processed/` directory while preserving the original dataset.

## Technologies Used
- Python
- OpenCV
- NumPy
- Pillow (PIL)
- Matplotlib

## Outcome
The preprocessing pipeline was successfully executed across the entire dataset, generating a standardized collection of processed images ready for the next phase of the Urdu OCR project. These preprocessed images provide cleaner, more consistent input for model training, which is expected to improve OCR performance and recognition accuracy.
