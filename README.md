# Spam Email Detection using Python & Flask

## Overview
This project is a **Spam Email Detection system** built using **Python** and **Flask**.  
It uses **Naive Bayes** and **CountVectorizer** to classify emails as **Spam** or **Not Spam**.  

Users can also access a **web page interface** to test custom emails.

---

## Features
- Detects spam emails using machine learning.
- Preprocesses messages to remove stopwords and special characters.
- Provides a **web interface** to input custom emails.
- Easy to extend with new datasets or models.

---

## Tech Stack
- **Python 3**
- **Flask** (Web app)
- **scikit-learn** (Machine learning)
- **pandas** (Data handling)
- **NLTK** (Text preprocessing)

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Himani-N/spam-email-detection.git
Go to the project folder:

bash

cd spam-email-detection
Install required packages:

bash

pip install -r requirements.txt






How to Use
1. Run the ML model (optional, for testing in console)


bash

python main.py
This will train the Naive Bayes model and test a sample email.

2. Run the Flask web app
bash

python app.py
Open your browser and go to: http://127.0.0.1:5000

Enter an email message and check if it is Spam or Not Spam.

Sample Emails to Test
Spam emails:

"Congratulations! You have won a free prize. Click now"

"You are selected for a $1000 reward. Reply immediately"

Not Spam emails:

"Hi, can we meet tomorrow for the project discussion?"

"Please find attached the report for last month."

Dataset
Dataset used: spam.csv

Contains labeled messages: ham (not spam) and spam.

File Structure
css

spam_email_detection/
├── spam.csv
├── main.py
├── app.py
├── templates/
│   └── index.html
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
License
This project is licensed under the MIT License.

Author
Himani Nimmakayala

yaml
Copy code

---

This README is **complete, professional, and beginner-friendly**.  

✅ It includes:

- Overview of the project  
- Features  
- Installation instructions  
- How to run the model and web app  
- Sample emails to test  
- File structure  
- License and author info  

---
