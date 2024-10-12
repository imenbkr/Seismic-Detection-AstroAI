# **Seismic Detection Across the Solar System**  
*Nasa SpaceApps Challenge 2024* | 🥉Achieved 3rd place out of 23 teams for the local Sfax event🥉 

## Project Overview
This project focuses on developing a **machine learning model** integrated with a **user-friendly web application** to detect seismic events in planetary data. Specifically, we worked with [seismic data](https://www.spaceappschallenge.org/nasa-space-apps-2024/challenges/seismic-detection-across-the-solar-system/?tab=resources) from NASA's **Apollo missions** and the **InSight Lander** to tackle the challenge of identifying seismic signals in noisy datasets. Our goal is to improve the accuracy of seismic event detection on other planets while optimizing data transmission.

## Features
- Upload `.mseed` seismic data files.
- View file metadata and visualize seismic waveforms in an interactive plot.
- Predict and classify seismic events using a machine learning model.
- Interactive threshold controls for fine-tuned seismic detection and noise reduction using the STA/LTA algorithm.

## Machine Learning Techniques
Our application leverages **machine learning models** to classify seismic events and filter noise. The model is selected based on the best **F1 score** and utilizes **SMOTE** (Synthetic Minority Over-sampling Technique) to handle class imbalance in seismic event data.  
Key features include:
- **Seismic Event Prediction**: Identify and classify seismic events.
- **Event Timing**: Pinpoint the start time of detected events to enhance analysis.

## Tech Stack
- **Python**: Core language.
- **Libraries**: Scikit-learn, Pandas, Streamlit for web app and data processing.
- **Machine Learning**: Model training and evaluation for seismic event classification.

## Visit our app!
OUr app is hosted on the cloud. You can visit it following this [Link1](https://astroai.streamlit.app/) or following this other [Link2](https://seismic-detection.onrender.com/)
Explore our app by uploading your own `.mseed` files or using the provided example. The app will allow you to view metadata, visualize waveforms, and identify seismic events.

**[Example .mseed file for input:](https://drive.google.com/file/d/1v-gJv-d8BdZARd6r53zdkR03G3WWbp3r/view?usp=drive_link)**

#### There is also a demo video of our app [Youtube Link](https://www.youtube.com/watch?v=jMFm2mk1L2g)

## How to Run the Application
1. Clone the repository:  
   ```bash
   git clone https://github.com/imenbkr/Seismic-Detection-AstroAI.git
   
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the App:
   ```bash
   streamlit run app.py

### Future Improvements
- Support for additional planetary missions and datasets.
- Real-time data streaming from spacecrafts or landers.
- Improved model accuracy with more advanced machine learning techniques.

### Screenshots from our application:

![image](https://github.com/user-attachments/assets/564f2c99-8818-4f12-8440-644bc5917e41)

![image](https://github.com/user-attachments/assets/764237e6-d74b-4a32-8b53-f42303c1d24e)

![image](https://github.com/user-attachments/assets/83e81be2-2c3f-4708-b867-46e165d3af34)

![image](https://github.com/user-attachments/assets/45a78130-3362-41f9-a766-212e153038a1)

![image](https://github.com/user-attachments/assets/e9bdc400-2869-4794-892e-a68740904d59)

![image](https://github.com/user-attachments/assets/29d484aa-306f-4531-b033-6db1775ac7b3)

![image](https://github.com/user-attachments/assets/97748749-455d-4a68-a209-d5fb85a569ef)

![image](https://github.com/user-attachments/assets/c6f396f8-d8d4-4239-9552-c117b1323420)

![image](https://github.com/user-attachments/assets/476bb4ed-d01d-49a0-b1bf-832bf3ccb262)

![image](https://github.com/user-attachments/assets/36646ccc-7a39-42b1-84f7-957bf6cbb408)


---
### Link to team page on Nasa SpaceApps Website: [AstroAI](https://www.spaceappschallenge.org/nasa-space-apps-2024/find-a-team/nasai/?tab=details)
#### 🚀 Team members: 🚀
- Imen Bakir, Islem Ben Moalem, Ons ABida.
