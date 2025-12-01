# mnist-digit-classifier-using-deep-learning
## Sample MNIST Images

<img width="785" height="334" alt="59979373-bc052480-9604-11e9-85b8-464367fdc891" src="https://github.com/user-attachments/assets/629eacfc-7fb5-4eaa-905d-f42f7d373132" />
<img width="1121" height="1198" alt="download" src="https://github.com/user-attachments/assets/3f596c1c-b922-4669-bb48-f375db6316d6" />
<img width="655" height="325" alt="MNIST_dataset_example" src="https://github.com/user-attachments/assets/f0283603-2c67-4588-ad69-285419227090" />
<img width="389" height="411" alt="download (3)" src="https://github.com/user-attachments/assets/2083ba26-9680-4a75-812d-221ff73fc5b4" />
<img width="549" height="413" alt="download (2)" src="https://github.com/user-attachments/assets/d5ee59a9-69b9-4b8c-8d7a-114cf3a1a3d5" />


🧠 MNIST Handwritten Digit Classifier  
  A Deep Learning project using TensorFlow and Keras to classify handwritten digits (0–9) from the MNIST dataset.      

📌 Project Overview  
  This project builds a Deep Neural Network (DNN) to recognize handwritten digits.  
  It uses the MNIST dataset containing 70,000 grayscale images of digits, each of size 28×28 pixels.    

Model Architecture:    
Flatten Layer  
  Dense Layer (300 neurons, ReLU)  
  Dense Layer (100 neurons, ReLU)  
  Output Layer (10 neurons, Softmax)  
  Achieved 97–98% test accuracy.  
  
📂 Dataset  
MNIST Dataset (TensorFlow Keras built-in)  
60,000 training images  
10,000 test images  
Pixel values normalized (0–1)  

🧠 Model Architecture  
model = Sequential([
    Input(shape=(28,28)),
    Flatten(),
    Dense(300, activation='relu'),
    Dense(100, activation='relu'),
    Dense(10, activation='softmax')
])
  
⚙️ Training Details  
Loss Function: Sparse Categorical Crossentropy  
Optimizer: SGD  
Epochs: 30  
Batch Size: 32  
Validation Split: 5000 images  
  
📊 Results  
Metric Score  
Training Accuracy	~99%  
Validation Accuracy	~97–98%  
Test Accuracy	97.8%  
  
Sample Predictions:  
y_pred = np.argmax(model.predict(X_test[:3]), axis=1)
  
📈 Visualizations  
Training vs Validation Accuracy  
Training vs Validation Loss  
Confusion Matrix  
Class prediction visualization  
  
🚀 How to Run >>
Clone the repo:  
git clone https://github.com/your-username/mnist-digit-classifier.git
  

Install dependencies:  
pip install tensorflow matplotlib seaborn numpy pandas
Open the notebook:  
jupyter notebook MNIST_Digit_Classifier.ipynb
Run all cells.  

📦 Dependencies  
Python 3.8+  
TensorFlow  
NumPy  
Pandas  
Matplotlib  
Seaborn  
  
📝 Future Improvements   
Use CNN (Convolutional Neural Network) for ~99.3% accuracy.  
Add dropout for regularization.  
Convert model to TensorFlow Lite for mobile use.  
