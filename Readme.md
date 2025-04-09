# Deepfake Video Detector

A minimalistic Django web application that detects deepfake videos using a CNN-based machine learning model.

## Features

- Clean, modern UI with Tailwind CSS
- Real-time deepfake probability analysis
- Comprehensive result display with confidence metrics
- Processes videos frame-by-frame with facial detection
- Responsive design for all devices

## Technology Stack

- **Backend**: Django 4.2
- **Frontend**: Tailwind CSS (via CDN)
- **ML Model**: CNN-based deepfake detector (MaanVad3r/DeepFake-Detector)
- **Face Detection**: MTCNN via facenet-pytorch
- **Video Processing**: OpenCV
- **Image Analysis**: TensorFlow/Keras

## Installation

### Prerequisites

- Python 3.9+
- Git (optional)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/deepfake-detector.git
   cd deepfake-detector
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run database migrations:
   ```bash
   python manage.py makemigrations detector
   python manage.py migrate
   ```

5. Ensure the CNN model file is in the correct location:
   - The default location is `D:\DeepFake-Detector\cnn_model.h5`
   - Alternatively, place it in `detector/ml_models/cnn_model.h5`

6. Start the development server:
   ```bash
   python manage.py runserver
   ```

7. Access the application at http://127.0.0.1:8000/

## Usage

1. Upload a video through the web interface (supported formats: MP4, AVI, MOV, MKV)
2. Wait for the analysis to complete
3. View the results showing the probability of the video being a deepfake
4. Use the "Analyze New Video" button to test additional videos

## How It Works

1. **Video Upload**: The application accepts video files up to 100MB.
2. **Frame Extraction**: Frames are extracted from the video at regular intervals.
3. **Face Detection**: MTCNN detects faces in each frame.
4. **Deepfake Analysis**: Each detected face is analyzed by the CNN model.
5. **Result Aggregation**: Multiple scoring techniques are combined for a final deepfake probability.
6. **Result Display**: A clean interface shows the verdict with visual indicators.

## Model Details

The application uses the MaanVad3r/DeepFake-Detector CNN model with the following specifications:
- Input Size: 128x128 pixels
- Architecture: Convolutional Neural Network (CNN)
- Accuracy: 71% on test dataset
- Binary Classification: Real vs Fake

## Customization

### UI Modifications

The application uses Tailwind CSS via CDN. You can modify the UI by editing the HTML templates in the `detector/templates/detector/` directory.

### Model Replacement

To use a different deepfake detection model:
1. Create a new TensorFlow/Keras model
2. Save it as an H5 file
3. Update the path in `detector/ml_models/deepfake_detector.py`

## Limitations

- The model has a 71% accuracy rate according to its documentation
- Very short videos or videos without clear faces may produce less reliable results
- Processing time depends on video length and hardware capabilities

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MaanVad3r for the DeepFake-Detector model
- The Django and TensorFlow communities
- FaceNet-PyTorch for the MTCNN implementation