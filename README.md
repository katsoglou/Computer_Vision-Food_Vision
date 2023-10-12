# Computer_Vision-Food_Vision
A Python project powered by TensorFlow for food recognition using computer vision techniques.

# Food Vision with Python and TensorFlow

This project is a demonstration of food recognition using computer vision techniques. It's implemented in Python, making use of the powerful TensorFlow library. 

Dataset: food101

## Introduction

Computer vision has a wide range of applications, and one exciting area is food recognition. This project showcases my ability to create a robust food recognition system using Python and TensorFlow. It can be a valuable addition to your computer vision projects, whether you're exploring the world of deep learning or food identification.

## Features

- Food item recognition: The system is capable of identifying and classifying various food items.
- Python and TensorFlow: The project is implemented in Python and leverages the TensorFlow library for machine learning and deep learning tasks.
- Easy to use: You can use and customize this project for your specific food recognition needs.

## Installation

To run this project locally, follow these steps:

1. Clone the repository to your local machine.
3. Install the required Python libraries and dependencies.
4. Run the food recognition script.

## Usage

from the file download the food_vision_model.h5. it is already trained with 75% accuracy
Load the model: food_vision_model.h5
```python
model = tf.keras.models.load_model("food_vision_model.h5")
```
make a folder with the food images you want to predict and then make a list with the images paths
```python
my_food_images = ["my_images/" + img_path for img_path in os.listdir("my_images")]
```
Load the classes
```python
with open("class_names.json", "r") as file:
    class_names = json.load(file)
```
At the end make predictions
```python
for img in my_food_images:
    img = load_and_prep_image(img, scale=False)  # Load in target image and turn it to tensor
    pred_prop = model.predict(tf.expand_dims(img, axis=0)) # make prediction on image with shape [None, 224, 224, 3]
    pred_class = class_names[pred_prop.argmax()] # find the predicted class label
    # Plot the images with appropriate annotations
    plt.figure()
    plt.imshow(img/255.)  # imshow() requires float inputs to be normalized
    plt.title(f"pred: {pred_class}, prob: {pred_prop.max():.2f}")
    plt.axis(False)
```
### License
This project is licensed under the [MIT License](https://chat.openai.com/c/LICENSE).

### Contact
If you have any questions or suggestions, feel free to reach out to me at [m.katsogloy@mail.com](mailto:m.katsogloy@gmail.com).
