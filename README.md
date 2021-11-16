# Facial-Expression-Detection
This program detects the emotion of a person (or people) from their facial expression.<br>
In order to run the detection program, run `main.py` in *src* directory without any parameters. The program will ask you to choose the display mode or train mode. If you choose the display mode, the program will detects your face and emotion according to your facial expression. On the other hand, if you choose the train mode, then you will be able to train the model. Just make sure that you specify the right directories for training when you are running the program in the train mode. You can do this by changing the variable `train_dir` and `val_dir`.

## Dependencies
To install the dependencies, run `pip install -r requirements.txt`.
It is possible that the program could work with different dependencies. 

## Sampling
Class imbalance exists in the data set that was used to train the model (<a href="https://www.kaggle.com/aadityasinghal/facial-expression-dataset">Check here for the dataset</a>). This issue was fixed by oversampling the data.

## Future improvements
<ul>
  <li>The convolutional neural network used in this project is still shallow. In the future, it is possible to make the model better by adding more layers and a few more things such as dropout and normalization layer. </li>
  <li>When oversampling, some data seems to break, which worsen the performance of the model. This has to be fixed somehow in the future to improve the accuracy of the model.</li>
</ul>

## Reference
<ul>
  <li><a href="https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml">haarcascade_frontalface_default.xml</a></li>
</ul>
