# Cats and Dogs Classifier Using Flask and GCP
This project was to reinforce machine learning concepts and to start getting into the world of MLOPs.

To run this, I made a GCP educational account and uploaded the project files from the 'inference' folder. Then I built the docker image using 'docker build -t kys2020/kys2020:catdog5.' Push it using 'docker push.'

Training was done on Google Colab with a single GPU. The training files can be found in the 'training' folder.

THE DATA MUST BE IN A FOLDER NAMED 'data' IN THE SAME DIRECTORY AS THE TRAINING FILES! You can find and download the dataset from here: https://www.microsoft.com/en-us/download/details.aspx?id=54765

Then use 'kubectl apply -f' commands on the two .yaml files, and on the workloads tab on GCP the endpoint should be exposed, allowing you to upload an image to be classified.

This was a really quick development, so the CSS styling is horrid. It would be worth it to go back and try to make the page look better.

