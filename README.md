# Cats and Dogs Classifier Using Flask and GCP
This project was to reinforce machine learning concepts and to start getting into the world of MLOPs.

To run this, I made a GCP educational account and uploaded the project files. Then I built the docker image using 'docker build -t kys2020/kys2020:catdog5.' Push it using 'docker push.'

Then use 'kubectl apply -f' commands on the two .yaml files, and on the workloads tab on GCP the endpoint should be exposed, allowing you to upload an image to be classified.

This was a really quick development, so the CSS styling is horrid. It would be worth it to go back and try to make the page look better.

