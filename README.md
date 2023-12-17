# Drowsiness-Yawn-checker

Simple code in python to detect Drowsiness and Yawn and alert the user using Dlib.
This is the source code for the Facial Drowsiness and Yawn Detection software for the CPS 843 project. The following tutorial was used to construct this code: https://youtu.be/ksi42rwGyas?si=ql8kzDZ4QWgokIYB

## Dependencies

1. Python 3
2. opencv
3. dlib
4. imutils
5. scipy
6. numpy
7. argparse

## Run 

```
Python3 drowsiness_yawn.py -- webcam 0		//For external webcam, use the webcam number accordingly
```

## Setups

Change the threshold values according to your need
```
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 15`	//change this according to the distance from the camera
```

## Authors

**Harsanjam Saini** 
**Ahmed Ramadan** 
**Syed Rameel Arshad** 


## Acknowledgments

* https://www.pyimagesearch.com/

* https://www.sciencedirect.com/science/article/pii/S2468227623004611![image](https://github.com/Harsanjam/Drowsiness-Yawn-checker/assets/56749102/4c644932-30fd-47f8-9ef8-f5e8a7401f0c)




