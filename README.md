# 4th Year Thesis
***
This repository is used to demonstrate the Python scripts and Jupyter notebooks that are associated with my 4th year thesis. Listed below is an abstract of the paper. 

***
This main aim of this paper will look at the effectiveness of convolutional neural networks for classifying mobile malware. Mobile malware was specifically chosen due to the development of these devices and the increase of everyday tasks which can be carried out on mobile phones.

Malware is a big problem and by developing ways to perform classification with a convolutional neural network, this can help aid a security professionals' approach to distinguish between the different types of malware found on mobile devices.

Differentiating between the various families associated with a type of malware is important to develop an understanding into the threat level a specific type of family may possess as well as mitigations which can be used to prevent the malware from spreading or causing more damage to a device.

By converting the classes.dex file found in APKs to a greyscale image, this paper proposes a novel approach by investigating classification through the use of a convolutional neural network using a modern, up to date dataset.

The approach used for this experimentation consists of a convolutional neural network and the Android Malware (CIC-InvesAndMal2019) dataset, which used 305 samples in total for the purpose of this experimentation. In total the average results achieved across the top 3 results for the adware samples was 0.70, the ransomware was the best performer in terms of accuracy, averaging an accuracy of .83 and the scare was the worst performer out of the 3 types of family with an average of 0.63.