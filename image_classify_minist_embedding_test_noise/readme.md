This floder gives the network with pixels embedding for images classification.
# enviroment
- python3
- tensorflow==1.14
- numpy
- and so on

# train the model
- The raw data is larger than 100Mb and we can't upload it to github.
  Therefore, you have to prepare the training dataset and put it in 
  the folder named train. There are few images in the folder as examples.

- You also have to prepare the test dataset,
  and put it in the folder named noise_test. 
  if you want to test on images with different types of noises, you 
  can just replace the noise_test folder.
  
- run 'python3 main.py --train True' to train the model. 
  - The folder with name ckpt_1 will be generated. 
  - The file with name  log_1.log will be generated.
  - The file with name label2id will be generated in the train folder.


