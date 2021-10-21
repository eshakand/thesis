# Introduction
The following is an attempt to process all full resolution images including cc and mlo only based on whether they are benign or malignant. Such images are uncleaned, and at the time of this attempt were using a resolution of 2900 by 4000. While such attempts presented an accuracy of high 60 percent. The efforts were abandoned in favor of building a model for cc and mlo views individually. It was during this process that we experimented with different optimizers and landed on NAdam optimizer. Similarly, as we go down in the number of convolutional layers, the model is able to memorize the input (overfit) and thus produces negative validation results. Finally, Batch Normalization was introduced after each convolutional layer to normalize the output, but this did not provide fruitful decisions. 