# LyfeStyle-TF
Machine Learning RecoSys for Sleep/Exercise

## Database
PMData

https://dl.acm.org/doi/10.1145/3339825.3394926

https://www.kaggle.com/datasets/vlbthambawita/pmdata-a-sports-logging-dataset?resource=download

## Network
This simple tensorflow-based neural network will consist of 3 hidden layers and 1 dropout layer to compensate for the low amount of data and resource.
Input -> 32 -> 64 -> Dropout -> 32 -> Output

From the dataset, we extract only the "good" sleeps to be used for RecoSys for sleep. As exercise and diet requires a more complicated calculations we won't be using machine learning for them.

To make the process simpler we'll use the following for the input:
- Gender
- Age
- Height
- Total time of exercise and type of exercise
- Last sleep total
- Last sleep end time

and the output:
- sleep start time
- sleep duration

Such calculation will be saved in the database, and will be recalculated whenever additonal exercise is done.
