![Classified a Three](./cover.png)

# Number Detector
This simple program lets you draw numbers and the AI will try and classify the number!

# Usage
You can use the existing `model.pth`, or train your own:
```py
py train.py
```

Then, to run the program:
```py
py main.py
```

## Workflow
Use main.py, draw some numbers, label with your keyboard.
Then, run convert.py to add it to the json database.
Finally, run train.py to retrain.

## Research results
No significant difference between using the MNIST normalization values, and finding the mean and std programmatically. Perhaps they are the same? (The values aren't though)