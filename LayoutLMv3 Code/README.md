How to use LayoutLM to create training data afer Label Studio annotations.

In label studio, after dataset labelling are complete, export as json-min

in "labelstudio_convert_to_layoutlm_code.py", it will convert the json to layoutlm usable format.

Using "layoutlm_trainingcode.py", start training the ai model. configuration of traning can be changed (i.e. how many datasets used for training).

run "LayoutLMv3_model.py" with a sample resume not previously used to training for validation. Remember to change the path at "image_path" line 11

* this environment uses Python 3.9.13 
