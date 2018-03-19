from itertools import product
from collections import OrderedDict
from Trainer import Trainer

class Hyperparameter:
    def __init__(self, model_params, trainer_params, model_definition, x, y, test_x, test_y, log=True):
        """
        model_params and trainer_params to be of the form:
            {param_name: [value1, value2, ..., valuen]}
        model_definition is a function that takes the parameters and returns the model.
        """
        self.x = x
        self.y = y
        self.log = log
        self.test_x = test_x
        self.test_y = test_y
        self.model_definition = model_definition
        self.model_params = OrderedDict(Hyperparameter.replace_single_items(model_params))
        self.trainer_params = OrderedDict(Hyperparameter.replace_single_items(trainer_params))
        self.create_trainer_keys = {"model", "learning_rate", "momentum", "CUDA"}


    def replace_single_items(obj):
        for key in obj:
            if type(obj[key]) != list:
                obj[key] = [obj[key]]
        return obj


    def train(self):
        results = []
        if self.log:
            step = 1

        for model_values in product(*self.model_params.values()):
            model_params = {
                key: model_values[index] for index, key in enumerate(self.model_params)
            }

            for trainer_values in product(*self.trainer_params.values()):
                model = self.model_definition(model_params)

                trainer_params = {
                    key: trainer_values[index] for index, key in enumerate(self.trainer_params)
                }
                trainer_params["model"] = model
                
                create_trainer_params = {
                    key: value for key, value in trainer_params.items() if key in self.create_trainer_keys
                }
                            
            
                training_params = {
                    key: value for key, value in trainer_params.items() if key not in self.create_trainer_keys
                }
                if 'log' not in training_params:
                    training_params['log'] = False
                training_params["train_x"] = self.x
                training_params["train_y"] = self.y
                trainer = Trainer(**create_trainer_params)
                trainer.train(**training_params)
                train_score = trainer.test(self.x, self.y)
                test_score = trainer.test(self.test_x, self.test_y)
                
                score_object = {"test_accuracy": test_score, "train_accuracy": train_score}
                for index, model_key in enumerate(self.model_params):
                    score_object[model_key] = model_values[index]

                for index, trainer_key in enumerate(self.trainer_params):
                    score_object[trainer_key] = trainer_values[index]

                results.append(score_object)

                if self.log:
                    print(step, score_object)
                    step += 1
        
        return results

                
