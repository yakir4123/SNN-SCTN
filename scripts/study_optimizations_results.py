import numpy as np
import optuna
import yaml
from matplotlib import pyplot as plt

if __name__ == '__main__':
    learns = [
        # (104, 5, 72),
        (2777, 3, 10, 375),
        # (3395, 3, 8),
        # (4365, 3, 6),
        # (6111, 3, 4),
        # (5555, 2, 10),
        # (6790, 2, 8),
        # (8730, 2, 6),
        # (12223, 2, 4)
    ]
    start_freq = 0
    f_pulse = 1.536 * (10 ** 6)
    with open("../secret.yaml", 'r') as stream:
        secrets = yaml.safe_load(stream)

    storage = f'postgresql://{secrets["USER"]}:{secrets["PASSWORD"]}@{secrets["ENDPOINT"]}:{secrets["PORT"]}/{secrets["DBNAME"]}'

    for freq0, LF, LP, lobe_wide in learns:
        study_name = f'Study-{freq0}-{LF}-{LP}-{lobe_wide}'
        study = optuna.create_study(study_name=study_name,
                                    storage=storage,
                                    load_if_exists=True)
        print(study.best_params)