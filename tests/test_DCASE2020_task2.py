from mtsa import (
    HitachiDCASE2020,
    
)
from mtsa.utils import files_train_test_split_dcase2020_task2

from mtsa.metrics import (
    calculate_aucroc
)

path = '/data/DCASE2020/dev/fan'
X_train, X_test, y_train, y_test = files_train_test_split_dcase2020_task2(path, pattern='id_0[0|4|6]')

model = HitachiDCASE2020()

model.fit(X_train, y_train)

roc = calculate_aucroc(model, X_test, y_test)

print(roc)