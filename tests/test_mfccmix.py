from mtsa import MFCCMix
from mtsa.utils import (
    files_train_test_split
)
from mtsa.metrics import (
    calculate_aucroc
)

model = MFCCMix()

path = "/data/MIMII/fan/id_00"

X_train, X_test, y_train, y_test = files_train_test_split(path)

model.fit(X_train, y_train)

roc = calculate_aucroc(model, X_test, y_test)

print(roc)
