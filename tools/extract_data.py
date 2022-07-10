import numpy as np


def GetSpacedElements(array, numElems=4):
    out = array[np.round(np.linspace(0, len(array) - 1, numElems)).astype(int)]
    return out


if __name__ == "__main__":

    arr = np.arange(17)
    print(arr)
    spacedArray = GetSpacedElements(arr, 5)
    print(spacedArray)
